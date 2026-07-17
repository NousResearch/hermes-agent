"""Coding verification evidence ledger.

This module records what the agent actually proved while working in a code
workspace. It is deliberately passive: it never decides to run a suite, never
blocks completion, and never upgrades targeted checks into "repo green".
"""

from __future__ import annotations

import configparser
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home


_DB_LOCK = threading.Lock()
_MAX_OUTPUT_SUMMARY_CHARS = 2000
_MAX_EVIDENCE_AGE_DAYS = 30
_MAX_EVENTS_PER_SESSION_ROOT = 100
_MAX_TOTAL_UNREFERENCED_EVENTS = 10_000
_MAX_ARTIFACT_PATHS = 10_000
_MAX_ARTIFACT_HASH_BYTES = 512 * 1024 * 1024
_MAX_GIT_QUERY_BYTES = 64 * 1024 * 1024
_AD_HOC_SCRIPT_NAME_PREFIXES = ("hermes-verify-", "hermes-ad-hoc-")
_VERIFY_SCHEMA_VERSION = 2
_SHELL_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;)\s*")


@dataclass(frozen=True)
class VerificationEvidence:
    """A classified command result worth recording."""

    command: str
    canonical_command: str
    kind: str
    scope: str
    status: str
    exit_code: int
    cwd: str
    root: str
    session_id: str
    output_summary: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retention_cutoff() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=_MAX_EVIDENCE_AGE_DAYS)).isoformat()


def _db_path() -> Path:
    return get_hermes_home() / "verification_evidence.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            session_id TEXT NOT NULL,
            cwd TEXT NOT NULL,
            root TEXT NOT NULL,
            command TEXT NOT NULL,
            canonical_command TEXT NOT NULL,
            kind TEXT NOT NULL,
            scope TEXT NOT NULL,
            status TEXT NOT NULL,
            exit_code INTEGER NOT NULL,
            output_summary TEXT NOT NULL,
            artifact_hash TEXT,
            changed_paths_json TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    event_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(verification_events)")
    }
    if "artifact_hash" not in event_columns:
        conn.execute("ALTER TABLE verification_events ADD COLUMN artifact_hash TEXT")
    if "changed_paths_json" not in event_columns:
        conn.execute(
            "ALTER TABLE verification_events "
            "ADD COLUMN changed_paths_json TEXT NOT NULL DEFAULT '[]'"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_state (
            session_id TEXT NOT NULL,
            root TEXT NOT NULL,
            last_event_id INTEGER,
            last_edit_at TEXT,
            changed_paths_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (session_id, root)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_verification_events_session_root
        ON verification_events(session_id, root, id DESC)
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(_VERIFY_SCHEMA_VERSION),),
    )
    conn.commit()


def _split_segment_tokens(command: str) -> list[list[str]]:
    segments: list[list[str]] = []
    for segment in _SHELL_SPLIT_RE.split(command.strip()):
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            continue
        if tokens:
            segments.append(tokens)
    return segments


def _clean_token(token: str) -> str:
    token = token.strip()
    while token.startswith("./"):
        token = token[2:]
    return token


def _canonical_tokens(canonical: str) -> list[str]:
    try:
        return [_clean_token(t) for t in shlex.split(canonical) if t]
    except ValueError:
        return []


def _find_subsequence(tokens: list[str], needle: list[str]) -> Optional[int]:
    if not tokens or not needle or len(needle) > len(tokens):
        return None
    cleaned = [_clean_token(t) for t in tokens]
    for idx in range(0, len(cleaned) - len(needle) + 1):
        if cleaned[idx:idx + len(needle)] == needle:
            return idx
    return None


def _strip_command_prefix(tokens: list[str]) -> list[str]:
    """Remove harmless command prefixes before matching canonical commands."""
    remaining = list(tokens)
    if remaining and remaining[0] == "env":
        remaining = remaining[1:]
    while remaining and "=" in remaining[0] and not remaining[0].startswith("-"):
        remaining = remaining[1:]
    while remaining and remaining[0] in {"command", "time", "noglob"}:
        remaining = remaining[1:]
    return remaining


def _equivalent_needles(needle: list[str]) -> list[list[str]]:
    """Return command spellings equivalent to the detected canonical command."""
    candidates = [needle]
    if len(needle) >= 3 and needle[1] == "run":
        package_manager = needle[0]
        script_name = needle[2]
        if package_manager in {"npm", "pnpm", "yarn", "bun"}:
            candidates.append([package_manager, script_name])
    if len(needle) == 1 and "/" in needle[0]:
        candidates.extend([["bash", needle[0]], ["sh", needle[0]]])
    if needle == ["pytest"]:
        candidates.extend(
            [
                ["python", "-m", "pytest"],
                ["python3", "-m", "pytest"],
                ["uv", "run", "pytest"],
                ["poetry", "run", "pytest"],
                ["pipenv", "run", "pytest"],
            ]
        )
    return candidates


def _find_canonical_match(command: str, canonical_commands: list[str]) -> Optional[tuple[str, list[str]]]:
    """Return ``(canonical, trailing_args)`` for the first detected command."""

    segments = _split_segment_tokens(command)
    for canonical in canonical_commands:
        needle = _canonical_tokens(canonical)
        if not needle:
            continue
        for tokens in segments:
            candidate_tokens = _strip_command_prefix(tokens)
            normalized_tokens = list(candidate_tokens)
            if normalized_tokens:
                executable = normalized_tokens[0]
                is_absolute_executable = Path(executable).is_absolute() or bool(
                    re.match(r"^[A-Za-z]:[\\/]", executable)
                )
                if is_absolute_executable:
                    normalized_tokens[0] = executable.replace("\\", "/").rsplit("/", 1)[-1]
                    if normalized_tokens[0].casefold().endswith(".exe"):
                        normalized_tokens[0] = normalized_tokens[0][:-4]
            for candidate in _equivalent_needles(needle):
                if normalized_tokens[:len(candidate)] == candidate:
                    return canonical, candidate_tokens[len(candidate):]
    return None


def _kind_for_command(canonical: str) -> str:
    lowered = canonical.lower()
    if any(word in lowered for word in ("lint", "eslint", "ruff")):
        return "lint"
    if any(word in lowered for word in ("typecheck", "tsc", "mypy", "pyright", "ty")):
        return "typecheck"
    if "build" in lowered:
        return "build"
    if "fmt" in lowered or "format" in lowered:
        return "format"
    if "check" in lowered and "test" not in lowered:
        return "check"
    return "test"


def _looks_like_target(arg: str) -> bool:
    if not arg or arg.startswith("-") or "=" in arg:
        return False
    return (
        "/" in arg
        or "\\" in arg
        or "::" in arg
        or arg.endswith((".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java"))
        or arg.startswith(("test_", "tests", "spec", "__tests__"))
    )


def _scope_for_args(args: list[str]) -> str:
    return "targeted" if any(_looks_like_target(arg) for arg in args) else "full"


def _is_under_temp_dir(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    try:
        path = Path(token).expanduser()
        if not path.is_absolute():
            return False
        resolved = path.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        return resolved == temp_root or temp_root in resolved.parents
    except Exception:
        return False


def _is_under_root(token: str, root: str | Path | None) -> bool:
    if not root:
        return False
    try:
        path = Path(token).expanduser().resolve()
        root_path = Path(root).expanduser().resolve()
        return path == root_path or root_path in path.parents
    except Exception:
        return False


def _is_temp_script_path(token: str, root: str | Path | None) -> bool:
    try:
        name = Path(token).expanduser().name
    except Exception:
        return False
    return (
        name.startswith(_AD_HOC_SCRIPT_NAME_PREFIXES)
        and _is_under_temp_dir(token)
        and not _is_under_root(token, root)
    )


def _ad_hoc_script_args(tokens: list[str], root: str | Path | None) -> Optional[list[str]]:
    candidate_tokens = _strip_command_prefix(tokens)
    if not candidate_tokens:
        return None
    command = candidate_tokens[0]
    if _is_temp_script_path(command, root):
        return candidate_tokens[1:]
    if command in {"python", "python3", "node", "bash", "sh", "ruby", "perl"}:
        for idx, token in enumerate(candidate_tokens[1:], start=1):
            if token == "--":
                continue
            if _is_temp_script_path(token, root):
                return candidate_tokens[idx + 1:]
            if not token.startswith("-"):
                return None
    return None


def _find_ad_hoc_match(command: str, root: str | Path | None) -> Optional[list[str]]:
    for tokens in _split_segment_tokens(command):
        trailing_args = _ad_hoc_script_args(tokens, root)
        if trailing_args is not None:
            return trailing_args
    return None


def _summarize_output(output: str) -> str:
    text = (output or "").strip()
    if len(text) <= _MAX_OUTPUT_SUMMARY_CHARS:
        return text
    head = _MAX_OUTPUT_SUMMARY_CHARS // 3
    tail = _MAX_OUTPUT_SUMMARY_CHARS - head
    return (
        text[:head]
        + f"\n... [{len(text) - _MAX_OUTPUT_SUMMARY_CHARS} chars omitted] ...\n"
        + text[-tail:]
    )


def _git_capture(root: Path, *args: str) -> Optional[bytes]:
    """Run a time- and output-bounded, non-shell Git identity query."""
    try:
        process = subprocess.Popen(
            ["git", "-C", str(root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    chunks: list[bytes] = []
    state = {"size": 0, "exceeded": False, "failed": False}

    def _drain_stdout() -> None:
        try:
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    break
                state["size"] += len(chunk)
                if state["size"] <= _MAX_GIT_QUERY_BYTES:
                    chunks.append(chunk)
                else:
                    state["exceeded"] = True
        except (OSError, ValueError):
            state["failed"] = True

    reader = threading.Thread(target=_drain_stdout, daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        reader.join(timeout=1)
        return None
    reader.join(timeout=1)
    if reader.is_alive() or state["failed"] or state["exceeded"] or returncode != 0:
        return None
    return b"".join(chunks)


def _git_workspace_paths(root: Path) -> Optional[set[str]]:
    """Return every Git-visible and ignored workspace path, or fail closed."""
    paths: set[str] = set()
    for args in (
        ("ls-files", "-z"),
        ("diff", "--name-only", "-z", "HEAD"),
        ("diff", "--cached", "--name-only", "-z", "HEAD"),
        ("ls-files", "--others", "--exclude-standard", "-z"),
        ("ls-files", "--others", "--ignored", "--exclude-standard", "-z"),
    ):
        raw = _git_capture(root, *args)
        if raw is None:
            return None
        for item in raw.split(b"\0"):
            if item:
                paths.add(os.fsdecode(item))
                if len(paths) > _MAX_ARTIFACT_PATHS:
                    return None
    return paths


def _parse_submodule_paths(text: str) -> Optional[set[str]]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(text)
    except (configparser.Error, UnicodeError):
        return None
    paths: set[str] = set()
    for section in parser.sections():
        if not section.casefold().startswith("submodule "):
            continue
        value = parser.get(section, "path", fallback="").strip()
        if value:
            paths.add(value)
            if len(paths) > _MAX_ARTIFACT_PATHS:
                return None
    return paths


def _declared_submodule_roots(root: Path, top_root: Path) -> Optional[list[Path]]:
    """Resolve initialized submodules declared by worktree or HEAD metadata."""
    declarations: set[str] = set()
    current = root / ".gitmodules"
    if current.exists():
        try:
            parsed = _parse_submodule_paths(current.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            return None
        if parsed is None:
            return None
        declarations.update(parsed)
        if len(declarations) > _MAX_ARTIFACT_PATHS:
            return None

    tracked = _git_capture(root, "ls-tree", "--name-only", "HEAD", "--", ".gitmodules")
    if tracked is None:
        return None
    if tracked.strip():
        historical = _git_capture(root, "show", "HEAD:.gitmodules")
        if historical is None:
            return None
        try:
            historical_text = historical.decode("utf-8", errors="strict")
        except UnicodeError:
            return None
        parsed = _parse_submodule_paths(historical_text)
        if parsed is None:
            return None
        declarations.update(parsed)
        if len(declarations) > _MAX_ARTIFACT_PATHS:
            return None

    roots: list[Path] = []
    for raw in declarations:
        try:
            candidate = (root / raw).resolve()
        except (OSError, RuntimeError):
            return None
        if candidate == top_root or top_root not in candidate.parents:
            return None
        if (candidate / ".git").exists():
            roots.append(candidate)
    return roots


def _collect_submodule_state(
    root: Path,
    top_root: Path,
    *,
    seen: set[Path],
) -> Optional[tuple[list[tuple[str, bytes]], set[Path]]]:
    identities: list[tuple[str, bytes]] = []
    candidates: set[Path] = set()
    submodule_roots = _declared_submodule_roots(root, top_root)
    if submodule_roots is None:
        return None
    for submodule_root in submodule_roots:
        if submodule_root in seen:
            return None
        seen.add(submodule_root)
        if len(seen) > _MAX_ARTIFACT_PATHS:
            return None
        head = _git_capture(submodule_root, "rev-parse", "--verify", "HEAD")
        paths = _git_workspace_paths(submodule_root)
        if not head or not head.strip() or paths is None:
            return None
        label = str(submodule_root.relative_to(top_root))
        identities.append((label, head.strip()))
        for raw in paths:
            candidates.add(Path(os.path.abspath(submodule_root / raw)))
        nested = _collect_submodule_state(
            submodule_root,
            top_root,
            seen=seen,
        )
        if nested is None:
            return None
        nested_identities, nested_candidates = nested
        identities.extend(nested_identities)
        candidates.update(nested_candidates)
        if len(identities) + len(candidates) > _MAX_ARTIFACT_PATHS:
            return None
    return identities, candidates


def workspace_artifact_fingerprint(
    root: str | Path,
    changed_paths: list[str] | tuple[str, ...] | None = None,
) -> tuple[Optional[str], list[str]]:
    """Hash Git and submodule HEADs plus tracked, untracked, ignored, or explicit paths.

    The returned path list is persisted with the verification event so the
    exact fingerprint can be recomputed later. Git-discovered paths catch edits
    made by terminal commands that bypass the file-tool mutation hook. Missing
    HEAD, inventory errors, and bounded-resource overflow return no identity.
    """
    try:
        root_path = Path(root).expanduser().resolve()
    except Exception:
        return None, []

    head = _git_capture(root_path, "rev-parse", "--verify", "HEAD")
    if not head or not head.strip():
        return None, []
    explicit_candidates: set[Path] = set()
    for raw in changed_paths or []:
        try:
            path = Path(raw).expanduser()
            candidate = root_path / path if not path.is_absolute() else path
            explicit_candidate = Path(os.path.abspath(candidate))
            if (
                explicit_candidate != root_path
                and root_path not in explicit_candidate.parents
            ):
                return None, []
            explicit_candidates.add(explicit_candidate)
        except Exception:
            continue
    candidates = set(explicit_candidates)
    git_paths = _git_workspace_paths(root_path)
    if git_paths is None:
        return None, []
    for raw in git_paths:
        try:
            candidates.add(Path(os.path.abspath(root_path / raw)))
        except Exception:
            continue

    submodule_state = _collect_submodule_state(
        root_path,
        root_path,
        seen={root_path},
    )
    if submodule_state is None:
        return None, []
    submodule_identities, submodule_candidates = submodule_state
    candidates.update(submodule_candidates)

    digest = hashlib.sha256()
    digest.update(b"hermes-workspace-artifact-v1\0")
    digest.update(str(root_path).encode("utf-8", errors="surrogateescape"))
    digest.update(b"\0head\0")
    digest.update(head.strip())
    for label, submodule_head in sorted(submodule_identities):
        digest.update(b"\0submodule\0")
        digest.update(label.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0head\0")
        digest.update(submodule_head)

    try:
        candidate_hermes_home = get_hermes_home().expanduser().resolve()
        hermes_home = (
            candidate_hermes_home
            if candidate_hermes_home != root_path and root_path in candidate_hermes_home.parents
            else None
        )
    except Exception:
        hermes_home = None
    artifact_candidates = []
    for path in candidates:
        if path != root_path and root_path not in path.parents:
            continue
        if hermes_home is not None and (path == hermes_home or hermes_home in path.parents):
            continue
        artifact_candidates.append(path)

    serialized_paths: list[str] = []
    if len(artifact_candidates) + len(submodule_identities) > _MAX_ARTIFACT_PATHS:
        return None, []
    total_hashed_bytes = 0
    for path in sorted(artifact_candidates, key=lambda item: str(item))[:_MAX_ARTIFACT_PATHS]:
        try:
            label = str(path.relative_to(root_path))
        except ValueError:
            label = str(path)
        serialized_paths.append(label)
        digest.update(b"\0path\0")
        digest.update(label.encode("utf-8", errors="surrogateescape"))
        try:
            stat = path.lstat()
        except OSError:
            digest.update(b"\0missing")
            continue
        digest.update(f"\0mode={stat.st_mode:o}\0size={stat.st_size}".encode())
        if path.is_symlink():
            try:
                target = os.readlink(path)
                digest.update(
                    b"\0symlink\0" + target.encode("utf-8", errors="surrogateescape")
                )
            except OSError:
                return None, serialized_paths
            continue
        if not path.is_file():
            digest.update(b"\0non-file")
            continue
        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    total_hashed_bytes += len(chunk)
                    if total_hashed_bytes > _MAX_ARTIFACT_HASH_BYTES:
                        # No partial fingerprints: callers must treat an
                        # oversized artifact as inconclusive, never cache PASS.
                        return None, serialized_paths
                    digest.update(chunk)
        except OSError:
            return None, serialized_paths
        try:
            final_stat = path.lstat()
        except OSError:
            return None, serialized_paths
        if (
            final_stat.st_size != stat.st_size
            or final_stat.st_mtime_ns != stat.st_mtime_ns
            or final_stat.st_ctime_ns != stat.st_ctime_ns
            or final_stat.st_mode != stat.st_mode
            or final_stat.st_dev != stat.st_dev
            or final_stat.st_ino != stat.st_ino
        ):
            return None, serialized_paths

    final_head = _git_capture(root_path, "rev-parse", "--verify", "HEAD")
    if not final_head or final_head.strip() != head.strip():
        return None, serialized_paths
    final_git_paths = _git_workspace_paths(root_path)
    if final_git_paths is None:
        return None, serialized_paths
    final_candidates = set(explicit_candidates)
    for raw in final_git_paths:
        try:
            final_candidates.add(Path(os.path.abspath(root_path / raw)))
        except Exception:
            continue
    final_submodule_state = _collect_submodule_state(
        root_path,
        root_path,
        seen={root_path},
    )
    if final_submodule_state is None:
        return None, serialized_paths
    final_submodule_identities, final_submodule_candidates = final_submodule_state
    final_candidates.update(final_submodule_candidates)
    final_artifact_candidates = {
        path
        for path in final_candidates
        if (path == root_path or root_path in path.parents)
        and not (
            hermes_home is not None
            and (path == hermes_home or hermes_home in path.parents)
        )
    }
    if (
        final_artifact_candidates != set(artifact_candidates)
        or sorted(final_submodule_identities) != sorted(submodule_identities)
    ):
        return None, serialized_paths
    return f"sha256:{digest.hexdigest()}", serialized_paths


def _prune_old_events(conn: sqlite3.Connection, *, session_id: str, root: str) -> None:
    """Bound ledger growth without deleting the current state pointer."""
    cutoff = _retention_cutoff()
    conn.execute(
        """
        DELETE FROM verification_events
        WHERE session_id = ?
          AND root = ?
          AND id NOT IN (
              SELECT id FROM verification_events
              WHERE session_id = ? AND root = ?
              ORDER BY id DESC
              LIMIT ?
          )
        """,
        (session_id, root, session_id, root, _MAX_EVENTS_PER_SESSION_ROOT),
    )
    conn.execute(
        """
        DELETE FROM verification_state
        WHERE (
            last_edit_at IS NOT NULL
            AND last_edit_at < ?
        )
        OR (
            last_edit_at IS NULL
            AND last_event_id IN (
                SELECT id FROM verification_events
                WHERE created_at < ?
            )
        )
        """,
        (cutoff, cutoff),
    )
    conn.execute(
        """
        DELETE FROM verification_events
        WHERE created_at < ?
          AND id NOT IN (
              SELECT last_event_id FROM verification_state
              WHERE last_event_id IS NOT NULL
          )
        """,
        (cutoff,),
    )
    conn.execute(
        """
        DELETE FROM verification_events
        WHERE id NOT IN (
            SELECT id FROM verification_events
            ORDER BY id DESC
            LIMIT ?
        )
          AND id NOT IN (
              SELECT last_event_id FROM verification_state
              WHERE last_event_id IS NOT NULL
          )
        """,
        (_MAX_TOTAL_UNREFERENCED_EVENTS,),
    )


def classify_verification_command(
    command: str,
    *,
    cwd: str | Path | None = None,
    session_id: str | None = None,
    exit_code: int = 0,
    output: str = "",
) -> Optional[VerificationEvidence]:
    """Classify a terminal command as verification evidence, if applicable."""

    if not command or not isinstance(command, str):
        return None
    try:
        from agent.coding_context import project_facts_for

        facts = project_facts_for(cwd)
    except Exception:
        facts = None
    if not facts:
        return None

    verify_commands = list(facts.get("verifyCommands") or [])
    match = _find_canonical_match(command, verify_commands)
    is_ad_hoc = False
    if match is None and not verify_commands:
        ad_hoc_args = _find_ad_hoc_match(command, facts.get("root"))
        if ad_hoc_args is not None:
            match = ("ad-hoc verification script", ad_hoc_args)
            is_ad_hoc = True
    if match is None:
        return None

    canonical, trailing_args = match
    return VerificationEvidence(
        command=command,
        canonical_command=canonical,
        kind="ad_hoc" if is_ad_hoc else _kind_for_command(canonical),
        scope="targeted" if is_ad_hoc else _scope_for_args(trailing_args),
        status="passed" if int(exit_code) == 0 else "failed",
        exit_code=int(exit_code),
        cwd=str(Path(cwd or ".").resolve()),
        root=str(facts.get("root") or Path(cwd or ".").resolve()),
        session_id=str(session_id or "default"),
        output_summary=_summarize_output(output),
    )


def record_terminal_result(
    *,
    command: str,
    cwd: str | Path | None,
    session_id: str | None,
    exit_code: int,
    output: str = "",
) -> Optional[dict[str, Any]]:
    """Record a foreground terminal result when it is verification evidence."""

    evidence = classify_verification_command(
        command,
        cwd=cwd,
        session_id=session_id,
        exit_code=exit_code,
        output=output,
    )
    if evidence is None:
        return None

    created_at = _utc_now()
    with _DB_LOCK:
        with _connect() as conn:
            state_row = conn.execute(
                """
                SELECT changed_paths_json FROM verification_state
                WHERE session_id = ? AND root = ?
                """,
                (evidence.session_id, evidence.root),
            ).fetchone()
            changed_paths: list[str] = []
            if state_row is not None:
                try:
                    parsed_paths = json.loads(state_row["changed_paths_json"] or "[]")
                    if isinstance(parsed_paths, list):
                        changed_paths = [str(path) for path in parsed_paths if path]
                except (TypeError, ValueError):
                    changed_paths = []
            artifact_hash, fingerprint_paths = workspace_artifact_fingerprint(
                evidence.root, changed_paths
            )
            cur = conn.execute(
                """
                INSERT INTO verification_events(
                    created_at, session_id, cwd, root, command, canonical_command,
                    kind, scope, status, exit_code, output_summary,
                    artifact_hash, changed_paths_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    evidence.session_id,
                    evidence.cwd,
                    evidence.root,
                    evidence.command,
                    evidence.canonical_command,
                    evidence.kind,
                    evidence.scope,
                    evidence.status,
                    evidence.exit_code,
                    evidence.output_summary,
                    artifact_hash,
                    json.dumps(fingerprint_paths),
                ),
            )
            if cur.lastrowid is None:
                raise RuntimeError("verification event insert did not return an id")
            event_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO verification_state(
                    session_id, root, last_event_id, last_edit_at, changed_paths_json
                ) VALUES (?, ?, ?, NULL, '[]')
                ON CONFLICT(session_id, root) DO UPDATE SET
                    last_event_id = excluded.last_event_id,
                    last_edit_at = NULL,
                    changed_paths_json = '[]'
                """,
                (evidence.session_id, evidence.root, event_id),
            )
            _prune_old_events(conn, session_id=evidence.session_id, root=evidence.root)
            conn.commit()

    return {
        "id": event_id,
        **evidence.__dict__,
        "created_at": created_at,
        "artifact_hash": artifact_hash,
        "changed_paths": fingerprint_paths,
    }


def mark_workspace_edited(
    *,
    session_id: str | None,
    cwd: str | Path | None,
    paths: list[str] | tuple[str, ...] | None = None,
) -> Optional[dict[str, Any]]:
    """Mark verification evidence stale after a successful file edit."""

    try:
        from agent.coding_context import project_facts_for

        facts = project_facts_for(cwd)
    except Exception:
        facts = None
    if not facts:
        return None

    sid = str(session_id or "default")
    root = str(facts.get("root") or Path(cwd or ".").resolve())
    changed_paths = sorted({str(p) for p in (paths or []) if p})
    edited_at = _utc_now()

    with _DB_LOCK:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT changed_paths_json FROM verification_state
                WHERE session_id = ? AND root = ?
                """,
                (sid, root),
            ).fetchone()
            existing: set[str] = set()
            if row is not None:
                try:
                    existing = set(json.loads(row["changed_paths_json"] or "[]"))
                except (TypeError, ValueError):
                    existing = set()
            merged = sorted((existing | set(changed_paths)))[-200:]
            conn.execute(
                """
                INSERT INTO verification_state(
                    session_id, root, last_event_id, last_edit_at, changed_paths_json
                ) VALUES (?, ?, NULL, ?, ?)
                ON CONFLICT(session_id, root) DO UPDATE SET
                    last_edit_at = excluded.last_edit_at,
                    changed_paths_json = excluded.changed_paths_json
                """,
                (sid, root, edited_at, json.dumps(merged)),
            )
            conn.commit()

    return {"session_id": sid, "root": root, "last_edit_at": edited_at, "changed_paths": changed_paths}


def verification_status(
    *,
    session_id: str | None,
    cwd: str | Path | None,
) -> dict[str, Any]:
    """Return the best known verification state for a session/workspace."""

    try:
        from agent.coding_context import project_facts_for

        facts = project_facts_for(cwd)
    except Exception:
        facts = None
    if not facts:
        return {"status": "not_applicable", "evidence": None}

    sid = str(session_id or "default")
    root = str(facts.get("root") or Path(cwd or ".").resolve())
    with _DB_LOCK:
        with _connect() as conn:
            state = conn.execute(
                """
                SELECT last_event_id, last_edit_at, changed_paths_json
                FROM verification_state
                WHERE session_id = ? AND root = ?
                """,
                (sid, root),
            ).fetchone()
            if state is None:
                return {
                    "status": "unverified",
                    "evidence": None,
                    "root": root,
                    "session_id": sid,
                    "changed_paths": [],
                }
            event = None
            if state["last_event_id"] is not None:
                event = conn.execute(
                    "SELECT * FROM verification_events WHERE id = ?",
                    (state["last_event_id"],),
                ).fetchone()

    changed_paths: list[str] = []
    try:
        changed_paths = json.loads(state["changed_paths_json"] or "[]")
    except (TypeError, ValueError):
        changed_paths = []

    if event is None:
        return {
            "status": "unverified",
            "evidence": None,
            "root": root,
            "session_id": sid,
            "changed_paths": changed_paths,
        }

    evidence = dict(event)
    stored_artifact_hash = evidence.get("artifact_hash")
    event_paths: list[str] = []
    try:
        parsed_event_paths = json.loads(evidence.get("changed_paths_json") or "[]")
        if isinstance(parsed_event_paths, list):
            event_paths = [str(path) for path in parsed_event_paths if path]
    except (TypeError, ValueError):
        event_paths = []
    current_artifact_hash: Optional[str] = None
    if stored_artifact_hash:
        current_artifact_hash, _ = workspace_artifact_fingerprint(root, event_paths)

    if state["last_edit_at"] and state["last_edit_at"] > evidence["created_at"]:
        status = "stale"
    elif stored_artifact_hash and current_artifact_hash != stored_artifact_hash:
        # Catches edits made outside Hermes' file tools, Git commits after the
        # verification run, and untracked artifacts that changed in place.
        status = "stale"
    else:
        status = evidence["status"]
    return {
        "status": status,
        "evidence": evidence,
        "root": root,
        "session_id": sid,
        "changed_paths": changed_paths,
        "artifact_hash": stored_artifact_hash,
        "current_artifact_hash": current_artifact_hash,
    }


def latest_verification_status(*, session_id: str | None) -> dict[str, Any]:
    """Return verification for the session's most recently active workspace."""
    sid = str(session_id or "default")
    with _DB_LOCK:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT s.root
                FROM verification_state AS s
                LEFT JOIN verification_events AS e ON e.id = s.last_event_id
                WHERE s.session_id = ?
                ORDER BY COALESCE(s.last_edit_at, e.created_at, '') DESC
                LIMIT 1
                """,
                (sid,),
            ).fetchone()
    if row is None:
        return {"status": "not_applicable", "evidence": None, "session_id": sid}
    return verification_status(session_id=sid, cwd=str(row["root"]))
