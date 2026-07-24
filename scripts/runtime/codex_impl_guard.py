#!/usr/bin/env python3
"""Guarded Codex implementation runner.

This is the write-capable counterpart to ``codex_review_guard.py``. It keeps
Codex implementation runs bounded and evidence-driven:

- collect dirty baseline before launching Codex;
- constrain candidate changes to repo-relative allowed files/globs;
- write raw Codex output to a log while emitting only bounded JSON;
- treat non-zero/terminated Codex runs with safe diffs as takeover candidates,
  not successful completions.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Reuse the existing review guard's output-containment helpers so the impl guard
# does not grow a parallel flood/truncation dialect.
_RUNTIME_DIR = Path(__file__).resolve().parent
if str(_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DIR))

from codex_review_guard import (  # type: ignore  # noqa: E402
    _bounded,
    _is_diff_like,
    _is_source_like,
    _json_field_flood,
)


_STATUS_EXIT = {
    "passed": 0,
    "review_needed": 0,
    "failed": 1,
    "blocked_by_allowlist": 1,
    "takeover_candidate": 2,
    "unusable": 3,
}


class GitError(RuntimeError):
    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Codex implementation behind git/allowlist guards.")
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", "codex-yuna"))
    parser.add_argument("--workdir", required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file")
    parser.add_argument("--allowed-file", action="append", default=[])
    parser.add_argument("--allowed-glob", action="append", default=[])
    parser.add_argument(
        "--dirty-baseline-policy",
        choices=["require-clean", "allow-listed-owned", "fail-on-overlap"],
        default="require-clean",
    )
    parser.add_argument("--verify-cmd-id", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=2.0)
    parser.add_argument("--raw-log")
    parser.add_argument("--final-file")
    parser.add_argument("--max-stdout-chars", type=int, default=200_000)
    parser.add_argument("--max-stdout-lines", type=int, default=4_000)
    parser.add_argument("--source-line-threshold", type=int, default=250)
    parser.add_argument("--diff-line-threshold", type=int, default=300)
    parser.add_argument("--json-field-char-threshold", type=int, default=40_000)
    return parser


def _emit(result: dict[str, Any]) -> int:
    status = str(result.get("status", "unusable"))
    print(json.dumps(_bounded(result), ensure_ascii=False, sort_keys=True))
    return _STATUS_EXIT.get(status, 3)


def _run_git(workdir: Path, args: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(workdir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if check and proc.returncode != 0:
        raise GitError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def _git_root(workdir: Path) -> Path:
    root = _run_git(workdir, ["rev-parse", "--show-toplevel"]).strip()
    if not root:
        raise GitError("not a git repository")
    return Path(root).resolve()


def _parse_porcelain_z(output: str) -> list[dict[str, str]]:
    parts = [part for part in output.split("\0") if part]
    records: list[dict[str, str]] = []
    i = 0
    while i < len(parts):
        entry = parts[i]
        if len(entry) < 4:
            i += 1
            continue
        status = entry[:2]
        path = entry[3:]
        if status[0] in {"R", "C"} or status[1] in {"R", "C"}:
            # In porcelain -z rename/copy records may include a second path.
            old_path = parts[i + 1] if i + 1 < len(parts) else ""
            records.append({"status": status, "path": path, "old_path": old_path})
            i += 2
        else:
            records.append({"status": status, "path": path})
            i += 1
    return records


def _status_records(workdir: Path) -> list[dict[str, str]]:
    return _parse_porcelain_z(_run_git(workdir, ["status", "--porcelain=v1", "-z", "--untracked-files=all"]))


_IGNORED_ARTIFACT_MAX_BYTES = 1_000_000


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _IGNORED_ARTIFACT_MAX_BYTES:
                return f"oversized:{total}"
            digest.update(chunk)
    return f"file:{total}:{digest.hexdigest()}"


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    count = 0
    try:
        children = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
    except OSError as exc:
        return f"dir-error:{type(exc).__name__}"
    for child in children:
        count += 1
        if count > 2_000:
            return f"dir-oversized:{count}"
        try:
            rel = child.relative_to(path).as_posix()
        except ValueError:
            return "dir-unsafe"
        digest.update(rel.encode("utf-8", errors="surrogateescape"))
        try:
            if child.is_symlink():
                digest.update(b"symlink:")
                digest.update(os.readlink(child).encode("utf-8", errors="surrogateescape"))
            elif child.is_file():
                digest.update(_file_fingerprint(child).encode("utf-8"))
            elif child.is_dir():
                digest.update(b"dir")
            else:
                digest.update(b"other")
        except OSError as exc:
            digest.update(f"error:{type(exc).__name__}".encode("utf-8"))
    return f"dir:{count}:{digest.hexdigest()}"


def _ignored_artifact_fingerprints(workdir: Path) -> dict[str, str]:
    raw = _run_git(workdir, ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"], check=False)
    paths = sorted({part for part in raw.split("\0") if part})
    fingerprints: dict[str, str] = {}
    for repo_path in paths:
        if not _candidate_path_safe(repo_path, workdir):
            fingerprints[repo_path] = "unsafe"
            continue
        path = workdir / repo_path
        try:
            if path.is_symlink():
                fingerprints[repo_path] = "symlink:" + os.readlink(path)
            elif path.is_dir():
                fingerprints[repo_path] = _directory_fingerprint(path)
            elif path.is_file():
                fingerprints[repo_path] = _file_fingerprint(path)
            else:
                fingerprints[repo_path] = "missing"
        except OSError as exc:
            fingerprints[repo_path] = f"error:{type(exc).__name__}"
    return fingerprints


def _changed_ignored_artifacts(before: dict[str, str], after: dict[str, str]) -> list[str]:
    changed: set[str] = set()
    for path, fingerprint in after.items():
        if before.get(path) != fingerprint:
            changed.add(path)
    for path in before:
        if path not in after:
            changed.add(path)
    return sorted(changed)


def _git_metadata_fingerprints(workdir: Path) -> dict[str, str]:
    git_dir = workdir / ".git"
    targets: list[Path] = []
    if git_dir.is_file():
        targets.append(git_dir)
    elif git_dir.exists():
        for path in git_dir.rglob("*"):
            if path.is_file() or path.is_symlink():
                targets.append(path)
    fingerprints: dict[str, str] = {}
    for path in targets:
        try:
            rel = path.relative_to(workdir).as_posix()
        except ValueError:
            continue
        try:
            if path.is_symlink():
                fingerprints[rel] = "symlink:" + os.readlink(path)
            elif path.is_file():
                fingerprints[rel] = _file_fingerprint(path)
            elif path.exists():
                fingerprints[rel] = "other"
            else:
                fingerprints[rel] = "missing"
        except OSError as exc:
            fingerprints[rel] = f"error:{type(exc).__name__}"
    return fingerprints


def _changed_git_metadata(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return _changed_ignored_artifacts(before, after)


def _status_paths(records: list[dict[str, str]]) -> list[str]:
    paths: list[str] = []
    for record in records:
        path = record.get("path", "")
        if path:
            paths.append(path)
        old_path = record.get("old_path", "")
        if old_path:
            paths.append(old_path)
    return sorted(dict.fromkeys(paths))


def _path_parts(path: str) -> list[str]:
    return [part for part in path.replace("\\", "/").split("/") if part]


def _valid_repo_relative_pattern(pattern: str) -> bool:
    if not pattern or os.path.isabs(pattern):
        return False
    parts = _path_parts(pattern)
    if not parts or any(part == ".." for part in parts):
        return False
    return not pattern.startswith("../")


def _validate_allowlist(files: list[str], globs: list[str]) -> list[str]:
    invalid: list[str] = []
    for pattern in [*files, *globs]:
        if not _valid_repo_relative_pattern(pattern):
            invalid.append(pattern)
    return invalid


def _repo_relative_for_existing(path: Path, root: Path) -> str | None:
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return None
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return None
    return rel.as_posix()


def _candidate_path_safe(repo_path: str, root: Path) -> bool:
    if not repo_path or os.path.isabs(repo_path):
        return False
    if any(part == ".." for part in _path_parts(repo_path)):
        return False
    candidate = root / repo_path
    if candidate.exists() or candidate.is_symlink():
        return _repo_relative_for_existing(candidate, root) is not None
    # Deleted paths cannot be resolved strictly. Lexical repo-relative paths are
    # acceptable because Git produced them from the target worktree.
    return True


def _matches_allowlist(path: str, files: list[str], globs: list[str]) -> bool:
    if path in files:
        return True
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in globs)


def _path_prefix_overlap(left: str, right: str) -> bool:
    left = left.strip("/")
    right = right.strip("/")
    if not left or not right:
        return False
    return left == right or right.startswith(left + "/") or left.startswith(right + "/")


def _literal_glob_prefix(pattern: str) -> str:
    parts: list[str] = []
    for part in _path_parts(pattern):
        if any(ch in part for ch in "*?["):
            break
        parts.append(part)
    return "/".join(parts)


def _dirty_path_overlaps_allowlist(path: str, files: list[str], globs: list[str]) -> bool:
    if _matches_allowlist(path, files, globs):
        return True
    if any(_path_prefix_overlap(path, file) for file in files):
        return True
    for pattern in globs:
        prefix = _literal_glob_prefix(pattern)
        if prefix and _path_prefix_overlap(path, prefix):
            return True
    return False


def _classify_records(records: list[dict[str, str]], *, excluded: set[str]) -> dict[str, list[str]]:
    changed: set[str] = set()
    untracked: set[str] = set()
    deleted: set[str] = set()
    renamed: set[str] = set()
    submodule: set[str] = set()

    def is_excluded(path: str) -> bool:
        return bool(path and path in excluded)

    for record in records:
        status = record.get("status", "")
        path = record.get("path", "")
        old_path = record.get("old_path", "")
        if not path or is_excluded(path):
            continue
        if status == "??":
            untracked.add(path)
            continue
        if "D" in status:
            deleted.add(path)
        if "R" in status or "C" in status:
            renamed.add(path)
            if old_path and not is_excluded(old_path):
                renamed.add(old_path)
        if status.strip():
            changed.add(path)
    return {
        "changed_files": sorted(changed),
        "untracked_files": sorted(untracked),
        "deleted_files": sorted(deleted),
        "renamed_files": sorted(renamed),
        "submodule_changes": sorted(submodule),
    }


def _excluded_output_paths(paths: list[Path], root: Path) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        try:
            rel = path.resolve(strict=False).relative_to(root)
        except ValueError:
            continue
        excluded.add(rel.as_posix())
    return excluded


def _allowlist_violations(paths: list[str], *, root: Path, files: list[str], globs: list[str]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        if not _candidate_path_safe(path, root) or not _matches_allowlist(path, files, globs):
            violations.append(path)
    return sorted(dict.fromkeys(violations))


def _dirty_baseline_fingerprints(workdir: Path, paths: list[str]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for repo_path in paths:
        if not _candidate_path_safe(repo_path, workdir):
            fingerprints[repo_path] = "unsafe"
            continue
        path = workdir / repo_path
        try:
            if path.is_symlink():
                fingerprints[repo_path] = "symlink:" + os.readlink(path)
            elif path.is_dir():
                fingerprints[repo_path] = _directory_fingerprint(path)
            elif path.is_file():
                fingerprints[repo_path] = _file_fingerprint(path)
            else:
                fingerprints[repo_path] = "missing"
        except OSError as exc:
            fingerprints[repo_path] = f"error:{type(exc).__name__}"
    return fingerprints


def _dirty_policy_error(
    *,
    policy: str,
    dirty_baseline: list[str],
    workdir: Path,
    files: list[str],
    globs: list[str],
) -> dict[str, Any] | None:
    if policy == "require-clean":
        if dirty_baseline:
            return {
                "status": "unusable",
                "reason": "dirty_baseline_not_clean",
                "dirty_baseline": dirty_baseline,
                "dirty_baseline_policy": policy,
            }
        return None

    if policy == "allow-listed-owned":
        violations = _allowlist_violations(dirty_baseline, root=workdir, files=files, globs=globs)
        if violations:
            return {
                "status": "unusable",
                "reason": "dirty_baseline_outside_allowlist",
                "dirty_baseline": dirty_baseline,
                "dirty_baseline_policy": policy,
                "dirty_baseline_violations": violations,
            }
        return None

    if policy == "fail-on-overlap":
        overlap = sorted(path for path in dirty_baseline if _dirty_path_overlaps_allowlist(path, files, globs))
        if overlap:
            return {
                "status": "unusable",
                "reason": "dirty_baseline_overlaps_allowlist",
                "dirty_baseline": dirty_baseline,
                "dirty_baseline_policy": policy,
                "dirty_baseline_overlap": overlap,
            }
        return None

    return {
        "status": "unusable",
        "reason": "unsupported_dirty_baseline_policy",
        "dirty_baseline": dirty_baseline,
        "dirty_baseline_policy": policy,
    }


def _policy_candidate_paths(
    paths: list[str],
    *,
    policy: str,
    dirty_baseline: list[str],
    baseline_fingerprints: dict[str, str],
    workdir: Path,
) -> list[str]:
    if policy != "fail-on-overlap":
        return sorted(dict.fromkeys(paths))

    baseline = set(dirty_baseline)
    candidates: set[str] = set()
    after_fingerprints = _dirty_baseline_fingerprints(workdir, dirty_baseline)
    for path in paths:
        if path not in baseline:
            candidates.add(path)
        elif after_fingerprints.get(path) != baseline_fingerprints.get(path):
            candidates.add(path)
    return sorted(candidates)


def _diff_stat(workdir: Path, paths: list[str]) -> str:
    args = ["diff", "--stat"]
    if paths:
        args.extend(["--", *paths])
    text = _run_git(workdir, args, check=False)
    if len(text) > 4000:
        return text[:3900] + "\n...[truncated]"
    return text


def _submodule_changes(workdir: Path) -> list[str]:
    """Return paths whose raw diff mode indicates a gitlink/submodule change."""
    paths: set[str] = set()
    for diff_args in (["diff", "--raw", "-z"], ["diff", "--cached", "--raw", "-z"]):
        raw = _run_git(workdir, diff_args, check=False)
        parts = [part for part in raw.split("\0") if part]
        index = 0
        while index < len(parts):
            header = parts[index]
            if not header.startswith(":"):
                index += 1
                continue
            fields = header.split()
            status = fields[4] if len(fields) >= 5 else ""
            path = parts[index + 1] if index + 1 < len(parts) else ""
            next_index = index + 2
            if status.startswith(("R", "C")):
                new_path = parts[index + 2] if index + 2 < len(parts) else ""
                next_index = index + 3
            else:
                new_path = ""
            modes = fields[0:2]
            if "160000" in modes:
                if path:
                    paths.add(path)
                if new_path:
                    paths.add(new_path)
            index = next_index
    return sorted(paths)


def _nested_repo_changes(workdir: Path, candidate_paths: list[str]) -> list[str]:
    nested: set[str] = set()
    root = workdir.resolve()
    for repo_path in candidate_paths:
        parts = _path_parts(repo_path)
        if ".git" in parts:
            git_index = parts.index(".git")
            nested_root = "/".join(parts[:git_index]) or "."
            nested.add(nested_root)
            continue
        path = (workdir / repo_path).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_dir() and (path / ".git").exists():
            nested.add(repo_path.rstrip("/"))

    # Git porcelain intentionally does not report files inside nested .git
    # directories as ordinary untracked files. Scan the filesystem as a second
    # line of defense so an allowlisted glob cannot hide a newly-created nested
    # repo under the candidate tree.
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if path == root:
            dirnames[:] = [name for name in dirnames if name != ".git"]
        if ".git" in dirnames or ".git" in filenames:
            rel = path.relative_to(root).as_posix()
            if rel and rel != ".":
                nested.add(rel)
            dirnames[:] = [name for name in dirnames if name != ".git"]
    return sorted(nested)


def _terminate(proc: subprocess.Popen[Any], grace: float) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=grace)
    except Exception:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            proc.kill()
        proc.wait(timeout=5)


def _build_codex_prompt(base_prompt: str, files: list[str], globs: list[str]) -> str:
    return "\n".join([
        "You are producing a candidate implementation patch for Hermes.",
        "Do not commit, push, deploy, restart services, reset, clean, or revert existing changes.",
        "Only modify the allowed files/globs listed below.",
        "Do not read or write secrets. Do not print full diffs or large logs.",
        "Avoid broad grep/find over the whole repo; use focused reads/searches.",
        "Hermes will inspect git evidence and run final verification separately.",
        "",
        "Allowed files:",
        *(f"- {item}" for item in files),
        "Allowed globs:",
        *(f"- {item}" for item in globs),
        "",
        "Task:",
        base_prompt,
    ])


def _run_codex(
    *,
    codex_bin: str,
    workdir: Path,
    prompt: str,
    final_path: Path,
    raw_log_path: Path,
    timeout_seconds: float,
    kill_grace_seconds: float,
    max_stdout_chars: int,
    max_stdout_lines: int,
    source_line_threshold: int,
    diff_line_threshold: int,
    json_field_char_threshold: int,
) -> dict[str, Any]:
    cmd = [
        codex_bin,
        "exec",
        "--sandbox",
        "workspace-write",
        "--json",
        "--output-last-message",
        str(final_path),
        "--color",
        "never",
        "--",
        prompt,
    ]
    start = time.monotonic()
    stdout_chars = 0
    stdout_lines = 0
    source_lines = 0
    diff_lines = 0
    json_flood: dict[str, Any] | None = None
    terminated = False
    reason = ""
    process_exited_before_guard = False

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            start_new_session=(os.name == "posix"),
        )
    except FileNotFoundError:
        return {
            "codex_exit_code": None,
            "terminated_by_guard": False,
            "reason": "codex_bin_not_found",
            "stdout_chars": 0,
            "stdout_lines": 0,
            "source_like_lines": 0,
            "diff_like_lines": 0,
            "source_flood_detected": False,
            "diff_flood_detected": False,
            "json_field_flood_detected": False,
        }

    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    selector.register(proc.stdout, selectors.EVENT_READ)
    partial = ""

    def process_line(line: str) -> None:
        nonlocal source_lines, diff_lines, json_flood
        if _is_source_like(line):
            source_lines += 1
        if _is_diff_like(line):
            diff_lines += 1
        if json_flood is None:
            json_flood = _json_field_flood(
                line,
                source_line_threshold=source_line_threshold,
                diff_line_threshold=diff_line_threshold,
                char_threshold=json_field_char_threshold,
            )

    def flood_reason() -> str | None:
        if json_flood:
            return str(json_flood["reason"])
        if stdout_chars > max_stdout_chars or stdout_lines > max_stdout_lines:
            return "stdout_limit_exceeded"
        if source_lines >= source_line_threshold:
            return "source_flood"
        if diff_lines >= diff_line_threshold:
            return "diff_flood"
        return None

    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_log_path.open("w", encoding="utf-8", errors="replace") as raw_log:
        while True:
            if timeout_seconds > 0 and time.monotonic() - start > timeout_seconds:
                terminated = True
                reason = "timeout"
                _terminate(proc, kill_grace_seconds)
                break
            events = selector.select(timeout=0.05)
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    try:
                        selector.unregister(key.fileobj)
                    except Exception:
                        pass
                    continue
                text = chunk.decode("utf-8", errors="replace")
                raw_log.write(text)
                raw_log.flush()
                stdout_chars += len(text)
                stdout_lines += text.count("\n")
                combined = partial + text
                lines = combined.split("\n")
                partial = lines[-1]
                for line in lines[:-1]:
                    process_line(line)
                current = flood_reason()
                if current:
                    terminated = True
                    reason = current
                    _terminate(proc, kill_grace_seconds)
                    break
            if terminated:
                break
            if proc.poll() is not None:
                while True:
                    try:
                        chunk = os.read(proc.stdout.fileno(), 8192)
                    except Exception:
                        chunk = b""
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    raw_log.write(text)
                    raw_log.flush()
                    stdout_chars += len(text)
                    stdout_lines += text.count("\n")
                    combined = partial + text
                    lines = combined.split("\n")
                    partial = lines[-1]
                    for line in lines[:-1]:
                        process_line(line)
                break
    if not terminated and partial:
        process_line(partial)
        current = flood_reason()
        if current:
            terminated = True
            process_exited_before_guard = proc.poll() is not None
            reason = current
            if not process_exited_before_guard:
                _terminate(proc, kill_grace_seconds)

    return {
        "codex_exit_code": proc.returncode,
        "terminated_by_guard": terminated and not process_exited_before_guard,
        "process_exited_before_guard": process_exited_before_guard,
        "reason": reason or "ok",
        "stdout_chars": stdout_chars,
        "stdout_lines": stdout_lines,
        "source_like_lines": source_lines,
        "diff_like_lines": diff_lines,
        "source_flood_detected": reason == "source_flood" or source_lines >= source_line_threshold,
        "diff_flood_detected": reason == "diff_flood" or diff_lines >= diff_line_threshold,
        "json_field_flood_detected": json_flood is not None,
        "json_flood_field": (json_flood or {}).get("json_flood_field"),
        "json_flood_path": (json_flood or {}).get("json_flood_path"),
        "json_flood_chars": (json_flood or {}).get("json_flood_chars", 0),
    }


_SUPPORTED_VERIFY_CMD_IDS = {"none", "diff-check"}
_ALLOWED_CODEX_BIN_NAMES = {"codex-yuna", "codex", "codex-yuna.exe", "codex.exe"}


def _codex_bin_allowed(codex_bin: str) -> bool:
    if os.environ.get("HERMES_CODEX_IMPL_GUARD_ALLOW_FAKE_CODEX") == "1" and os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    candidate = Path(codex_bin)
    if candidate.name not in _ALLOWED_CODEX_BIN_NAMES:
        return False
    # Normal runtime use should resolve through PATH. Reject explicit paths here
    # so this foreground-safe wrapper cannot become an arbitrary executable shim.
    return len(candidate.parts) == 1


def _sandbox_verified() -> bool:
    if os.environ.get("HERMES_CODEX_IMPL_GUARD_ALLOW_FAKE_CODEX") == "1" and os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return os.environ.get("HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED") == "1"


def _safe_output_path(path: Path, *, workdir: Path, label: str) -> tuple[Path, dict[str, Any] | None]:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(workdir)
        return resolved, {
            "status": "unusable",
            "reason": "unsafe_output_path",
            "unsafe_output_path_label": label,
            "unsafe_output_path": str(resolved),
            "unsafe_output_path_detail": "inside_workdir",
        }
    except ValueError:
        pass
    if resolved.exists() or resolved.is_symlink():
        return resolved, {
            "status": "unusable",
            "reason": "unsafe_output_path",
            "unsafe_output_path_label": label,
            "unsafe_output_path": str(resolved),
            "unsafe_output_path_detail": "path_exists_or_symlink",
        }
    parent = resolved.parent.resolve(strict=False)
    try:
        parent.relative_to(workdir)
        return resolved, {
            "status": "unusable",
            "reason": "unsafe_output_path",
            "unsafe_output_path_label": label,
            "unsafe_output_path": str(resolved),
            "unsafe_output_path_detail": "parent_inside_workdir",
        }
    except ValueError:
        pass
    return resolved, None


def _verification_result(workdir: Path, verify_id: str, excluded: set[str]) -> dict[str, Any]:
    pre = _status_paths([r for r in _status_records(workdir) if r.get("path") not in excluded])
    if verify_id == "none":
        return {"id": verify_id, "status": "skipped", "exit_code": 0, "new_artifacts": [], "artifact_violations": []}
    if verify_id not in _SUPPORTED_VERIFY_CMD_IDS:
        return {"id": verify_id, "status": "failed", "exit_code": None, "reason": "unknown_verify_cmd_id"}
    start = time.monotonic()
    proc = subprocess.run(
        ["git", "diff", "--check"],
        cwd=str(workdir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    cached_proc = subprocess.run(
        ["git", "diff", "--cached", "--check"],
        cwd=str(workdir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    post_records = [r for r in _status_records(workdir) if r.get("path") not in excluded]
    post = _status_paths(post_records)
    new_artifacts = sorted(set(post) - set(pre))
    artifact_violations = new_artifacts
    status = "passed" if proc.returncode == 0 and cached_proc.returncode == 0 and not artifact_violations else "failed"
    if artifact_violations:
        status = "blocked_by_artifact_violation"
    return {
        "id": verify_id,
        "status": status,
        "exit_code": proc.returncode,
        "cached_exit_code": cached_proc.returncode,
        "stdout_preview": (proc.stdout + cached_proc.stdout)[:1000],
        "stderr_preview": (proc.stderr + cached_proc.stderr)[:1000],
        "duration_seconds": round(time.monotonic() - start, 3),
        "pre_status_count": len(pre),
        "post_status_count": len(post),
        "new_artifacts": new_artifacts,
        "allowed_artifacts": [],
        "artifact_violations": artifact_violations,
    }


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    workdir = Path(args.workdir).resolve()
    if not workdir.is_dir():
        return _emit({"status": "unusable", "reason": "workdir_not_found", "workdir": str(workdir)})
    try:
        root = _git_root(workdir)
    except GitError as exc:
        return _emit({"status": "unusable", "reason": "not_git_repo", "error": str(exc), "workdir": str(workdir)})
    if root != workdir:
        # Keep v1 simple and deterministic: the guard operates from repo root.
        workdir = root

    invalid = _validate_allowlist(args.allowed_file, args.allowed_glob)
    if invalid:
        return _emit({"status": "unusable", "reason": "invalid_allowlist", "invalid_allowlist": invalid})

    unsupported_verify_ids = [verify_id for verify_id in args.verify_cmd_id if verify_id not in _SUPPORTED_VERIFY_CMD_IDS]
    if unsupported_verify_ids:
        return _emit({
            "status": "unusable",
            "reason": "unsupported_verify_cmd_id",
            "unsupported_verify_cmd_ids": unsupported_verify_ids,
        })
    real_verify_ids = [verify_id for verify_id in args.verify_cmd_id if verify_id != "none"]

    if not _codex_bin_allowed(args.codex_bin):
        return _emit({
            "status": "unusable",
            "reason": "unsupported_codex_bin",
            "codex_bin": args.codex_bin,
        })
    if not _sandbox_verified():
        return _emit({
            "status": "unusable",
            "reason": "sandbox_not_verified",
            "codex_bin": args.codex_bin,
            "required_env": "HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED=1",
        })

    guard_dir = Path(tempfile.mkdtemp(prefix="codex-impl-guard-"))
    raw_log_path, raw_error = _safe_output_path(
        Path(args.raw_log) if args.raw_log else guard_dir / "raw.log",
        workdir=workdir,
        label="raw_log_path",
    )
    if raw_error:
        return _emit(raw_error)
    final_path, final_error = _safe_output_path(
        Path(args.final_file) if args.final_file else guard_dir / "final.json",
        workdir=workdir,
        label="final_file",
    )
    if final_error:
        return _emit(final_error)
    if raw_log_path == final_path:
        return _emit({
            "status": "unusable",
            "reason": "unsafe_output_path",
            "unsafe_output_path_label": "raw_log_path/final_file",
            "unsafe_output_path": str(raw_log_path),
            "unsafe_output_path_detail": "same_output_path",
        })
    final_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    excluded = _excluded_output_paths([raw_log_path, final_path], workdir)

    baseline_records = [r for r in _status_records(workdir) if r.get("path") not in excluded]
    baseline_ignored_artifacts = _ignored_artifact_fingerprints(workdir)
    baseline_git_metadata = _git_metadata_fingerprints(workdir)
    dirty_baseline = _status_paths(baseline_records)
    dirty_baseline_fingerprints = _dirty_baseline_fingerprints(workdir, dirty_baseline)
    dirty_error = _dirty_policy_error(
        policy=args.dirty_baseline_policy,
        dirty_baseline=dirty_baseline,
        workdir=workdir,
        files=args.allowed_file,
        globs=args.allowed_glob,
    )
    if dirty_error:
        return _emit(dirty_error)

    prompt = args.prompt or Path(args.prompt_file).read_text(encoding="utf-8", errors="replace")
    codex_result = _run_codex(
        codex_bin=args.codex_bin,
        workdir=workdir,
        prompt=_build_codex_prompt(prompt, args.allowed_file, args.allowed_glob),
        final_path=final_path,
        raw_log_path=raw_log_path,
        timeout_seconds=args.timeout_seconds,
        kill_grace_seconds=args.kill_grace_seconds,
        max_stdout_chars=args.max_stdout_chars,
        max_stdout_lines=args.max_stdout_lines,
        source_line_threshold=args.source_line_threshold,
        diff_line_threshold=args.diff_line_threshold,
        json_field_char_threshold=args.json_field_char_threshold,
    )

    final_records = [r for r in _status_records(workdir) if r.get("path") not in excluded]
    ignored_artifacts = _changed_ignored_artifacts(baseline_ignored_artifacts, _ignored_artifact_fingerprints(workdir))
    git_metadata_changes = _changed_git_metadata(baseline_git_metadata, _git_metadata_fingerprints(workdir))
    classified = _classify_records(final_records, excluded=excluded)
    submodule_changes = [path for path in _submodule_changes(workdir) if path not in excluded]
    candidate_paths = sorted(dict.fromkeys([*_status_paths(final_records), *submodule_changes]))
    nested_repo_changes = [path for path in _nested_repo_changes(workdir, candidate_paths) if path not in excluded]
    submodule_changes = sorted(dict.fromkeys([*submodule_changes, *nested_repo_changes]))
    classified["submodule_changes"] = submodule_changes
    classified["ignored_artifacts"] = ignored_artifacts
    classified["git_metadata_changes"] = git_metadata_changes
    candidate_paths = sorted(dict.fromkeys([*_status_paths(final_records), *submodule_changes, *ignored_artifacts, *git_metadata_changes]))
    candidate_paths = _policy_candidate_paths(
        candidate_paths,
        policy=args.dirty_baseline_policy,
        dirty_baseline=dirty_baseline,
        baseline_fingerprints=dirty_baseline_fingerprints,
        workdir=workdir,
    )
    violations = _allowlist_violations(
        candidate_paths,
        root=workdir,
        files=args.allowed_file,
        globs=args.allowed_glob,
    )
    # V1 has no safe submodule, ignored-artifact, or git-metadata write path.
    # These are blocked even if a broad glob would otherwise match them.
    violations = sorted(dict.fromkeys([*violations, *submodule_changes, *ignored_artifacts, *git_metadata_changes]))
    if violations:
        status = "blocked_by_allowlist"
        reason = "allowlist_violation"
    else:
        exit_code = codex_result.get("codex_exit_code")
        terminated = bool(codex_result.get("terminated_by_guard")) or codex_result.get("reason") in {
            "timeout",
            "source_flood",
            "diff_flood",
            "stdout_limit_exceeded",
            "aggregated_output_flood",
            "json_field_flood",
        }
        has_candidate = bool(candidate_paths)
        if codex_result.get("reason") == "codex_bin_not_found":
            status = "unusable"
            reason = "codex_bin_not_found"
        elif (terminated or exit_code not in (0, None)) and has_candidate:
            if real_verify_ids:
                status = "takeover_candidate"
                reason = "codex_terminated_with_safe_diff" if terminated else "codex_nonzero_with_safe_diff"
            else:
                status = "failed"
                reason = "takeover_missing_verification"
        elif terminated:
            status = "failed"
            reason = str(codex_result.get("reason") or "codex_terminated_without_diff")
        elif exit_code == 0:
            status = "review_needed"
            reason = "codex_exit_zero_safe_diff"
        else:
            status = "failed"
            reason = "codex_nonzero_without_diff"

    verification = []
    if status in {"review_needed", "passed", "takeover_candidate"}:
        verify_ids_to_run = real_verify_ids if status == "takeover_candidate" else args.verify_cmd_id
        for verify_id in verify_ids_to_run:
            verification.append(_verification_result(workdir, verify_id, excluded))

        # Verification commands are part of the safety boundary: even low-risk
        # commands can leave cache or mutate files. Recompute evidence after the
        # verification phase so final JSON and allowlist decisions describe the
        # actual final worktree, not only the post-Codex/pre-verify state.
        final_records = [r for r in _status_records(workdir) if r.get("path") not in excluded]
        ignored_artifacts = _changed_ignored_artifacts(baseline_ignored_artifacts, _ignored_artifact_fingerprints(workdir))
        git_metadata_changes = _changed_git_metadata(baseline_git_metadata, _git_metadata_fingerprints(workdir))
        classified = _classify_records(final_records, excluded=excluded)
        submodule_changes = [path for path in _submodule_changes(workdir) if path not in excluded]
        candidate_paths = sorted(dict.fromkeys([*_status_paths(final_records), *submodule_changes]))
        nested_repo_changes = [path for path in _nested_repo_changes(workdir, candidate_paths) if path not in excluded]
        submodule_changes = sorted(dict.fromkeys([*submodule_changes, *nested_repo_changes]))
        classified["submodule_changes"] = submodule_changes
        classified["ignored_artifacts"] = ignored_artifacts
        classified["git_metadata_changes"] = git_metadata_changes
        candidate_paths = sorted(dict.fromkeys([*_status_paths(final_records), *submodule_changes, *ignored_artifacts, *git_metadata_changes]))
        candidate_paths = _policy_candidate_paths(
            candidate_paths,
            policy=args.dirty_baseline_policy,
            dirty_baseline=dirty_baseline,
            baseline_fingerprints=dirty_baseline_fingerprints,
            workdir=workdir,
        )
        violations = _allowlist_violations(
            candidate_paths,
            root=workdir,
            files=args.allowed_file,
            globs=args.allowed_glob,
        )
        violations = sorted(dict.fromkeys([*violations, *submodule_changes, *ignored_artifacts, *git_metadata_changes]))

        if violations:
            status = "blocked_by_allowlist"
            reason = "allowlist_violation"
        elif status == "takeover_candidate" and not candidate_paths:
            status = "failed"
            reason = "takeover_missing_final_diff"
        elif any(item.get("status") in {"failed", "blocked_by_artifact_violation", "timed_out"} for item in verification):
            status = "failed"
            reason = "verification_failed"
        elif status == "takeover_candidate":
            pass
        elif verification and all(item.get("status") == "passed" for item in verification):
            status = "passed"
            reason = "verification_passed"

    trusted_completion = bool(codex_result.get("codex_exit_code") == 0 and not codex_result.get("terminated_by_guard") and status != "blocked_by_allowlist")
    if status in {"takeover_candidate", "failed", "unusable", "blocked_by_allowlist"}:
        trusted_completion = False

    codex_metadata = dict(codex_result)
    codex_metadata["codex_reason"] = codex_metadata.pop("reason", "")
    result: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "trusted_completion": trusted_completion,
        "verification": verification,
        **codex_metadata,
        **classified,
        "dirty_baseline": dirty_baseline,
        "dirty_baseline_policy": args.dirty_baseline_policy,
        "allowlist_violations": violations,
        "raw_log_path": str(raw_log_path),
        "final_file": str(final_path),
        "diff_stat": _diff_stat(workdir, candidate_paths),
        "verification": verification,
        "recommended_next_action": "Run read-only review and Hermes verification." if status == "review_needed" else "Inspect candidate diff before continuing." if status == "takeover_candidate" else "Fix blocker before continuing." if status in {"blocked_by_allowlist", "failed"} else "Continue.",
    }
    return _emit(result)


if __name__ == "__main__":
    raise SystemExit(run())
