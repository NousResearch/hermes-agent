"""Move bundle: the portable copy of one profile that "Move to Hermes Cloud" carries.

One profile in, one zip out, built for MERGING into a live install rather than overwriting it.
The archive holds the user's own state only and never a credential file:

``move-manifest.json``
    Schema version, source facts, the inventory the receiving side previews, and the
    *names* (never values) of the keys the user will have to re-enter.
``config.yaml``
    The source's raw config, with credential-shaped keys and machine-local sections dropped.
``SOUL.md``, ``memories/MEMORY.md``, ``memories/USER.md``
    Persona and curated memory.
``skills/<category>/<name>/...``
    User-authored and hub-installed skills only. Bundled skills the target already ships
    (an unmodified copy whose hash matches ``.bundled_manifest``) stay behind.
``cron/jobs.json``
    Job definitions, stripped of run state. They arrive paused.
``sessions/NNNN.json``
    Optional chat history, one portable export per file (``SessionDB.export_session``).

The receiving side is :mod:`hermes_cli.move_receive`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

MOVE_SCHEMA_VERSION = 1
MANIFEST_NAME = "move-manifest.json"

# A skill directory is copied whole unless it is one of these, at any depth.
_SKILL_SKIP_DIRS = frozenset({
    ".git", ".hub", ".archive", ".curator_backups", ".locks", "__pycache__", "node_modules",
    ".venv", "venv", "site-packages", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})
_SKILL_SKIP_SUFFIXES = (".pyc", ".pyo", ".lock")
# A single skill bigger than this is almost always vendored data (models, datasets, media).
# It is listed in the manifest as skipped so the user knows, not silently dropped.
SKILL_MAX_BYTES = 25 * 1024 * 1024
# Sessions per bundle and per file. The receiver imports one file per request, so a file is
# also the unit of retry.
HISTORY_DEFAULT_LIMIT = 200
_SESSION_FILE_MAX_BYTES = 4 * 1024 * 1024

# Sources whose sessions are transient tool plumbing rather than conversations the user had.
_HISTORY_SKIP_SOURCES = frozenset({"tool", "subagent", "cron"})

# config.yaml keys that describe THIS machine. Carrying them across would point the cloud agent
# at a local binary, a local path or a desktop-only device.
_MACHINE_LOCAL_CONFIG_KEYS = frozenset({
    "terminal", "browser", "computer_use", "bot_desktop", "desktop", "local_runtime", "proxy",
    "lsp", "wake_word", "voice", "worktrees", "checkpoints", "updates", "logging", "monitoring",
    "telemetry", "dashboard", "webapp", "gateway", "database", "runtime", "onboarding", "vault",
    "secrets", "plugins", "network",
})
_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|token|secret|passw(or)?d|credential|private[_-]?key|bearer)")


@dataclass
class BundleReport:
    """What a bundle carries and what it leaves behind, shown to the user before anything moves."""

    profile: str
    path: Optional[Path] = None
    bytes: int = 0
    sha256: str = ""
    skills: list[str] = field(default_factory=list)
    skills_skipped: list[dict[str, Any]] = field(default_factory=list)
    memory_entries: dict[str, int] = field(default_factory=dict)
    cron_jobs: list[dict[str, Any]] = field(default_factory=list)
    sessions: int = 0
    sessions_skipped: int = 0
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    keys_to_reenter: list[dict[str, str]] = field(default_factory=list)
    config_keys: list[str] = field(default_factory=list)
    config_dropped: list[str] = field(default_factory=list)
    has_soul: bool = False
    redacted: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile, "path": str(self.path) if self.path else None,
            "bytes": self.bytes, "sha256": self.sha256, "skills": self.skills,
            "skills_skipped": self.skills_skipped, "memory_entries": self.memory_entries,
            "cron_jobs": self.cron_jobs, "sessions": self.sessions,
            "sessions_skipped": self.sessions_skipped, "mcp_servers": self.mcp_servers,
            "keys_to_reenter": self.keys_to_reenter, "config_keys": self.config_keys,
            "config_dropped": self.config_dropped, "has_soul": self.has_soul,
            "redacted": self.redacted,
        }


# --------------------------------------------------------------------------- reading the source


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None


def _raw_config(home: Path) -> dict[str, Any]:
    from hermes_cli.config import read_user_config_raw
    data = read_user_config_raw(home / "config.yaml")
    return data if isinstance(data, dict) else {}


def _scrub_secrets(value: Any) -> Any:
    """Drop credential-shaped keys at any depth; keep everything else as written."""
    if isinstance(value, dict):
        return {k: _scrub_secrets(v) for k, v in value.items()
                if not (isinstance(k, str) and _SECRET_KEY_RE.search(k) and isinstance(v, (str, int)))}
    if isinstance(value, list):
        return [_scrub_secrets(v) for v in value]
    return value


def portable_config(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``(config, dropped_top_level_keys)``: machine-local sections and credential values removed."""
    dropped = sorted(k for k in raw if k in _MACHINE_LOCAL_CONFIG_KEYS or k.startswith("_"))
    kept = {k: v for k, v in raw.items() if k not in dropped}
    return _scrub_secrets(kept), dropped


def _mcp_inventory(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Each MCP server with whether it can run anywhere but this machine."""
    servers = raw.get("mcp_servers")
    if not isinstance(servers, dict):
        return []
    out = []
    for name, spec in sorted(servers.items()):
        spec = spec if isinstance(spec, dict) else {}
        url = spec.get("url")
        command = spec.get("command")
        local_path = isinstance(command, str) and (command.startswith(("/", "~", ".")) or "\\" in command)
        out.append({
            "name": str(name), "transport": "http" if url else "stdio",
            "portable": bool(url) or (bool(command) and not local_path),
            "reason": None if url or (command and not local_path) else
                      "runs a program from this computer's disk" if local_path else "has no url or command",
        })
    return out


def _env_key_names(home: Path) -> list[str]:
    text = _read_text(home / ".env") or ""
    names = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        if key and value.strip().strip("\"'"):
            names.append(key)
    return names


# Keys the cloud provides itself (Nous-managed tool gateway, portal identity) or that only mean
# something on this machine. The user is never asked to re-enter these.
_CLOUD_PROVIDED_KEYS = frozenset({
    "FIRECRAWL_API_KEY", "FAL_KEY", "TOOL_GATEWAY_USER_TOKEN", "BROWSERBASE_API_KEY",
    "BROWSERBASE_PROJECT_ID", "NOUS_API_KEY", "API_SERVER_KEY", "HERMES_ANON_API_SECRET",
})
_NEVER_MOVE_KEYS = frozenset({"SUDO_PASSWORD"})


def keys_to_reenter(home: Path) -> list[dict[str, str]]:
    """Secret names set on this machine that the cloud agent will need, with a category label.

    Values are never read into the result. Platform tokens (Telegram, Discord, iMessage…) are
    listed separately: a bot token can only be held by one gateway at a time, so moving it while
    the local gateway still runs disconnects one of them.
    """
    from hermes_cli.config import OPTIONAL_ENV_VARS
    out = []
    for key in _env_key_names(home):
        if key in _CLOUD_PROVIDED_KEYS or key in _NEVER_MOVE_KEYS:
            continue
        meta = OPTIONAL_ENV_VARS.get(key) or {}
        category = meta.get("category") or ("secret" if _SECRET_KEY_RE.search(key) else "")
        if not category or (category not in {"provider", "tool", "skill", "messaging", "secret"}):
            continue
        if category == "messaging" and not meta.get("password"):
            continue
        out.append({"key": key, "category": category, "label": str(meta.get("prompt") or key)})
    return sorted(out, key=lambda item: (item["category"], item["key"]))


# --------------------------------------------------------------------------- secret values


# auth.json fields that hold a credential value (``access_token``, ``api_key``, ``client_secret``…),
# not the labels beside them (``secret_source``, ``key_id``, ``token_type``).
_AUTH_SECRET_FIELD_RE = re.compile(r"(?i)(token|secret|api_?key|password|credential)$")
_AUTH_SECRET_MIN_LEN = 16
_MIN_SECRET_LEN = 8


def _known_secret_values(home: Path) -> dict[str, str]:
    """Every credential VALUE this machine holds (``.env`` + ``auth.json``), keyed by value.

    Transcripts and skills can quote a key the user pasted once; the bundle replaces each exact
    value with a named placeholder before it is written, so no stored credential leaves the
    machine even inside text.
    """
    values: dict[str, str] = {}
    for line in (_read_text(home / ".env") or "").splitlines():
        key, sep, value = line.strip().partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        if sep and key and not key.startswith("#") and len(value) >= _MIN_SECRET_LEN and _SECRET_KEY_RE.search(key):
            values[value] = key

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and len(v) >= _AUTH_SECRET_MIN_LEN and _AUTH_SECRET_FIELD_RE.search(str(k)):
                    values.setdefault(v, f"auth.json:{k}")
                else:
                    walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    try:
        walk(json.loads((home / "auth.json").read_text(encoding="utf-8-sig")), "")
    except (OSError, ValueError):
        pass
    return values


class _Redactor:
    def __init__(self, values: dict[str, str]):
        # Longest first so a value containing another is replaced whole.
        self._pairs = sorted(values.items(), key=lambda kv: len(kv[0]), reverse=True)
        self.replaced: dict[str, int] = {}

    def text(self, value: str) -> str:
        for secret, name in self._pairs:
            if secret in value:
                self.replaced[name] = self.replaced.get(name, 0) + value.count(secret)
                value = value.replace(secret, f"[redacted {name}]")
        return value

    def data(self, raw: bytes) -> bytes:
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw
        cleaned = self.text(decoded)
        return raw if cleaned is decoded else cleaned.encode("utf-8")


# --------------------------------------------------------------------------- skills


def _bundled_hashes(skills_root: Path) -> dict[str, str]:
    path = skills_root / ".bundled_manifest"
    out: dict[str, str] = {}
    for line in (_read_text(path) or "").splitlines():
        name, sep, digest = line.strip().partition(":")
        if name and sep:
            out[name.strip()] = digest.strip()
    return out


def _skill_dirs(skills_root: Path) -> Iterator[Path]:
    from agent.skill_utils import is_excluded_skill_path
    for skill_md in sorted(skills_root.rglob("SKILL.md")):
        rel = skill_md.relative_to(skills_root)
        if any(part in _SKILL_SKIP_DIRS for part in rel.parts) or is_excluded_skill_path(rel, root=skills_root):
            continue
        yield skill_md.parent


def _skill_files(skill_dir: Path) -> Iterator[Path]:
    for path in sorted(skill_dir.rglob("*")):
        rel = path.relative_to(skill_dir)
        if any(part in _SKILL_SKIP_DIRS for part in rel.parts) or path.suffix in _SKILL_SKIP_SUFFIXES:
            continue
        if path.is_file() and not path.is_symlink():
            yield path


def _skill_is_pristine_bundled(skill_dir: Path, bundled: dict[str, str]) -> bool:
    """True for an unmodified copy of a skill Hermes ships: the target already has it."""
    from tools.skill_usage import read_skill_name
    from tools.skills_sync import _matches_origin_hash
    name = read_skill_name(skill_dir / "SKILL.md", skill_dir.name)
    digest = bundled.get(name) or bundled.get(skill_dir.name)
    return bool(digest) and _matches_origin_hash(skill_dir, digest)


def _plan_skills(skills_root: Path) -> tuple[list[tuple[str, Path, list[Path]]], list[dict[str, Any]]]:
    """``(carried, skipped)``: carried = ``(rel_dir, dir, files)``; skipped carries a reason."""
    if not skills_root.is_dir():
        return [], []
    bundled = _bundled_hashes(skills_root)
    carried, skipped = [], []
    nested: list[Path] = []
    # Shallowest first, so a skill nested inside another skill's tree is seen after its parent.
    for skill_dir in sorted(_skill_dirs(skills_root), key=lambda p: (len(p.parts), str(p))):
        # A skill inside another skill's tree travels with its parent.
        if any(skill_dir.is_relative_to(parent) for parent in nested):
            continue
        nested.append(skill_dir)
        rel = skill_dir.relative_to(skills_root).as_posix()
        if _skill_is_pristine_bundled(skill_dir, bundled):
            continue
        files = list(_skill_files(skill_dir))
        size = sum(f.stat().st_size for f in files)
        if size > SKILL_MAX_BYTES:
            skipped.append({"skill": rel, "bytes": size, "reason": "too_large"})
            continue
        carried.append((rel, skill_dir, files))
    return carried, skipped


# --------------------------------------------------------------------------- cron


_CRON_RUN_STATE_KEYS = frozenset({
    "last_run_at", "last_status", "last_error", "last_delivery_error", "last_delivery_unverified",
    "last_failure", "failure_streak", "next_run_at", "monitor_state", "state", "paused_at",
    "paused_reason", "enabled",
})


def portable_cron_jobs(home: Path) -> list[dict[str, Any]]:
    """Job definitions without run state, each tagged with whether it needs this machine."""
    try:
        data = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    jobs = data.get("jobs") if isinstance(data, dict) else data
    out = []
    for job in jobs if isinstance(jobs, list) else []:
        if not isinstance(job, dict) or not job.get("id"):
            continue
        clean = {k: v for k, v in job.items() if k not in _CRON_RUN_STATE_KEYS}
        local = [k for k in ("script", "monitor_script", "workdir", "interpreter") if job.get(k)]
        clean["_move"] = {"needs_this_machine": local, "was_enabled": bool(job.get("enabled", True))}
        out.append(clean)
    return out


# --------------------------------------------------------------------------- sessions


def _export_sessions(home: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    """The ``limit`` most recent conversations as portable exports, oldest first."""
    db_path = home / "state.db"
    if limit <= 0 or not db_path.is_file():
        return [], 0
    from hermes_state import SessionDB
    db = SessionDB(db_path=db_path, read_only=True)
    try:
        rows = db.list_sessions_rich(limit=limit * 3, order_by_last_active=True, include_archived=False)
        rows = [r for r in rows if (r.get("source") or "") not in _HISTORY_SKIP_SOURCES][:limit]
        exports, skipped = [], 0
        for row in reversed(rows):
            # A compressed conversation is a chain of rows; export the whole chain as one session.
            data = (db.export_session_lineage(row["id"], include_inactive=True)
                    or db.export_session(row["id"], include_inactive=True))
            if not data or not data.get("messages"):
                continue
            for key in ("timings", "segments", "lineage_session_ids", "parent_session_id"):
                data.pop(key, None)
            # The receiver's import caps one session at 5 MiB and 10k messages; skip rather than fail.
            size = len(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode())
            if size > _SESSION_FILE_MAX_BYTES or len(data["messages"]) > 10_000:
                skipped += 1
                continue
            exports.append(data)
        return exports, skipped
    finally:
        db.close()


# --------------------------------------------------------------------------- assembly


def _host_label() -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", socket.gethostname().split(".")[0])[:40] or "computer"


def _tool_version() -> str:
    try:
        from importlib import metadata
        return metadata.version("hermes-agent")
    except Exception:  # health: allow BLE001 -- version is informational only
        return "unknown"


def _memory_entries(text: Optional[str]) -> int:
    from tools.memory_tool_store import ENTRY_DELIMITER
    return len([e for e in (text or "").split(ENTRY_DELIMITER) if e.strip()])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(home: Path, profile: str, out_path: Path, *, include_history: bool = True,
                 history_limit: int = HISTORY_DEFAULT_LIMIT, dry_run: bool = False) -> BundleReport:
    """Write the move bundle for ``home`` to ``out_path`` (or only inventory it with ``dry_run``)."""
    from hermes_cli.config_backend import require_file_tooling
    require_file_tooling("Moving to Hermes Cloud")
    report = BundleReport(profile=profile)
    raw = _raw_config(home)
    config, dropped = portable_config(raw)
    report.config_keys, report.config_dropped = sorted(config), dropped
    report.mcp_servers = _mcp_inventory(raw)
    report.keys_to_reenter = keys_to_reenter(home)
    soul = _read_text(home / "SOUL.md")
    report.has_soul = bool(soul and soul.strip())
    memory = {name: _read_text(home / "memories" / name) for name in ("MEMORY.md", "USER.md")}
    report.memory_entries = {name: _memory_entries(text) for name, text in memory.items() if text}
    carried, report.skills_skipped = _plan_skills(home / "skills")
    report.skills = [rel for rel, _dir, _files in carried]
    jobs = portable_cron_jobs(home)
    report.cron_jobs = [{"id": j["id"], "name": j.get("name") or j["id"],
                         "needs_this_machine": j["_move"]["needs_this_machine"]} for j in jobs]
    sessions, report.sessions_skipped = _export_sessions(home, history_limit if include_history else 0)
    report.sessions = len(sessions)
    if dry_run:
        return report

    import hermes_yaml as yaml
    manifest = {
        "schema_version": MOVE_SCHEMA_VERSION, "kind": "hermes-move-bundle",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"host": _host_label(), "platform": platform.system().lower(),
                   "profile": profile, "tool_version": _tool_version()},
        "inventory": report.as_dict(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.name + ".partial")
    red = _Redactor(_known_secret_values(home))
    members: list[tuple[str, bytes]] = [
        ("config.yaml", red.text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True)).encode()),
    ]
    if report.has_soul and soul:
        members.append(("SOUL.md", red.text(soul).encode()))
    members += [(f"memories/{name}", red.text(text).encode()) for name, text in memory.items() if text]
    for rel, skill_dir, files in carried:
        members += [(f"skills/{rel}/{path.relative_to(skill_dir).as_posix()}", red.data(path.read_bytes()))
                    for path in files]
    if jobs:
        members.append(("cron/jobs.json", red.text(json.dumps({"jobs": jobs}, indent=2, ensure_ascii=False)).encode()))
    members += [(f"sessions/{index:04d}.json", red.text(json.dumps(session, ensure_ascii=False)).encode())
                for index, session in enumerate(sessions)]
    report.redacted = dict(sorted(red.replaced.items()))
    manifest["inventory"]["redacted"] = report.redacted
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))
        for name, data in members:
            zf.writestr(name, data)
    os.replace(partial, out_path)
    try:
        os.chmod(out_path, 0o600)
    except OSError:
        pass
    report.path, report.bytes, report.sha256 = out_path, out_path.stat().st_size, _sha256(out_path)
    return report


def read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    """The bundle's manifest, or ``ValueError`` naming why this zip is not a move bundle."""
    try:
        manifest = json.loads(zf.read(MANIFEST_NAME))
    except KeyError:
        raise ValueError("This file is not a Hermes move bundle (no move-manifest.json).") from None
    except ValueError:
        raise ValueError("The move bundle's manifest is unreadable.") from None
    if not isinstance(manifest, dict) or manifest.get("kind") != "hermes-move-bundle":
        raise ValueError("This file is not a Hermes move bundle.")
    version = manifest.get("schema_version")
    if not isinstance(version, int) or version > MOVE_SCHEMA_VERSION:
        raise ValueError(
            f"This move bundle was made by a newer Hermes (format {version}). "
            f"Update this agent, then send it again.")
    return manifest
