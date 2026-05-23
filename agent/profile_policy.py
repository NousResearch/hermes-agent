"""Profile-scoped fail-closed policy guards.

Stage 4B: narrow guard for the hephaestus-h profile.  The module is
intentionally cold-loaded at every public check so policy.yaml changes do not
require a gateway restart.
"""
from __future__ import annotations

import logging
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except Exception:  # pragma: no cover
    logging.getLogger(__name__).warning("PyYAML import failed; profile policy loading disabled", exc_info=True)
    yaml = None  # type: ignore

GUARDED_PROFILES = {"hephaestus-h", "cassandra-h"}
GUARDED_PROFILE = "hephaestus-h"  # Backcompat for existing tests/references.
EXPECT_ENV = "HERMES_PROFILE_GUARD_EXPECTED"

_TOKEN_PATTERNS = (
    re.compile(r"\b\d+:[A-Za-z0-9_-]{35}\b"),
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|xox[pbar]-[A-Za-z0-9-]{20,})\b"),
)
_HIGH_ENTROPY_RE = re.compile(r"\b[A-Za-z0-9+/=_-]{40,}\b")


class ProfilePolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProfilePolicy:
    profile: str
    path: Path
    allowed_write_roots: tuple[Path, ...]
    forbidden_basenames: tuple[str, ...]
    forbidden_path_substrings: tuple[str, ...]
    allowed_chat_ids: tuple[str, ...]
    dm_only: bool = True
    allow_bots: bool = False
    state_dir: Path | None = None
    idempotency_ttl_seconds: int = 24 * 3600


def _active_profile_name() -> str:
    from hermes_cli.profiles import get_active_profile_name

    return get_active_profile_name()


def _guard_expected() -> str:
    return (os.getenv(EXPECT_ENV) or "").strip()


def _is_guard_context(profile: str) -> bool:
    expected = _guard_expected()
    return profile in GUARDED_PROFILES or expected in GUARDED_PROFILES


def _ensure_guard_profile() -> str | None:
    profile = _active_profile_name()
    expected = _guard_expected()
    if expected in GUARDED_PROFILES and profile != expected:
        raise ProfilePolicyError(
            f"profile guard expected {expected}, got {profile or 'unresolved'}"
        )
    if profile not in GUARDED_PROFILES:
        return None
    if profile in {"", "default", "custom"}:
        raise ProfilePolicyError(f"guarded profile identity unresolved: {profile!r}")
    return profile


def _as_list(data: Mapping[str, Any], key: str) -> list[str]:
    value = data.get(key) or []
    if not isinstance(value, list):
        raise ProfilePolicyError(f"policy field {key} must be a list")
    return [str(v) for v in value]


def _load_policy() -> ProfilePolicy | None:
    profile = _ensure_guard_profile()
    if profile is None:
        return None
    if yaml is None:
        raise ProfilePolicyError("PyYAML unavailable; cannot load profile policy")
    from hermes_constants import get_hermes_home

    policy_path = get_hermes_home() / "policy.yaml"
    try:
        raw = yaml.safe_load(policy_path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ProfilePolicyError(f"missing guarded profile policy: {policy_path}") from exc
    except Exception as exc:
        raise ProfilePolicyError(f"failed to read policy: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfilePolicyError("policy root must be a mapping")
    identity = str(raw.get("identity") or "")
    if identity != profile:
        raise ProfilePolicyError(f"policy identity mismatch: {identity!r}; expected {profile!r}")

    roots = tuple(Path(p).expanduser().resolve() for p in _as_list(raw, "allowed_write_roots"))
    if not roots:
        raise ProfilePolicyError("allowed_write_roots must not be empty")
    telegram = raw.get("telegram") or {}
    if not isinstance(telegram, dict):
        raise ProfilePolicyError("telegram policy must be a mapping")
    loop = raw.get("loop") or {}
    state_raw = raw.get("state_dir") or f"/root/agent-workspaces/companions/{profile}/state"
    return ProfilePolicy(
        profile=profile,
        path=policy_path,
        allowed_write_roots=roots,
        forbidden_basenames=tuple(_as_list(raw, "forbidden_basenames") or [".env", ".session"]),
        forbidden_path_substrings=tuple(_as_list(raw, "forbidden_path_substrings") or ["token", "secret", "key"]),
        allowed_chat_ids=tuple(str(x) for x in (telegram.get("allowed_chat_ids") or [])),
        dm_only=bool(telegram.get("dm_only", True)),
        allow_bots=bool(telegram.get("allow_bots", False)),
        state_dir=Path(str(state_raw)).expanduser().resolve(),
        idempotency_ttl_seconds=int(loop.get("idempotency_ttl_seconds") or 24 * 3600),
    )


def _relative_to_any(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _check_path(
    policy: ProfilePolicy,
    path_value: str,
    *,
    purpose: str = "write",
    base_dir: str | Path | None = None,
) -> None:
    if not path_value:
        return
    raw_path = Path(path_value).expanduser()
    if not raw_path.is_absolute() and base_dir is not None:
        raw_path = Path(base_dir).expanduser() / raw_path
    path = raw_path.resolve()
    if path.name in policy.forbidden_basenames:
        raise ProfilePolicyError(f"{purpose} blocked: forbidden basename {path.name}")
    path_str = str(path).lower()
    for marker in policy.forbidden_path_substrings:
        if marker and marker.lower() in path_str:
            raise ProfilePolicyError(f"{purpose} blocked: forbidden path marker {marker!r}")
    if not _relative_to_any(path, policy.allowed_write_roots):
        raise ProfilePolicyError(f"{purpose} blocked outside allowed roots: {path}")




def _extract_v4a_patch_paths(patch_text: str) -> list[str]:
    """Extract target paths from V4A-style multi-file patch text."""
    paths: list[str] = []
    for line in (patch_text or "").splitlines():
        line = line.strip()
        for prefix in ("*** Update File:", "*** Add File:", "*** Delete File:"):
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                if value:
                    paths.append(value)
                break
    return paths


def _check_patch_payload(policy: ProfilePolicy, patch_text: str) -> None:
    paths = _extract_v4a_patch_paths(patch_text)
    if not paths:
        raise ProfilePolicyError("patch blocked: no target paths found in patch payload")
    for path in paths:
        _check_path(policy, path, purpose="patch")


def _shell_write_candidates(command: str) -> list[str]:
    """Best-effort extraction of paths written by simple shell commands.

    This is intentionally conservative for guarded profiles: if we recognize a
    write primitive, validate the destination operand rather than the source.
    Complex shell syntax is still covered by the base dangerous-command policy
    and higher-level sandboxing.
    """
    import shlex

    candidates: list[str] = []
    redir_ops = {">", ">>", "&>", "1>", "1>>", "2>", "2>>"}
    for segment in re.split(r"\s*(?:&&|\|\||;|\|)\s*", command):
        try:
            parts = shlex.split(segment)
        except ValueError:
            continue
        if not parts:
            continue

        idx = 0
        while idx < len(parts):
            token = parts[idx]
            if token in redir_ops and idx + 1 < len(parts):
                candidates.append(parts[idx + 1])
                idx += 2
                continue
            redir_match = re.match(r"^(?:\d?>?>|&>)(.+)$", token)
            if redir_match:
                candidates.append(redir_match.group(1))
            idx += 1

        cmd = Path(parts[0]).name

        def _option_value(flag: str) -> str | None:
            for pos, part in enumerate(parts[1:], start=1):
                if part == flag and pos + 1 < len(parts):
                    return parts[pos + 1]
                prefix = flag + "="
                if part.startswith(prefix):
                    return part[len(prefix):]
            return None

        if cmd in {"cp", "mv", "install"}:
            target_dir = _option_value("--target-directory") or _option_value("-t")
            if target_dir:
                candidates.append(target_dir)
            else:
                operands = [p for p in parts[1:] if not p.startswith("-")]
                if len(operands) >= 2:
                    candidates.append(operands[-1])
        elif cmd == "tee":
            operands = [p for p in parts[1:] if not p.startswith("-")]
            candidates.extend(operands)
        elif cmd in {"touch", "mkdir"}:
            operands = [p for p in parts[1:] if not p.startswith("-")]
            candidates.extend(operands)
        elif cmd == "rsync":
            operands = [p for p in parts[1:] if not p.startswith("-")]
            if len(operands) >= 2:
                candidates.append(operands[-1])
        elif cmd == "dd":
            for part in parts[1:]:
                if part.startswith("of=") and len(part) > 3:
                    candidates.append(part[3:])
    return candidates



def _terminal_segment_is_allowed(segment: str) -> bool:
    """Allow only read-only commands or write primitives whose destinations we parse."""
    import shlex

    # Shell command substitution can execute hidden side-effecting commands
    # before an otherwise read-only allowlisted command runs. Block it globally.
    if "$(" in segment or "`" in segment:
        return False

    try:
        parts = shlex.split(segment)
    except ValueError:
        return False
    if not parts:
        return True
    cmd = Path(parts[0]).name
    write_cmds = {"cp", "mv", "install", "tee", "touch", "mkdir", "rsync", "dd"}
    if cmd in write_cmds:
        return True
    # echo/printf are safe only when they do not contain command substitution;
    # any redirection destination is parsed and checked separately.
    if cmd in {"echo", "printf"}:
        return "$(" not in segment and "`" not in segment
    if cmd in {"pwd", "true", "false", "test", "[", "which", "command", "readlink", "realpath", "date", "hostname", "uname", "id", "whoami"}:
        return True
    if cmd in {"grep", "rg", "wc", "head", "tail", "cat"}:
        return True
    if cmd == "find":
        blocked_find_actions = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"}
        return not any(part in blocked_find_actions for part in parts[1:])
    if cmd == "sed":
        return not any(part == "-i" or part.startswith("-i") for part in parts[1:])
    if cmd == "git":
        readonly = {"status", "diff", "log", "show", "rev-parse", "ls-files", "grep", "branch", "remote"}
        return len(parts) >= 2 and parts[1] in readonly
    if cmd in {"python", "python3"}:
        return len(parts) >= 3 and parts[1] == "-m" and parts[2] in {"pytest", "py_compile"}
    return False


def _terminal_command_is_allowed(command: str) -> bool:
    for segment in re.split(r"\s*(?:&&|\|\||;|\|)\s*", command):
        if segment.strip() and not _terminal_segment_is_allowed(segment):
            return False
    return True

def check_tool_call(tool_name: str, args: Mapping[str, Any] | None) -> None:
    """Fail-closed policy check before executing a tool call."""
    policy = _load_policy()
    if policy is None:
        return
    args = args or {}
    if tool_name == "write_file":
        path = args.get("path")
        if isinstance(path, str):
            _check_path(policy, path, purpose=tool_name)
    elif tool_name == "patch":
        mode = str(args.get("mode") or "replace")
        if mode == "patch" or args.get("patch"):
            patch_text = args.get("patch")
            if isinstance(patch_text, str):
                _check_patch_payload(policy, patch_text)
            else:
                raise ProfilePolicyError("patch blocked: patch payload missing")
        else:
            path = args.get("path")
            if isinstance(path, str):
                _check_path(policy, path, purpose=tool_name)
            else:
                raise ProfilePolicyError("patch blocked: path missing")
    elif tool_name == "skill_manage":
        fp = args.get("file_path")
        if isinstance(fp, str) and fp:
            _check_path(policy, fp, purpose="skill_manage")
    elif tool_name == "terminal":
        command = str(args.get("command") or "")
        from tools.approval import detect_dangerous_command

        dangerous, _key, desc = detect_dangerous_command(command)
        if dangerous:
            raise ProfilePolicyError(f"terminal blocked by base dangerous-command policy: {desc}")
        workdir = args.get("workdir")
        base_dir = workdir if isinstance(workdir, str) and workdir else None
        for candidate in _shell_write_candidates(command):
            if candidate.startswith("-"):
                continue
            _check_path(policy, candidate, purpose="terminal", base_dir=base_dir)
        if not _terminal_command_is_allowed(command):
            raise ProfilePolicyError("terminal blocked: command is not in guarded-profile allowlist")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {ch: s.count(ch) for ch in set(s)}
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in counts.values())


def contains_secret(text: str) -> bool:
    if not text:
        return False
    if any(p.search(text) for p in _TOKEN_PATTERNS):
        return True
    for match in _HIGH_ENTROPY_RE.finditer(text):
        token = match.group(0)
        if _entropy(token) >= 4.5:
            return True
    return False


def check_outbound(platform: str, chat_id: str, content: str, *, kind: str = "text") -> None:
    policy = _load_policy()
    if policy is None:
        return
    if not policy.allowed_chat_ids:
        raise ProfilePolicyError("outbound blocked: no allowed chat ids configured")
    if str(chat_id) not in policy.allowed_chat_ids:
        raise ProfilePolicyError(f"outbound blocked: unauthorized chat_id {chat_id}")
    probe = content or ""
    if kind == "media":
        path = Path(probe.replace("file://", "")).expanduser()
        # Check path/name and small text-like contents for token leakage.
        if contains_secret(str(path)):
            raise ProfilePolicyError("outbound media blocked: secret-looking path")
        try:
            if path.is_file() and path.stat().st_size <= 1024 * 1024:
                sample = path.read_text(errors="ignore")[:200_000]
                if contains_secret(sample):
                    raise ProfilePolicyError("outbound media blocked: secret-looking content")
        except ProfilePolicyError:
            raise
        except Exception:
            pass
    elif contains_secret(probe):
        raise ProfilePolicyError("outbound text blocked: secret-looking content")


def _event_update_key(event: Any) -> str | None:
    update_id = getattr(event, "platform_update_id", None)
    if update_id is not None:
        return str(update_id)
    msg_id = getattr(event, "message_id", None) or getattr(getattr(event, "source", None), "message_id", None)
    return str(msg_id) if msg_id else None


def _mark_idempotent(policy: ProfilePolicy, platform: str, update_id: str) -> bool:
    if policy.state_dir is None:
        raise ProfilePolicyError("state_dir missing for idempotency")
    policy.state_dir.mkdir(parents=True, exist_ok=True)
    db_path = policy.state_dir / "idempotency.db"
    now = int(time.time())
    cutoff = now - int(policy.idempotency_ttl_seconds)
    last_error: sqlite3.OperationalError | None = None
    for _attempt in range(5):
        try:
            with sqlite3.connect(str(db_path), timeout=10) as conn:
                conn.execute("PRAGMA busy_timeout=10000")
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS inbound_idempotency ("
                    "platform TEXT NOT NULL, update_id TEXT NOT NULL, processed_at INTEGER NOT NULL, "
                    "PRIMARY KEY(platform, update_id))"
                )
                conn.execute("DELETE FROM inbound_idempotency WHERE processed_at < ?", (cutoff,))
                cur = conn.execute(
                    "INSERT OR IGNORE INTO inbound_idempotency(platform, update_id, processed_at) VALUES (?, ?, ?)",
                    (platform, update_id, now),
                )
                conn.commit()
                return cur.rowcount == 1
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            last_error = exc
            time.sleep(0.05)
    raise ProfilePolicyError(f"inbound idempotency store locked: {last_error}")


def check_inbound_event(event: Any) -> None:
    policy = _load_policy()
    if policy is None:
        return
    source = getattr(event, "source", None)
    if source is None:
        raise ProfilePolicyError("inbound blocked: missing source")
    platform = getattr(getattr(source, "platform", None), "value", None) or str(getattr(source, "platform", ""))
    chat_id = str(getattr(source, "chat_id", "") or "")
    chat_type = str(getattr(source, "chat_type", "") or "")
    if policy.dm_only and chat_type != "dm":
        raise ProfilePolicyError(f"inbound blocked: non-DM chat_type {chat_type!r}")
    if not policy.allow_bots and bool(getattr(source, "is_bot", False)):
        raise ProfilePolicyError("inbound blocked: bot sender")
    if not policy.allowed_chat_ids:
        raise ProfilePolicyError("inbound blocked: no allowed chat ids configured")
    if chat_id not in policy.allowed_chat_ids:
        raise ProfilePolicyError(f"inbound blocked: unauthorized chat_id {chat_id}")
    update_key = _event_update_key(event)
    if update_key and not _mark_idempotent(policy, platform, update_key):
        raise ProfilePolicyError(f"inbound blocked: duplicate update {platform}:{update_key}")
