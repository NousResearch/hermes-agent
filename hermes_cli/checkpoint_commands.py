"""StrongIQ/Pacey native chat checkpoint slash-command helpers.

These helpers are intentionally candidate-first. They inspect only the current
session window, redact risky material, write operator proof packets, and advance
checkpoint state only when an explicit approval flag is present. They do not
write Graphiti/Honcho/SOUL/reference surfaces directly.

Candidate classification is a lightweight English keyword heuristic for operator
review. It is not a policy engine, it is not multilingual, and it must not be
used to apply durable memory or law without human/agent review. Secret redaction
is best-effort regex redaction over snippets only; raw transcript text is never
included in packets.

Checkpoint boundaries are persisted in native ``SessionDB.state_meta`` using
one namespaced key per session, mirroring existing session-scoped features such
as goals and heartbeats.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

APPROVAL_FLAG = "--approve-candidate-checkpoint"
DRY_RUN_FLAGS = {"--dry-run", "--preview"}
META_PREFIX = "strongiq_checkpoint"

_SECRET_PATTERNS = [
    # Authorization schemes are open-ended (Bearer, Basic, Digest,
    # AWS4-HMAC-SHA256, Negotiate, custom schemes). Redact the whole header/value
    # to the line boundary so multi-token params cannot leak in checkpoint text.
    re.compile(r"(?im)\bauthorization\b\s*[:=]\s*[^\r\n]+"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
]
_LOOKBACK_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[mhd])$")


@dataclass(frozen=True)
class CheckpointResult:
    text: str
    packet_path: str
    state_path: str
    approved: bool
    candidates: list[dict[str, Any]]
    window: dict[str, Any]


def _workspace() -> Path:
    path = get_hermes_home() / "workspace" / "chat-checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _db_path() -> Path:
    return get_hermes_home() / "state.db"


def checkpoint_state_path() -> Path:
    """Return the native state store path used by this command.

    Kept for tests/proofs and compatibility with the first pass of this helper.
    It now points at ``state.db`` rather than a sidecar JSON file.
    """
    return _db_path()


def checkpoint_meta_key(session_id: str) -> str:
    return f"{META_PREFIX}:{session_id}"


def _get_session_db() -> Any | None:
    """Open a SessionDB bound to the active HERMES_HOME state.db.

    Do not reuse the goals/heartbeat cache here. Tests and profile-switched
    gateway processes can change HERMES_HOME during one Python process, and a
    cached DB from a previous home would advance the wrong checkpoint.
    """
    try:
        from hermes_state import SessionDB

        return SessionDB(db_path=_db_path())
    except Exception:
        return None


def _load_checkpoint(session_id: str) -> dict[str, Any]:
    if not session_id:
        return {}
    key = checkpoint_meta_key(session_id)
    raw: str | None = None
    db = _get_session_db()
    if db is not None:
        try:
            raw = db.get_meta(key)
        except Exception:
            raw = None
    if raw is None:
        # Test/minimal-DB fallback. Live runtime uses SessionDB.get_meta above.
        path = _db_path()
        if path.exists():
            try:
                with sqlite3.connect(str(path)) as conn:
                    row = conn.execute(
                        "SELECT value FROM state_meta WHERE key = ?",
                        (key,),
                    ).fetchone()
                raw = row[0] if row else None
            except Exception:
                raw = None
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_checkpoint(session_id: str, state: dict[str, Any]) -> None:
    if not session_id:
        return
    key = checkpoint_meta_key(session_id)
    value = json.dumps(state, sort_keys=True)
    db = _get_session_db()
    if db is not None:
        try:
            db.set_meta(key, value)
            return
        except Exception:
            pass
    # Test/minimal-DB fallback. Live runtime uses SessionDB.set_meta above.
    path = _db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def redact(text: str) -> str:
    out = str(text or "")

    def _replacement(match: re.Match[str]) -> str:
        value = match.group(0)
        auth_match = re.match(r"(?i)\s*(authorization)\s*[:=]", value)
        if auth_match:
            key = value[: auth_match.end(1)].strip()
            return f"{key}=[REDACTED]"
        if re.search(r"(?i)\b(api[_-]?key|token|secret|password)\b", value):
            sep = "=" if "=" in value else ":" if ":" in value else ""
            if sep:
                key = value.split(sep, 1)[0].strip()
                return f"{key}=[REDACTED]"
        return "[REDACTED]"

    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(_replacement, out)
    return out


def _parse_lookback(raw_args: str) -> tuple[int | None, str | None, str | None]:
    tokens = [t for t in str(raw_args or "").split() if t not in DRY_RUN_FLAGS and t != APPROVAL_FLAG]
    if not tokens:
        return None, None, None
    token = tokens[0].strip().lower()
    match = _LOOKBACK_RE.match(token)
    if not match:
        return None, None, f"Unsupported checkpoint lookback `{tokens[0]}`. Use forms like 90m, 12h, or 4d."
    value = int(match.group("value"))
    unit = match.group("unit")
    multiplier = {"m": 60, "h": 3600, "d": 86400}[unit]
    seconds = value * multiplier
    if seconds <= 0:
        return None, None, "Lookback must be greater than zero."
    return seconds, token, None


def _safe_packet_session_slug(session_id: str) -> str:
    raw = session_id or "unknown"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")[:64] or "unknown"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


def _read_messages(session_id: str, after_id: int = 0, limit: int = 200, since_timestamp: float | None = None) -> list[dict[str, Any]]:
    db_path = _db_path()
    if not db_path.exists() or not session_id:
        return []
    db = _get_session_db()
    if db is not None:
        try:
            rows = db.get_messages(session_id, after_id=int(after_id or 0), limit=int(limit))
            filtered = []
            for row in rows:
                if row.get("role") not in {"user", "assistant"}:
                    continue
                if since_timestamp is not None and float(row.get("timestamp") or 0) < since_timestamp:
                    continue
                filtered.append(
                    {
                        "id": row.get("id"),
                        "role": row.get("role"),
                        "content": row.get("content"),
                        "timestamp": row.get("timestamp"),
                    }
                )
            return filtered
        except Exception:
            # Minimal tests or old state stores may not have the full SessionDB
            # schema. Fall back to the narrow query used by this helper only.
            pass
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, role, content, timestamp
            FROM messages
            WHERE session_id = ? AND id > ? AND active = 1 AND role IN ('user', 'assistant')
              AND (? IS NULL OR timestamp >= ?)
            ORDER BY id ASC
            LIMIT ?
            """,
            (session_id, int(after_id or 0), since_timestamp, since_timestamp, int(limit)),
        ).fetchall()
    return [dict(row) for row in rows]


def _classify(text: str) -> str:
    lowered = text.lower()
    if any(w in lowered for w in ("prefer", "correction", "don't", "do not", "always", "never")):
        return "USER memory candidate"
    if any(w in lowered for w in ("rule", "law", "approval", "gate", "must", "mandatory")):
        return "SOUL-law candidate"
    if any(w in lowered for w in ("path", "runbook", "reference", "procedure", "standard")):
        return "reference-law candidate"
    if any(w in lowered for w in ("decided", "decision", "commitment", "owner", "deadline")):
        return "Graphiti fact candidate"
    return "no-promotion/session-only"


def _snippet(text: str, max_chars: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", redact(text)).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _build_candidates(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in messages:
        content = str(row.get("content") or "")
        if not content.strip():
            continue
        snippet = _snippet(content)
        if not snippet:
            continue
        candidates.append(
            {
                "message_id": int(row.get("id") or 0),
                "role": row.get("role"),
                "class": _classify(content),
                "classification_method": "english-keyword-heuristic",
                "classification_requires_review": True,
                "snippet": snippet,
            }
        )
    return candidates


def _write_packet(
    *,
    command: str,
    session_id: str,
    approved: bool,
    window: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe_command = command.replace("/", "").replace("-", "_")
    path = _workspace() / f"{stamp}-{safe_command}-{_safe_packet_session_slug(session_id)}.json"
    packet = {
        "command": command,
        "session_id": session_id,
        "approved": approved,
        "window": window,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "durable_writes": [] if not approved else ["native state_meta checkpoint boundary advanced", "approved candidate packet recorded"],
        "raw_transcript_included": False,
        "classification_policy": {
            "method": "english-keyword-heuristic",
            "limitations": [
                "candidate routing only; not a durable memory/law decision",
                "English keyword matching can misclassify negation and nuance",
                "non-English inputs may be missed",
            ],
            "requires_review": True,
        },
        "redaction_policy": {
            "scope": "snippets only",
            "method": "best-effort regex redaction",
            "raw_transcript_included": False,
            "limitations": ["not an exhaustive secret scanner"],
        },
        "state_store": "SessionDB.state_meta",
        "state_key": checkpoint_meta_key(session_id) if session_id else "",
        "created_at": stamp,
    }
    path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _advance_checkpoint(session_id: str, *, last_message_id: int, packet_path: Path) -> None:
    _save_checkpoint(
        session_id,
        {
            "last_message_id": int(last_message_id or 0),
            "packet_path": str(packet_path),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def run_checkpoint_command(command: str, *, session_id: str, args: str = "") -> CheckpointResult:
    raw_args = str(args or "")
    tokens = raw_args.split()
    approval_requested = APPROVAL_FLAG in tokens
    explicit_dry_run = any(flag in tokens for flag in DRY_RUN_FLAGS)
    approved = approval_requested and not explicit_dry_run
    dry_run = not approved
    lookback_seconds, lookback_raw, lookback_error = _parse_lookback(raw_args)
    if lookback_error:
        raise ValueError(lookback_error)

    prior = _load_checkpoint(session_id)
    prior_after_id = int(prior.get("last_message_id") or 0)
    since_timestamp = (time.time() - lookback_seconds) if lookback_seconds is not None else None
    after_id = 0 if lookback_seconds is not None else prior_after_id
    messages = _read_messages(session_id, after_id=after_id, since_timestamp=since_timestamp)
    candidates = _build_candidates(messages)
    last_message_id = max([int(m.get("id") or 0) for m in messages] or [after_id])
    if lookback_raw:
        scope = f"explicit lookback {lookback_raw}"
        boundary_mode = "lookback"
    else:
        scope = "since last checkpoint" if after_id else "session start"
        boundary_mode = "checkpoint" if after_id else "session_start"
    window = {
        "from_exclusive_message_id": after_id,
        "to_inclusive_message_id": last_message_id,
        "message_count": len(messages),
        "scope": scope,
        "boundary_mode": boundary_mode,
        "lookback_raw": lookback_raw,
        "since_timestamp": since_timestamp,
    }
    packet_path = _write_packet(
        command=command,
        session_id=session_id,
        approved=approved,
        window=window,
        candidates=candidates,
    )
    if approved:
        _advance_checkpoint(session_id, last_message_id=last_message_id, packet_path=packet_path)

    mode = "approved checkpoint boundary" if approved else "dry-run candidate packet"
    lines = [
        f"{command}: {mode}",
        f"Session: {session_id or 'unknown'}",
        f"Window: {window['scope']} ({window['from_exclusive_message_id']} → {window['to_inclusive_message_id']})",
        f"Messages analysed: {len(messages)}",
        f"Candidates: {len(candidates)}",
        f"Packet: {packet_path}",
        f"State: SessionDB.state_meta key `{checkpoint_meta_key(session_id) if session_id else ''}`",
    ]
    if approved:
        lines.append("Checkpoint boundary advanced in native SessionDB.state_meta. No Graphiti/Honcho/SOUL/reference writes were applied by this command.")
    else:
        if approval_requested and explicit_dry_run:
            lines.append(f"Approval flag was ignored because dry-run/preview was also supplied. No checkpoint boundary advanced.")
        lines.append(f"No durable memory/SOUL/reference writes were applied. To advance the checkpoint boundary only, rerun with `{APPROVAL_FLAG}` and without dry-run/preview.")
    return CheckpointResult(
        text="\n".join(lines),
        packet_path=str(packet_path),
        state_path=str(checkpoint_state_path()),
        approved=approved,
        candidates=candidates,
        window=window,
    )
