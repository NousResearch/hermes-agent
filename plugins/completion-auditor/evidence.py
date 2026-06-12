"""In-memory evidence ledger for completion-auditor."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_LEDGER_TTL_SECONDS = 60 * 60
_DEFAULT_MAX_LEDGER_TURNS = 512

_KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)\b(?P<key>api[_-]?key|token|secret|password|passwd|pwd|authorization)\b"
    r"(?P<sep>\s*[:=]\s*)(?P<quote>['\"]?)(?P<value>[^'\"\s,;}]+)(?P=quote)"
)
_QUOTED_KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)(?P<prefix>['\"](?:api[_-]?key|token|secret|password|passwd|pwd|authorization)['\"]\s*:\s*)"
    r"(?P<quote>['\"])[^'\"]+(?P=quote)"
)
_BEARER_SECRET_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{12,}")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b")
_VERIFICATION_COMMAND_RE = re.compile(
    r"\b(pytest|unittest|tox|nox|ruff|mypy|pyright)\b"
    r"|\b(python\d?(?:\.\d+)?|uv)\s+(?:run\s+)?(?:-m\s+)?pytest\b"
    r"|\b(npm|pnpm|yarn)\s+(?:run\s+)?(test|lint|typecheck|build)\b"
    r"|\b(cargo|go|swift|gradle|mvn|make)\s+test\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceRef:
    """Metadata-only reference to a tool call observed in the current turn."""

    evidence_id: str
    tool_name: str
    status: str
    tool_call_id: str | None = None
    duration_ms: float | int | None = None
    exit_code: int | None = None
    arg_refs: list[str] = field(default_factory=list)
    command_kind: str | None = None
    result_excerpt: str | None = None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "evidence_id": self.evidence_id,
            "tool_name": self.tool_name,
            "status": self.status,
        }
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        if self.exit_code is not None:
            data["exit_code"] = self.exit_code
        if self.arg_refs:
            data["arg_refs"] = self.arg_refs
        if self.command_kind is not None:
            data["command_kind"] = self.command_kind
        if self.result_excerpt is not None:
            data["result_excerpt"] = self.result_excerpt
        return data


@dataclass
class TurnEvidence:
    session_id: str
    turn_id: str
    task_id: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    last_seen_at: float = field(default_factory=time.monotonic)
    evidence: list[EvidenceRef] = field(default_factory=list)

    def to_refs(self) -> list[dict[str, Any]]:
        return [item.to_json() for item in self.evidence]


_LOCK = threading.Lock()
_LEDGER: dict[tuple[str, str], TurnEvidence] = {}


def redact_text(text: str) -> str:
    """Mask common secret shapes before optional excerpts are persisted."""

    text = _BEARER_SECRET_RE.sub("Bearer [REDACTED]", text)
    text = _QUOTED_KEY_VALUE_SECRET_RE.sub(
        lambda m: f"{m.group('prefix')}{m.group('quote')}[REDACTED]{m.group('quote')}",
        text,
    )
    text = _KEY_VALUE_SECRET_RE.sub(
        lambda m: f"{m.group('key')}{m.group('sep')}[REDACTED]", text
    )
    text = _OPENAI_KEY_RE.sub("sk-[REDACTED]", text)
    return _GITHUB_TOKEN_RE.sub("gh_[REDACTED]", text)


# Backwards-compatible private alias used by tests and older slices.
_redact = redact_text


def _prune_locked(
    *,
    now: float | None = None,
    ttl_seconds: int = _DEFAULT_LEDGER_TTL_SECONDS,
    max_turns: int = _DEFAULT_MAX_LEDGER_TURNS,
) -> None:
    """Bound ledger memory for long-running gateway processes."""

    now = time.monotonic() if now is None else now
    if ttl_seconds > 0:
        expired = [
            key for key, turn in _LEDGER.items() if now - turn.last_seen_at > ttl_seconds
        ]
        for key in expired:
            _LEDGER.pop(key, None)
    if max_turns > 0 and len(_LEDGER) > max_turns:
        by_age = sorted(_LEDGER.items(), key=lambda item: item[1].last_seen_at)
        for key, _ in by_age[: len(_LEDGER) - max_turns]:
            _LEDGER.pop(key, None)


def _stable_evidence_id(
    session_id: str,
    turn_id: str,
    tool_name: str,
    tool_call_id: str | None,
    index: int,
) -> str:
    raw = json.dumps(
        {
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "index": index,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "ev_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _path_like(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if "/" in text or re.search(
        r"\.(py|js|ts|tsx|json|ya?ml|md|txt|toml|ini|sh|bash|css|html)$", text
    ):
        return text[:500]
    return None


def _arg_refs(args: Any) -> list[str]:
    if not isinstance(args, dict):
        return []
    refs: list[str] = []
    for key in ("path", "file", "file_path", "target", "output_path"):
        ref = _path_like(args.get(key))
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _command_kind(tool_name: str, args: Any) -> str | None:
    if tool_name == "execute_code":
        return "verification"
    if tool_name != "terminal" or not isinstance(args, dict):
        return None
    command = args.get("command")
    if not isinstance(command, str):
        return None
    return "verification" if _VERIFICATION_COMMAND_RE.search(command) else "other"


def _result_exit_code(result: Any) -> int | None:
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    raw = data.get("exit_code")
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def _excerpt(
    result: Any, *, include: bool, max_chars: int, redact_secrets: bool
) -> str | None:
    if not include:
        return None
    text = result if isinstance(result, str) else repr(result)
    if redact_secrets:
        text = redact_text(text)
    if max_chars <= 0:
        return ""
    return text[:max_chars]


def record_tool_call(
    *,
    session_id: str | None = None,
    turn_id: str | None = None,
    task_id: str | None = None,
    tool_name: str = "",
    tool_call_id: str | None = None,
    status: str | None = None,
    duration_ms: float | int | None = None,
    args: Any = None,
    result: Any = None,
    include_result_excerpt: bool = False,
    max_result_excerpt_chars: int = 800,
    redact_secrets: bool = True,
    **_: Any,
) -> bool:
    """Record one post_tool_call event.

    Returns ``False`` when the hook payload cannot be correlated to a turn.
    """
    if not session_id or not turn_id:
        return False
    key = (str(session_id), str(turn_id))
    with _LOCK:
        _prune_locked()
        turn = _LEDGER.get(key)
        if turn is None:
            turn = TurnEvidence(
                session_id=str(session_id), turn_id=str(turn_id), task_id=task_id
            )
            _LEDGER[key] = turn
        elif turn.task_id and task_id and turn.task_id != task_id:
            return False
        elif not turn.task_id and task_id:
            turn.task_id = task_id
        turn.last_seen_at = time.monotonic()
        index = len(turn.evidence)
        tool_name_text = str(tool_name or "unknown")
        turn.evidence.append(
            EvidenceRef(
                evidence_id=_stable_evidence_id(
                    str(session_id), str(turn_id), tool_name_text, tool_call_id, index
                ),
                tool_name=tool_name_text,
                status=str(status or "unknown"),
                tool_call_id=tool_call_id,
                duration_ms=duration_ms,
                exit_code=_result_exit_code(result),
                arg_refs=_arg_refs(args),
                command_kind=_command_kind(tool_name_text, args),
                result_excerpt=_excerpt(
                    result,
                    include=include_result_excerpt,
                    max_chars=max_result_excerpt_chars,
                    redact_secrets=redact_secrets,
                ),
            )
        )
    return True


def pop_turn(session_id: str | None, turn_id: str | None) -> TurnEvidence | None:
    """Return and remove evidence for a completed turn."""
    if not session_id or not turn_id:
        return None
    with _LOCK:
        _prune_locked()
        return _LEDGER.pop((str(session_id), str(turn_id)), None)


def clear() -> None:
    """Clear the in-memory ledger. Intended for tests."""
    with _LOCK:
        _LEDGER.clear()


def size() -> int:
    with _LOCK:
        _prune_locked()
        return len(_LEDGER)
