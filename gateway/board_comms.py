"""Board-room communication governor for Telegram gateway sends.

This module is intentionally deterministic and side-effect-light: it only
classifies proposed outbound Board replies, writes an append-only suppression
log, and returns ALLOW/SUPPRESS.  It does not call Telegram or mutate sessions.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - import safety for partial installs
    get_hermes_home = None  # type: ignore[assignment]


DEFAULT_BOARD_CHAT_IDS = frozenset({"-1003817293915"})
DEFAULT_ACK_WINDOW_SECONDS = 300

_ACK_RE = re.compile(
    r"^(?:\s*(?:prime\s+hermes\s*/\s*ezra|prime\s+hermes|ezra)\s*[:—\-–])?\s*"
    r"(?:ack(?:nowledged)?|received|logged|noted|standing\s+by|holding|copy|roger|understood)"
    r"(?:[.!\s]*(?:no\s+further\s+action(?:\s+needed)?|standing\s+by|with\s+you|board\s+thread\s+is\s+calm))*\s*[.!]*$",
    re.IGNORECASE | re.DOTALL,
)
_STOP_KILL_RE = re.compile(r"\b(?:stop|kill|hold|quiet|silence)\b", re.IGNORECASE)
_BOT_MENTION_RE = re.compile(r"@([A-Za-z0-9_]{4,64})")
_CRITICAL_RE = re.compile(
    r"\b(?:critical|failed|failure|error|blocked|blocker|degraded|unsafe|incident|outage|security|cannot|can't)\b",
    re.IGNORECASE,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _csv_set(raw: Optional[str]) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def board_chat_ids() -> set[str]:
    configured = _csv_set(os.environ.get("HERMES_BOARD_CHAT_IDS"))
    return configured or set(DEFAULT_BOARD_CHAT_IDS)


def own_bot_usernames() -> set[str]:
    configured = _csv_set(os.environ.get("HERMES_BOARD_OWN_BOT_USERNAMES"))
    defaults = {"mac_pro_hermes_bot", "hermes_ezra_ai_bot", "ezra_ai_bot", "hermes_bot"}
    return {u.lstrip("@").lower() for u in (configured or defaults)}


def agent_name() -> str:
    return os.environ.get("HERMES_BOARD_AGENT_NAME", "Prime Hermes / Ezra")


def _state_root() -> Path:
    if os.environ.get("HERMES_BOARD_COMMS_STATE_DIR"):
        return Path(os.environ["HERMES_BOARD_COMMS_STATE_DIR"])
    if get_hermes_home is not None:
        try:
            return Path(get_hermes_home()) / "state" / "board-comms"
        except Exception:
            pass
    return Path.home() / ".hermes" / "state" / "board-comms"


def suppression_log_path() -> Path:
    configured = os.environ.get("HERMES_BOARD_COMMS_SUPPRESSION_LOG")
    if configured:
        return Path(configured)
    return _state_root() / "suppression.jsonl"


def ack_state_path() -> Path:
    configured = os.environ.get("HERMES_BOARD_COMMS_ACK_STATE")
    if configured:
        return Path(configured)
    return _state_root() / "ack-state.json"


def _plain(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)
    text = re.sub(r"[`*_~>|#\[\]()]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_ack_only(text: str) -> bool:
    plain = _plain(text)
    if not plain:
        return False
    if len(plain) > 180:
        return False
    if _ACK_RE.match(plain):
        return True
    # Common noisy Board acknowledgements with a little extra phrasing.
    lower = plain.lower()
    if lower in {
        "no further action needed from prime hermes",
        "no further action from prime hermes",
        "prime hermes standing by",
        "prime hermes is standing by",
    }:
        return True
    return False


@dataclass(frozen=True)
class BoardDecision:
    allow: bool
    reason: str = "ALLOW"
    detail: str = ""


def _metadata(metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return metadata if isinstance(metadata, Mapping) else {}


def _board_context(metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    meta = _metadata(metadata)
    ctx = meta.get("board_context")
    return ctx if isinstance(ctx, Mapping) else {}


def _is_board_chat(chat_id: str, metadata: Optional[Mapping[str, Any]]) -> bool:
    meta = _metadata(metadata)
    if meta.get("board_comms_force") is True:
        return True
    if meta.get("board_comms_disable") is True:
        return False
    return str(chat_id) in board_chat_ids()


def _targeted_other_bot(inbound_text: str) -> bool:
    if not inbound_text or not _STOP_KILL_RE.search(inbound_text):
        return False
    mentions = {m.group(1).lower() for m in _BOT_MENTION_RE.finditer(inbound_text)}
    if not mentions:
        return False
    return not bool(mentions & own_bot_usernames())


def _read_ack_state() -> dict[str, Any]:
    path = ack_state_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _write_ack_state(data: Mapping[str, Any]) -> None:
    path = ack_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(dict(data), ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def _ack_key(chat_id: str, metadata: Optional[Mapping[str, Any]]) -> str:
    meta = _metadata(metadata)
    thread = meta.get("thread_id") or _board_context(metadata).get("thread_id") or ""
    return f"{chat_id}:{thread}"


def _recent_ack_exists(chat_id: str, metadata: Optional[Mapping[str, Any]], now: float) -> bool:
    try:
        window = float(os.environ.get("HERMES_BOARD_ACK_WINDOW_SECONDS", DEFAULT_ACK_WINDOW_SECONDS))
    except (TypeError, ValueError):
        window = DEFAULT_ACK_WINDOW_SECONDS
    state = _read_ack_state()
    last = state.get(_ack_key(chat_id, metadata))
    try:
        return bool(last is not None and now - float(last) <= window)
    except (TypeError, ValueError):
        return False


def _record_ack(chat_id: str, metadata: Optional[Mapping[str, Any]], now: float) -> None:
    state = _read_ack_state()
    state[_ack_key(chat_id, metadata)] = now
    _write_ack_state(state)


def log_suppression(
    *,
    chat_id: str,
    content: str,
    reason: str,
    detail: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    if not _env_bool("HERMES_BOARD_COMMS_LOG_SUPPRESSIONS", True):
        return
    ctx = dict(_board_context(metadata))
    event = {
        "ts": int(time.time()),
        "agent": agent_name(),
        "chat_id": str(chat_id),
        "thread_id": str(_metadata(metadata).get("thread_id") or ctx.get("thread_id") or ""),
        "reason": reason,
        "detail": detail,
        "inbound_text": str(ctx.get("inbound_text") or "")[:1000],
        "inbound_user_id": str(ctx.get("user_id") or ""),
        "inbound_user_name": str(ctx.get("user_name") or ""),
        "inbound_is_bot": bool(ctx.get("is_bot")),
        "proposed_reply": str(content or "")[:1200],
    }
    path = suppression_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass


def evaluate_send(
    *,
    chat_id: str,
    content: str,
    metadata: Optional[Mapping[str, Any]] = None,
    now: Optional[float] = None,
) -> BoardDecision:
    """Classify a proposed outbound Board message.

    Return ALLOW for non-Board chats and for critical/status messages.  Return
    SUPPRESS_* only for Board chats where the proposed reply is low-signal or
    violates targeted-command discipline.
    """
    if not _env_bool("HERMES_BOARD_COMMS_ENABLED", True):
        return BoardDecision(True)
    if not _is_board_chat(str(chat_id), metadata):
        return BoardDecision(True)

    ctx = _board_context(metadata)
    inbound_text = str(ctx.get("inbound_text") or "")
    inbound_is_bot = bool(ctx.get("is_bot"))
    proposed = content or ""
    current = time.time() if now is None else now
    ack_only = is_ack_only(proposed)

    if _targeted_other_bot(inbound_text) and not _CRITICAL_RE.search(proposed):
        return BoardDecision(False, "SUPPRESS_TARGETED_OTHER_BOT", "Anthony targeted a different bot/lane with stop/kill/hold/quiet")

    if ack_only:
        if inbound_is_bot:
            return BoardDecision(False, "SUPPRESS_BOT_TO_BOT_ACK", "Bot-originated message produced acknowledgement-only reply")
        return BoardDecision(False, "SUPPRESS_ACK_ONLY", "Acknowledgement-only Board reply is below the public signal bar")

    return BoardDecision(True)
