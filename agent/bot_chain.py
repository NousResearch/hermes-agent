"""Bot-to-bot chains in Bot Mode: a runtime-owned hop count, identical-ask dedupe and the reply rule.

A human message starts a chain. Each ``message_agent`` send inside it is one hop; the hop number rides
in the delivery's author envelope (``hops``), which only the runtime writes, so the model cannot reset
it. Past ``gateway.bot_loop_guard.max_hops`` (config.yaml, the same cap the messaging gateway applies)
a send is refused and nothing is delivered. The same message to the same teammate twice in one chain
is refused as a duplicate. A message to the bot whose message started this turn is a reply: it goes
notify-only (``expects_reply`` false in the envelope, no wake for its answer), and a turn started by
such a reply cannot message that bot back. Each side also counts the bot turns it runs, so a lane that
carries no envelope (Desktop relay) is still capped (#137027).
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Optional

# Keyed by session id, not held on the agent: a gateway may rebuild the agent between turns, and a
# lost count would hand the chain a fresh budget. Expired chains are pruned on every write.
_CHAINS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _settings():
    from gateway.bot_loop_guard import load_settings
    return load_settings()


def _hops(author: Optional[dict]) -> int:
    value = (author or {}).get("hops")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _profile_part(identity: str) -> str:
    """``bot:<profile>`` / ``bot:<origin>/<profile>`` / a target ``peer/agent`` or ``agent@conn`` -> agent."""
    text = str(identity or "").strip().lower().removeprefix("bot:").lstrip("@")
    return text.rsplit("/", 1)[-1].split("@", 1)[0]


def on_turn_start(agent: Any, turn_author: Optional[dict], display_kind: Optional[str]) -> None:
    """Set the hop this session's turn runs at. A bot-authored turn runs at the hop its envelope names
    (or one past this side's own count, whichever is higher); a completion wake after this bot sent a
    message carries the answer, which is one more hop; a person's message ends the chain."""
    from gateway.response_filters import is_machinery_display_kind
    key, ttl = _key(agent), _settings().cooldown_seconds
    now = time.monotonic()
    with _LOCK:
        for stale in [k for k, v in _CHAINS.items() if now - v["updated"] > ttl]:
            del _CHAINS[stale]
        state = _CHAINS.get(key)
        if isinstance(turn_author, dict) and turn_author.get("is_bot"):
            state = _CHAINS.setdefault(key, _new_state(now))
            state["hops"] = max(state["hops"] + 1, _hops(turn_author))
        elif state is not None and is_machinery_display_kind(display_kind):
            if state["sent"] > state["hops"]:
                state["hops"] = state["sent"] + 1
        elif state is not None:
            del _CHAINS[key]  # a person spoke: a new chain starts with their message
            return
        if state is not None:
            state["updated"] = now


def _new_state(now: float) -> dict:
    return {"hops": 0, "sent": 0, "asks": set(), "updated": now}


def _key(agent: Any) -> str:
    return str(getattr(agent, "session_id", "") or id(agent))


def admit_send(agent: Any, target: str, body: str) -> tuple[Optional[str], dict]:
    """Check one ``message_agent`` send. Returns ``(refusal, envelope)``: a refusal sentence for the
    model (nothing is delivered), else the fields the delivery's author envelope carries."""
    author = getattr(agent, "_turn_author", None) or {}
    replying = bool(author.get("is_bot")) and _profile_part(author.get("id") or author.get("name") or "") == \
        _profile_part(target)
    if replying and author.get("expects_reply") is False:
        return (f"@{_profile_part(target)} sent you a reply that needs no answer, so this exchange is over. "
                "Do not message them again in this turn; tell your user what you learned instead."), {}
    cap = _settings().max_hops
    ask = (_profile_part(target), hashlib.sha256(" ".join(str(body).split()).encode()).hexdigest())
    with _LOCK:
        state = _CHAINS.setdefault(_key(agent), _new_state(time.monotonic()))
        hops = state["hops"] + 1
        if hops > cap:
            return (f"This bot-to-bot exchange already reached its limit of {cap} messages "
                    "(gateway.bot_loop_guard.max_hops), so nothing was sent. Do not retry; finish your turn "
                    "and tell your user where it stands."), {}
        # Inside a bot exchange the same ask twice is one hop, not a new message (a person's turn may repeat).
        if state["hops"] and ask in state["asks"]:
            return (f"You already sent this exact message to @{ask[0]} in this exchange, so it was not sent "
                    "again. Wait for their answer or say something new."), {}
        state["asks"].add(ask)
        state.update(sent=max(state["sent"], hops), updated=time.monotonic())
    # Hop 1 is what a receiver assumes for an envelope without the field, so it is left out.
    return None, {**({"hops": hops} if hops > 1 else {}), **({"expects_reply": False} if replying else {})}
