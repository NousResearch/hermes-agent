"""Bot-to-bot chains in Bot Mode: a runtime-owned hop count, identical-ask dedupe and the reply rule.

A human message starts a chain. Inside a bot-authored turn every ``message_agent`` call is one more
hop (a fan-out is one call, so its targets share the hop); a person's own turn sends at hop 1. The
hop number rides in the delivery's author envelope (``hops``), which only the runtime writes, so the
model cannot reset it. Past ``gateway.bot_loop_guard.max_hops`` (config.yaml, the same cap the
messaging gateway applies) a send is refused and nothing is delivered. The same message to the same
recipient twice in one chain is refused as a duplicate. A message to the bot whose message started
this turn is a reply: unless ``bot_replies_expect_reply`` is true it goes notify-only (#144789's
``[no reply needed]`` marker and ``wake="none"``), and a turn started by such a message cannot message
that bot back. Each side also counts the bot turns it runs, so a lane that carries no envelope (the
Desktop relay) is still capped. ``gateway.bot_loop_guard.enabled: false`` turns all of it off (#137027).

Recipients are compared by the identity the receiving turn's author id carries (``bot:<profile>``,
``bot:<connection>/<profile>``), resolved by the caller with the same resolver delivery uses.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# Keyed by session id, not held on the agent: a gateway may rebuild the agent between turns, and a
# lost count would hand the chain a fresh budget. A session id change (compression rotation, turn-start
# recovery, lease reload) moves the entry along (``carry_chain`` / ``_bound_key``).
_CHAINS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _settings():
    from gateway.bot_loop_guard import load_settings
    return load_settings()


def _hops(author: Optional[dict]) -> int:
    value = (author or {}).get("hops")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _new_state(now: float) -> dict:
    return {"hops": 0, "sent": 0, "asks": set(), "no_reply_from": None, "updated": now}


def _bound_key(agent: Any) -> str:
    """The agent's chain key, carrying the chain over when its session id moved since the last turn
    or send. Caller holds ``_LOCK``."""
    key = str(getattr(agent, "session_id", "") or id(agent))
    previous = getattr(agent, "_bot_chain_session", None)
    if previous and previous != key and previous in _CHAINS and key not in _CHAINS:
        _CHAINS[key] = _CHAINS.pop(previous)
    try:
        agent._bot_chain_session = key
    except AttributeError:
        pass
    return key


def carry_chain(old_session_id: str, new_session_id: str) -> None:
    """Move a chain to the session id a compression rotation hands the conversation."""
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return
    with _LOCK:
        state = _CHAINS.pop(str(old_session_id), None)
        if state is not None:
            _CHAINS.setdefault(str(new_session_id), state)


def on_turn_start(agent: Any, turn_author: Optional[dict], display_kind: Optional[str],
                  user_message: Any = None) -> None:
    """Set the hop this session's turn runs at. A bot-authored turn runs at the hop its envelope names
    (or one past this side's own count, whichever is higher); a completion wake after this bot sent a
    message carries the answer, which is one more hop; a person's message ends the chain. A fork
    (background review, side question) shares the session id but is not the conversation: no change."""
    if getattr(agent, "_turn_origin", None):
        return
    settings = _settings()
    if not settings.enabled:
        return
    from gateway.response_filters import is_machinery_display_kind
    from tools.bot_mode_probe import NO_REPLY_MARKER
    now = time.monotonic()
    with _LOCK:
        key = _bound_key(agent)
        for stale in [k for k, v in _CHAINS.items() if now - v["updated"] > settings.cooldown_seconds]:
            del _CHAINS[stale]
        state = _CHAINS.get(key)
        if isinstance(turn_author, dict) and turn_author.get("is_bot"):
            state = _CHAINS.setdefault(key, _new_state(now))
            state["hops"] = max(state["hops"] + 1, _hops(turn_author))
            state["sent"] = max(state["sent"], state["hops"])
            # The sender's notify-only marker, read from the text like gateway/session_bot_outcome.py
            # does, so it survives every transport (the Desktop relay rebuilds the author).
            state["no_reply_from"] = turn_author.get("id") if NO_REPLY_MARKER in str(user_message or "") else None
        elif state is not None and is_machinery_display_kind(display_kind):
            if state["sent"] > state["hops"]:
                state["hops"] = state["sent"] = state["sent"] + 1
            state["no_reply_from"] = None
        elif state is not None:
            del _CHAINS[key]  # a person spoke: a new chain starts with their message
            return
        if state is not None:
            state["updated"] = now


@dataclass
class Admission:
    """One ``message_agent`` call's verdict. ``refusal`` refuses the whole call; ``refused`` maps a
    recipient key to its own refusal; ``notify_only`` holds the keys whose message is a reply that
    needs no answer. ``release`` hands back what a recipient that was definitely not sent reserved."""

    refusal: Optional[str] = None
    hop: int = 0
    refused: dict = field(default_factory=dict)
    notify_only: set = field(default_factory=set)
    key: str = ""
    asks: dict = field(default_factory=dict)
    prior_sent: int = 0

    def envelope(self) -> dict:
        # Hop 1 is what a receiver assumes for an envelope without the field, so it is left out.
        return {"hops": self.hop} if self.hop > 1 else {}

    def release(self, recipient: str) -> None:
        """Nothing reached ``recipient`` (a definite no-send): forget its ask, and the call's hop once
        no recipient of the call went out, so a retry is neither a duplicate nor another hop."""
        with _LOCK:
            ask = self.asks.pop(recipient, None)
            state = _CHAINS.get(self.key)
            if ask is None or state is None:
                return
            state["asks"].discard(ask)
            if not self.asks and state["sent"] == self.hop:
                state["sent"] = self.prior_sent


def admit_send(agent: Any, recipients: list[str], body: str) -> Admission:
    """Check one ``message_agent`` call to ``recipients`` (resolved identities, see the module doc).
    Admitted recipients reserve their ask and the call's hop until ``Admission.release``."""
    settings = _settings()
    if not settings.enabled:
        return Admission()
    author = getattr(agent, "_turn_author", None) or {}
    author_id = str(author.get("id") or "") if author.get("is_bot") else ""
    digest = hashlib.sha256(" ".join(str(body).split()).encode()).hexdigest()
    with _LOCK:
        key = _bound_key(agent)
        state = _CHAINS.setdefault(key, _new_state(time.monotonic()))
        bot_turn = state["hops"] > 0
        hop = state["sent"] + 1 if bot_turn else 1
        if hop > settings.max_hops:
            return Admission(refusal=(
                f"This bot-to-bot exchange already reached its limit of {settings.max_hops} messages "
                "(gateway.bot_loop_guard.max_hops), so nothing was sent. Do not retry; finish your turn "
                "and tell your user where it stands."))
        admission = Admission(hop=hop, key=key, prior_sent=state["sent"])
        for recipient in recipients:
            ask = (recipient, digest)
            replying = bool(author_id) and recipient == author_id
            if replying and state["no_reply_from"] == author_id:
                admission.refused[recipient] = (
                    "That bot sent you a reply that needs no answer, so this exchange is over. Do not "
                    "message them again in this turn; tell your user what you learned instead.")
            elif bot_turn and ask in state["asks"]:
                admission.refused[recipient] = (
                    "You already sent this exact message to that bot in this exchange, so it was not "
                    "sent again. Wait for their answer or say something new.")
            else:
                state["asks"].add(ask)
                admission.asks[recipient] = ask
                if replying and not settings.bot_replies_expect_reply:
                    admission.notify_only.add(recipient)
        if admission.asks:
            state["sent"] = max(state["sent"], hop)
        state["updated"] = time.monotonic()
    return admission
