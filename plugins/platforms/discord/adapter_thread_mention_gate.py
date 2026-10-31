"""Opt-in admission gate: drop mentions-of-others inside a thread the bot joined (#142739).

A thread the bot already participates in is normally exempt from the no-mention drop — it is its
own conversation context, not someone else's (#116568 / PR #116581). Operators running multi-bot
or human+bot threads asked for the channel-level "mentions someone else, not me" policy in those
threads too, without turning the thread into a strict mention-only room: ``DISCORD_THREAD_REQUIRE_MENTION``
demotes everything that is not a typed mention, which also drops the plain no-mention follow-ups a
thread exists for.

So the gate is narrow and off by default: only a message that already carries mentions (and neither
mentions the bot nor another bot — those are handled above it) is dropped, and only in a bot thread
admitted by the exemption. Every other thread message keeps the thread's normal flow.

The predicate lives beside the adapter rather than inside it: ``adapter.py`` is at its line cap, and
this is one decision plus the reader every other Discord gate already uses (YAML key → scoped env →
default), so a secondary profile never inherits another profile's bridged value.
"""
from __future__ import annotations

from typing import Any

ENV_KEY = "DISCORD_THREAD_IGNORE_NO_MENTION"
EXTRA_KEY = "thread_ignore_no_mention"


def thread_ignore_no_mention_enabled(adapter: Any) -> bool:
    """``thread_ignore_no_mention`` YAML (or ``DISCORD_THREAD_IGNORE_NO_MENTION``) — default off."""
    return adapter._extra_or_env_flag(EXTRA_KEY, ENV_KEY, "false", truthy=True)


def drops_unaddressed_thread_mention(adapter: Any, message: Any) -> bool:
    """True when this message is admitted ONLY by the bot-thread exemption and the gate closes it.

    Requires all three: the gate is on, the message addresses someone else (``mentions`` is
    non-empty — the caller has already ruled out the bot itself and other bots), and the channel is
    a thread this bot joined while ``thread_require_mention`` is off (``_in_bot_thread``; with
    strict multi-bot threads the caller's own channel path already drops it).

    Free-response channels keep their meaning: the caller's channel branch runs next and admits
    the message, so an operator who opted the channel into free response never sees this gate.
    """
    if not getattr(message, "mentions", None):
        return False
    if not adapter._in_bot_thread(message):
        return False
    return thread_ignore_no_mention_enabled(adapter)
