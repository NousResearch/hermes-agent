"""Cross-session rate limit guard for aggregator-style custom providers.

Gateways like a self-hosted multi-account router (``provider: custom``)
multiplex several unrelated upstream accounts/models behind one endpoint and
report which upstream actually served (or rejected) a request inside the
error body, e.g. ``[antigravity/gemini-pro-agent] [429]: ...``. Hermes has no
way to know in advance which upstream a given call will land on — that
routing decision happens entirely inside the gateway — so unlike
:mod:`agent.nous_rate_guard` (one provider, one bucket) this guard is keyed
by whatever upstream tag shows up in the error text.

The problem this solves: Hermes can run many concurrent sessions against the
same gateway (WhatsApp DMs, the CLI/TUI, subagents, background review). When
one session's request gets rate-limited by a specific upstream, its own
retry already honors the reset window via ``Retry-After`` — but that
knowledge was private to that one session. A different concurrent session
whose own request lands on the *same* upstream moments later has no idea a
cooldown is already running, computes its own (possibly shorter) backoff
from scratch, and retries straight back into the same wall.

This module lets any session that observes a rate limit for upstream tag
``X`` publish the cooldown to a shared file, and lets any other session that
independently gets rate-limited on the same tag ``X`` merge its own wait with
the already-known one instead of retrying blind. It only ever *extends* a
wait computed from that session's own response — it never shortens it and
never blocks a request outright, so a stale or wrong tag costs at most one
slightly-longer sleep, not a dropped request.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from typing import Optional

from utils import atomic_replace

logger = logging.getLogger(__name__)

_STATE_SUBDIR = "rate_limits"
_KEY_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")

# Matches the "[<tag>/<model>] [429]" shape a multi-account router embeds in
# its proxied error body to identify which upstream rejected the request,
# e.g. "[antigravity/gemini-pro-agent] [429]: {...}". The tag is the router's
# own provider/account-pool name, not necessarily a Hermes-recognized
# provider id — it is only ever used as an opaque cache key here.
_UPSTREAM_TAG_RE = re.compile(r"\[([a-zA-Z][\w.-]*)/[^\]/]+\]\s*\[429\]")

# Reset windows shorter than this are treated as instant secondary jitter,
# not worth persisting a cross-session cooldown for.
_MIN_COOLDOWN_SECONDS = 1.0
_MAX_COOLDOWN_SECONDS = 600.0


def extract_upstream_tag(error_text: str) -> Optional[str]:
    """Pull the upstream tag out of a router-proxied 429 error string.

    Returns ``None`` when the text doesn't match the expected
    ``[tag/model] [429]`` shape (e.g. providers that don't proxy other
    providers, or an error format this router doesn't use).
    """
    if not error_text:
        return None
    m = _UPSTREAM_TAG_RE.search(error_text)
    return m.group(1) if m else None


def _state_path(key: str) -> str:
    safe_key = _KEY_SAFE_RE.sub("_", key).strip("_") or "unknown"
    try:
        from hermes_constants import get_hermes_home
        base = get_hermes_home()
    except ImportError:
        base = os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(base, _STATE_SUBDIR, f"downstream_{safe_key}.json")


def record_downstream_rate_limit(key: str, *, seconds: float) -> None:
    """Publish that upstream ``key`` is rate-limited for ``seconds`` more.

    A no-op for non-positive or absurdly large values — those are almost
    certainly a parsing mistake, not a real reset window.
    """
    if seconds < _MIN_COOLDOWN_SECONDS:
        return
    seconds = min(seconds, _MAX_COOLDOWN_SECONDS)
    now = time.time()
    reset_at = now + seconds

    path = _state_path(key)
    try:
        state_dir = os.path.dirname(path)
        os.makedirs(state_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"reset_at": reset_at, "recorded_at": now}, f)
            atomic_replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.info(
            "Downstream rate limit recorded for %r: resets in %.0fs", key, seconds,
        )
    except Exception as exc:
        logger.debug("Failed to write downstream rate limit state for %r: %s", key, exc)


def downstream_rate_limit_remaining(key: str) -> Optional[float]:
    """Seconds remaining in a previously recorded cooldown for ``key``.

    Returns ``None`` when there is no active cooldown (never recorded,
    already expired, or the state file is unreadable/corrupt — fails open
    rather than blocking a request over a guard-file problem).
    """
    path = _state_path(key)
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        remaining = float(state.get("reset_at", 0)) - time.time()
        if remaining > 0:
            return remaining
        try:
            os.unlink(path)
        except OSError:
            pass
        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def clear_downstream_rate_limit(key: str) -> None:
    try:
        os.unlink(_state_path(key))
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Failed to clear downstream rate limit state for %r: %s", key, exc)
