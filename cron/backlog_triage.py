"""Optional pre-classification of the cron Bot Chat backlog via a local triage model.

The scheduler already handles each queued item; this module lets an operator ask a
local classification service (Laya, by default on 127.0.0.1) whether a *queued*
item is worth processing at all, so "suppress" items can be settled without a turn.

Design contract — shadow-first and fail-open, so the normal flow is never at risk:

* ``HERMES_BACKLOG_TRIAGE`` unset / ``off`` / ``0`` / ``false`` / ``no`` (default):
  no network call, no behaviour change.
* ``=shadow``: classify and log the verdict, but never act on it.
* ``=on`` / ``1`` / ``true``: act on a ``suppress`` verdict; every other action
  (``watch`` / ``review`` / ``notify`` / ``page``) and every failure is a no-op.

Any error (service down, timeout, malformed reply, unknown verdict) returns
``"suppress" is not the outcome`` — i.e. the item proceeds exactly as before.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# LAYA_GATE_URL wins; otherwise probe the local service the reference client uses.
# 18765 is the loopback "gate" port; 8765 is the plain Laya serve port.
_URLS = [url for url in (
    os.environ.get("LAYA_GATE_URL"),
    "http://127.0.0.1:18765/v1/systemone",
    "http://127.0.0.1:8765/v1/systemone",
) if url]

_OFF = {"", "0", "false", "off", "no"}

_QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "What should happen with this queued cron backlog item?",
        "criteria": {
            "suppress": "routine noise or an already-known condition, safe to ignore",
            "watch": "interesting, keep an eye out, no action now",
            "review": "worth a human or agent looking at soon",
            "notify": "notify the owner now",
            "page": "urgent, cannot wait, wake someone",
        },
    },
}


def _mode() -> str:
    return os.environ.get("HERMES_BACKLOG_TRIAGE", "").strip().lower()


def _timeout() -> float:
    try:
        return float(os.environ.get("HERMES_BACKLOG_TRIAGE_TIMEOUT", "3"))
    except (TypeError, ValueError):
        return 3.0


def classify(text: str) -> str | None:
    """Return the triage verdict, or None when it cannot be obtained."""
    if not text:
        return None
    body = json.dumps({"state": {"log_excerpt": text}, "questions": _QUESTIONS}).encode()
    for url in _URLS:
        try:
            req = urllib.request.Request(
                url, data=body, method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=_timeout()) as resp:
                data = json.load(resp)
            return str(data["answers"]["action"]["choice"])
        except Exception:
            continue
    return None


def should_suppress(text: str, *, source: str = "backlog") -> bool:
    """True only when triage is enabled to act AND the verdict is ``suppress``.

    Never raises: every failure path returns False so the caller's flow is unchanged.
    """
    mode = _mode()
    if mode in _OFF:
        return False
    action = classify(text)
    if action is None:
        return False
    if mode == "shadow":
        logger.info("backlog triage shadow (%s): verdict=%s", source, action)
        return False
    logger.warning("backlog triage (%s): verdict=%s", source, action)
    return action == "suppress"
