"""
Regression tests for the kind=status adapter-policy patch.

Background — live bug (May 2026): status / lifecycle / interrupt-ack
messages from the gateway (e.g. ``⚡ Interrupting current task
(iteration 2/90, …)``, ``⚠️ No response from provider for Ns…``) were
leaking into iMessage group chats as if they were normal agent replies,
because the delivery path made no distinction between a user-facing
reply and an internal status bubble.

The fix lives centrally in :mod:`gateway.platforms.base`:

  1. :class:`SendResult` gains ``suppressed: bool``.
  2. :meth:`BasePlatformAdapter._send_with_retry` gains a keyword-only
     ``kind`` argument (default ``"reply"``).
  3. :meth:`BasePlatformAdapter._should_suppress_status` policy hook
     (default ``False``) lets adapters opt out of delivering
     ``kind="status"`` on chat surfaces where it would be intrusive
     (e.g. iMessage groups, where every message is a permanent,
     un-editable bubble visible to everyone).
  4. Status call sites in ``gateway/run.py`` (interrupt/queue/steer-ack,
     drain-ack, ``_status_callback_sync`` bridge, shutdown notifier)
     pass ``kind="status"``.

These tests pin down each of those guarantees so the fix can't quietly
regress.  They are deliberately self-contained — they exercise the
central contract via a stub adapter rather than any specific platform,
so they work in CI without needing user-installed platform plugins.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# tests/gateway/<this file>  →  repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]

from gateway.platforms.base import BasePlatformAdapter, SendResult  # noqa: E402


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------

class _StubAdapter(BasePlatformAdapter):
    """Minimal adapter that records send() calls.  Used to verify the
    central ``_send_with_retry`` contract independent of any real
    platform.  ``BasePlatformAdapter.__init__`` is bypassed because we
    don't need its config / Platform plumbing for these tests."""

    def __init__(self, *, suppress_status: bool = False):
        self._suppress_status = suppress_status
        self.sent: list[dict] = []
        self._name = "stub"
        # Mirror the few attributes _send_with_retry / its helpers may
        # touch through ``self``.
        self.config = None  # not used by the code paths we exercise

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    async def connect(self) -> bool:  # pragma: no cover
        return True

    async def disconnect(self) -> None:  # pragma: no cover
        return None

    async def send(  # type: ignore[override]
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id="stub-1")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:  # type: ignore[override]
        return {"name": str(chat_id), "type": "dm", "chat_id": str(chat_id)}

    async def _should_suppress_status(  # type: ignore[override]
        self,
        chat_id: str,
        *,
        kind: str = "status",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self._suppress_status
def test_send_with_retry_kind_defaults_to_reply_and_delivers():
    """Default ``kind="reply"`` must always deliver — no policy intercept,
    no behavior change for the thousands of existing call sites."""
    a = _StubAdapter(suppress_status=True)  # would suppress *if* asked
    result = asyncio.run(a._send_with_retry(chat_id="c1", content="hi"))
    assert result.success
    assert result.suppressed is False
    assert len(a.sent) == 1
    assert a.sent[0]["content"] == "hi"


def test_send_with_retry_kind_status_suppressed_in_group():
    """When ``kind="status"`` and the adapter policy returns True, the
    message is NOT delivered and the result is flagged ``suppressed``."""
    a = _StubAdapter(suppress_status=True)
    result = asyncio.run(
        a._send_with_retry(chat_id="c1", content="⚡ …", kind="status")
    )
    assert result.success is True
    assert result.suppressed is True
    assert result.message_id is None
    assert a.sent == [], "suppressed status must not reach adapter.send()"


def test_send_with_retry_kind_status_delivered_in_dm():
    """When the policy returns False (e.g. DM), status goes through
    normally — useful progress signal there."""
    a = _StubAdapter(suppress_status=False)
    result = asyncio.run(
        a._send_with_retry(chat_id="c1", content="⚡ …", kind="status")
    )
    assert result.success is True
    assert result.suppressed is False
    assert len(a.sent) == 1


def test_send_with_retry_policy_hook_exception_falls_through_to_send():
    """If the policy hook raises, we MUST NOT drop the message silently
    — log the error and deliver as if the hook had returned False.  The
    fix is meant to silence noise, not silently swallow real
    notifications."""
    a = _StubAdapter()

    async def _boom(chat_id, *, kind="status", metadata=None):
        raise RuntimeError("simulated policy bug")

    a._should_suppress_status = _boom  # type: ignore[assignment]
    result = asyncio.run(
        a._send_with_retry(chat_id="c1", content="⚡ …", kind="status")
    )
    assert result.success is True
    assert result.suppressed is False
    assert len(a.sent) == 1


def test_suppressed_is_distinguishable_from_delivered():
    """Audit/logging callers must be able to tell delivered from
    suppressed without parsing message content."""
    delivered = SendResult(success=True, message_id="m1")
    suppressed = SendResult(success=True, suppressed=True)
    assert delivered.success and not delivered.suppressed
    assert suppressed.success and suppressed.suppressed
    assert delivered.message_id and not suppressed.message_id


def test_kind_is_keyword_only_so_legacy_positional_callers_still_work():
    """``kind`` was added as a kwarg with a default — legacy callers
    that pass ``max_retries`` / ``base_delay`` positionally must not
    accidentally bind their value to ``kind``.

    The signature uses ``max_retries: int = 2, base_delay: float = 2.0,
    kind: str = "reply"`` so this verifies the order in code.
    """
    import inspect
    sig = inspect.signature(BasePlatformAdapter._send_with_retry)
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    # ``kind`` must appear after the retry-knobs to preserve the
    # historical positional contract for existing internal callers.
    assert "kind" in names
    assert names.index("kind") > names.index("max_retries")
    assert names.index("kind") > names.index("base_delay")
    assert sig.parameters["kind"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["kind"].default == "reply"


# ---------------------------------------------------------------------------
# 2. Anti-regression: gateway/run.py status sites must use kind="status"
# ---------------------------------------------------------------------------

def _read_run_py() -> str:
    return (REPO_ROOT / "gateway" / "run.py").read_text(encoding="utf-8")


def test_every_adapter_send_with_retry_in_run_py_declares_kind_status():
    """The two ``await adapter._send_with_retry(...)`` call blocks in
    gateway/run.py are interrupt/queue/steer-ack and drain/busy-ack —
    both are pure status bubbles.  Each must declare ``kind="status"``
    so they route through the adapter policy and don't leak into
    groups."""
    src = _read_run_py()
    lines = src.splitlines()
    starts = [
        i for i, ln in enumerate(lines)
        if "await adapter._send_with_retry(" in ln
    ]
    assert starts, "expected at least one adapter._send_with_retry block in run.py"
    bad: list[str] = []
    for idx in starts:
        # Capture forward until the closing ``)`` whose indentation
        # matches the opening ``await adapter._send_with_retry(`` line
        # — that's the end of the call.
        open_indent = len(lines[idx]) - len(lines[idx].lstrip())
        block_lines = [lines[idx]]
        for j in range(idx + 1, min(idx + 60, len(lines))):
            block_lines.append(lines[j])
            stripped = lines[j].lstrip()
            if stripped.startswith(")"):
                close_indent = len(lines[j]) - len(stripped)
                if close_indent == open_indent:
                    break
        block = "\n".join(block_lines)
        if 'kind="status"' not in block:
            bad.append(block)
    assert not bad, (
        "every adapter._send_with_retry block in gateway/run.py must "
        "declare kind=\"status\" (these are interrupt/drain/queue acks); "
        "missing in:\n\n" + "\n---\n".join(bad)
    )


def test_status_callback_bridge_routes_through_send_with_retry_with_kind_status():
    """``_status_callback_sync`` is the bridge that posts agent-emitted
    lifecycle/warn events (provider stalls, compression warnings, …) to
    the user.  It used to call ``_status_adapter.send(...)`` directly,
    which bypassed every policy — that's the original leak.  It must
    now go through ``_send_with_retry`` with ``kind="status"``."""
    src = _read_run_py()

    # Locate the function and grab everything up to the next def at the
    # same indentation level.  The function is nested inside _run_agent
    # so its def starts at 8-space indent.
    m = re.search(
        r"^( {8})def _status_callback_sync\(.*?\)\s*->\s*None:\s*\n"
        r"(?P<body>(?:\1 .*\n|\s*\n)+)",
        src,
        flags=re.MULTILINE,
    )
    assert m, "couldn't locate _status_callback_sync in gateway/run.py"
    body = m.group("body")
    assert "_send_with_retry" in body, (
        "_status_callback_sync must route through _send_with_retry; "
        "a raw _status_adapter.send() bypasses the kind=\"status\" "
        "policy and is the original leak.\nBody was:\n" + body
    )
    assert 'kind="status"' in body, (
        "_status_callback_sync must pass kind=\"status\" so the adapter "
        "policy can suppress it in group chats.\nBody was:\n" + body
    )


def test_shutdown_notifier_uses_kind_status():
    """``_notify_active_sessions_of_shutdown`` posts
    ``⚠️ Gateway shutting down / restarting…`` to every active chat
    plus home channels.  Those are lifecycle messages too — same
    suppression rules apply."""
    src = _read_run_py()
    # Grab the whole function — find its def line, then capture until
    # the next def at the same indent (4 spaces — method on the class).
    m = re.search(
        r"^( {4})async def _notify_active_sessions_of_shutdown\(.*?\)\s*->\s*None:\s*\n"
        r"(?P<body>(?:\1 .*\n|\s*\n)+)",
        src,
        flags=re.MULTILINE,
    )
    assert m, "couldn't locate _notify_active_sessions_of_shutdown"
    body = m.group("body")
    # All adapter sends in this function should be status-tagged.
    # We don't insist on _send_with_retry specifically because adapters
    # may differ, but if .send( is used at all it would bypass the
    # policy — flag that as a regression.
    raw_sends = [
        ln for ln in body.splitlines()
        if "adapter.send(" in ln and not ln.lstrip().startswith("#")
    ]
    assert not raw_sends, (
        "_notify_active_sessions_of_shutdown must not use raw adapter.send() "
        "— route through _send_with_retry(..., kind=\"status\") so iMessage "
        "groups don't get a shutdown bubble.\nFound raw adapter.send() in:\n"
        + body
    )
    # And it must actually pass kind="status" somewhere.
    assert 'kind="status"' in body, (
        "_notify_active_sessions_of_shutdown must declare kind=\"status\" "
        "on its adapter sends.\nBody was:\n" + body
    )
