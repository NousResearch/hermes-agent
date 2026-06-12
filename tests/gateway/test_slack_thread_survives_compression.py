"""Reproduction: a Slack channel-thread reply must carry ``thread_ts`` when the
gateway delivers a queued follow-up via the non-streaming
``_status_thread_metadata`` path.

Prod incident (2026-06-11, channel C0AH3RY3DK6):
  - A dropped-thread-followup nudge was posted IN-thread (thread root
    1781067291.071199).
  - The agent's run was interrupted by an injected background-process
    completion event and hit mid-run context compression
    ("Session split detected … (compression)"); a queued follow-up then
    drained.
  - The agent's final reply (Slack ts 1781234725.805679, text
    "*Dropped-thread followup — status: not done*") and its
    ":brain: Memories used" footer (ts 1781234725.832809) were posted at the
    CHANNEL ROOT (thread_ts=None) instead of in-thread — confirmed via
    conversations.history (both have thread_ts=None).

Root cause — Slack/Telegram asymmetry in ``_run_agent`` (gateway/run.py):

  The progress-thread id is computed with a Slack-specific reply-anchor
  fallback::

      if source.platform == Platform.SLACK:
          _progress_thread_id = source.thread_id or event_message_id   # truthy
      ...

  ``_progress_metadata`` (run.py ~13381) correctly honours that fallback:
  when ``_progress_thread_id != source.thread_id`` it uses
  ``{"thread_id": _progress_thread_id}``.

  But ``_status_thread_metadata`` (run.py ~13772) does NOT::

      _status_thread_metadata = (
          self._thread_metadata_for_source(source, event_message_id)
          if _progress_thread_id else None
      )

  ``_thread_metadata_for_source`` returns ``None`` whenever
  ``source.thread_id is None`` — even though ``_progress_thread_id`` is truthy
  via the ``event_message_id`` fallback.  The queued-follow-up delivery
  (run.py ~15296) then calls ``adapter.send(chat_id, text,
  metadata=_status_thread_metadata)`` with **no** ``reply_to`` argument, so
  ``SlackAdapter._resolve_thread_ts(None, None)`` yields ``None`` and
  ``chat_postMessage`` is sent without ``thread_ts`` → the reply lands at the
  channel root.

This test drives the real ``GatewayRunner._run_agent`` (streaming OFF, matching
prod) with a Slack source whose ``thread_id is None`` but a populated
``event_message_id`` reply anchor, queues a pending follow-up so the
queued-delivery branch fires, and asserts the captured Slack
``chat_postMessage`` carried the in-thread ``thread_ts``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.slack import SlackAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


THREAD_TS = "1781067291.071199"
CHANNEL = "C0AH3RY3DK6"


def _make_slack_adapter():
    """Build a SlackAdapter whose chat_postMessage is captured (no real I/O)."""
    cfg = PlatformConfig(enabled=True, extra={"reply_in_thread": True})
    adapter = SlackAdapter(cfg)

    captured = []

    async def _chat_postMessage(**kwargs):
        captured.append(kwargs)
        return {"ts": "9999.0001", "ok": True}

    fake_client = SimpleNamespace(chat_postMessage=AsyncMock(side_effect=_chat_postMessage))
    # _app truthiness gates send(); _get_client() falls back to _app.client.
    adapter._app = SimpleNamespace(client=fake_client)
    return adapter, captured


class _CompletingAgent:
    """Stub AIAgent that completes normally (not interrupted) with a final
    reply.  The gateway delivers this first reply via the queued-follow-up
    branch when a pending message is waiting."""

    def __init__(self, *args, **kwargs):
        self.session_id = "pre_split"
        self.stream_delta_callback = None
        self.clarify_callback = None
        self.tools = []
        self.interrupt = lambda *_a, **_k: None

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        # Simulate the mid-run context compression that rotated the session id
        # in prod (the leaked reply was delivered post-split).
        self.session_id = "post_split_47aa2e"
        return {
            "final_response": "Dropped-thread followup — status: not done.",
            "messages": [],
            "api_calls": 17,
            "tools": [],
            "completed": True,
            "interrupted": False,
        }


@pytest.fixture
def runner(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    r = GatewayRunner(GatewayConfig())
    # Match prod: streaming disabled → reply goes through the non-streaming
    # queued-delivery path that uses _status_thread_metadata.
    r.config.streaming.enabled = False
    return r


@pytest.mark.asyncio
async def test_slack_queued_reply_stays_in_thread_with_anchor_only(
    monkeypatch, tmp_path, runner
):
    import run_agent as run_agent_mod

    monkeypatch.setattr(run_agent_mod, "AIAgent", _CompletingAgent)
    # Bypass real provider/auth resolution so the run reaches agent construction
    # and the queued-delivery path under test.
    monkeypatch.setattr(
        GatewayRunner,
        "_resolve_session_agent_runtime",
        lambda self, **kw: ("test-model", {}),
    )
    monkeypatch.setattr(
        GatewayRunner,
        "_resolve_turn_agent_config",
        lambda self, msg, model, rk: {"model": "test-model", "runtime": {}},
    )

    adapter, captured = _make_slack_adapter()
    runner.adapters[Platform.SLACK] = adapter

    # Slack channel source where thread_id is None but a reply anchor
    # (event_message_id) IS present — the prod leak shape. _progress_thread_id
    # falls back to event_message_id (truthy), but _status_thread_metadata is
    # derived only from source.thread_id and so collapses to None.
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id=CHANNEL,
        chat_type="group",
        thread_id=None,
        user_id="U0A4G7LDJ4R",
        user_name="MCP Agent Mail",
    )
    session_key = f"agent:main:slack:group:{CHANNEL}:{THREAD_TS}"

    # Queue a pending follow-up so the queued-delivery branch (run.py ~15296)
    # fires adapter.send(..., metadata=_status_thread_metadata).  The pending
    # event mirrors the synthetic background-process completion injected in
    # prod (same lane/source).
    from gateway.platforms.base import MessageEvent, MessageType

    pending_event = MessageEvent(
        text="follow-up nudge",
        message_type=MessageType.TEXT,
        source=source,
        message_id=THREAD_TS,
    )
    adapter._pending_messages[session_key] = pending_event

    await runner._run_agent(
        message="[IMPORTANT: Background process completed]",
        context_prompt="",
        history=[],
        source=source,
        session_id="pre_split",
        session_key=session_key,
        event_message_id=THREAD_TS,
    )

    assert captured, "Expected at least one chat_postMessage call"
    leaked = [c for c in captured if not c.get("thread_ts")]
    assert not leaked, (
        f"LEAK: {len(leaked)} Slack message(s) posted at channel root "
        f"(thread_ts missing) — the queued follow-up dropped the thread anchor. "
        f"kwargs={leaked!r}"
    )
    for c in captured:
        assert c.get("thread_ts") == THREAD_TS, (
            f"Slack reply routed to wrong thread: thread_ts={c.get('thread_ts')!r} "
            f"expected {THREAD_TS!r}"
        )
