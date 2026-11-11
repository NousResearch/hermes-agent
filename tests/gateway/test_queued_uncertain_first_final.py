"""A queued first answer whose streamed send timed out is ledgered, not lost and not resent.

The stream consumer counts a final send that timed out after the request left as shown (sending it
again could put a second copy in the chat) and flags it ``delivery_uncertain``; the completion path
then hands it to the delivery ledger, whose redelivery resends it with the may-be-a-duplicate
marker. With a follow-up queued behind the turn, the legacy recursive lane delivers the first
answer and returns the FOLLOW-UP's result, so the completion path never sees the first turn's
flag. The queued lane must ledger it itself, or an answer that never landed is gone for good.
"""

import os

import pytest

from gateway import delivery_ledger as dl
from gateway.platforms.base import SendResult
from persistence_assembly import profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _run_with_agent

FIRST = "The complete first answer."
SECOND = "The follow-up answer."


@pytest.fixture(autouse=True)
def _ledger_db(tmp_path, monkeypatch):
    home = tmp_path / "ledger"
    home.mkdir()
    monkeypatch.setattr(dl, "_scope", lambda: profile_scope(home=home))
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)


def _rows():
    with raw_holder(storage_handle(dl._scope().profile_dir))._read_ctx() as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'delivery_obligations'").fetchone() is None:
            return []  # the ledger creates its table on its first use of the store
        return [dict(zip(("state", "content", "last_error"), r)) for r in conn.execute(
            "SELECT state, content, last_error FROM delivery_obligations").fetchall()]


class _LostAckAdapter(ProgressCaptureAdapter):
    """The first send times out after the request left and never reaches the chat."""

    def __init__(self, platform=None):
        super().__init__(**({"platform": platform} if platform else {}))
        self.attempts = 0

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.attempts += 1
        if self.attempts == 1:
            return SendResult(success=False, error="Timed out", retryable=False)
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


class _TwoTurnStreamingAgent:
    turns = 0

    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).turns += 1
        text = FIRST if type(self).turns == 1 else SECOND
        if self.stream_delta_callback:
            self.stream_delta_callback(text)
        return {"final_response": text, "response_previewed": True, "messages": [], "api_calls": 1}


@pytest.mark.asyncio
async def test_timed_out_queued_first_answer_is_ledgered_not_resent(monkeypatch, tmp_path):
    _TwoTurnStreamingAgent.turns = 0
    adapter, result = await _run_with_agent(
        monkeypatch, tmp_path, _TwoTurnStreamingAgent, session_id="sess-queued-uncertain",
        pending_text="and one more thing",  # the legacy recursive pending-event lane
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "streaming": {"enabled": True, "edit_interval": 0.01, "buffer_threshold": 1, "cursor": ""},
        },
        adapter_cls=_LostAckAdapter,
    )

    assert _TwoTurnStreamingAgent.turns == 2
    assert result["final_response"] == SECOND
    # No immediate second copy: the timed-out send may have landed.
    assert FIRST not in [c["content"] for c in adapter.sent]
    # ...but the answer is owed: a failed ledger row the timed redelivery sends again, marked.
    owed = [r for r in _rows() if r["content"] == FIRST]
    assert owed and owed[0]["state"] == "failed", (
        f"uncertain first answer left no ledger obligation: rows={_rows()} sent={adapter.sent}")
