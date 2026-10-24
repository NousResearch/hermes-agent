"""Real hygiene turn admission, durable cooldown and notice sink in three modes."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hermes_state import SessionDB
from tests.gateway.test_session_hygiene import _make_cooldown_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
async def test_aborted_hygiene_retains_cooldown_across_restart_and_final_result(tmp_path, monkeypatch, caplog, setting):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    sid = "hygiene-policy"
    calls = []
    recovering = False
    class CompressorAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]
            self._session_db = kwargs.get("session_db")
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(), _last_compress_aborted=True,
                _last_summary_error="fixture auxiliary failure", _last_aux_model_failure_model=None)
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
        def _compress_context(self, messages, *args, **kwargs):
            calls.append(list(messages))
            if recovering:
                compacted = [{"role": "user", "content": "retained summary"}]
                self._session_db.archive_and_compact(self.session_id, compacted)
                self._session_db.clear_compression_failure_cooldown(self.session_id)
                self._last_compaction_in_place = True
                self.context_compressor._last_compress_aborted = False
                self.context_compressor._last_aux_model_failure_model = "fixture-aux"
                self.context_compressor._last_aux_model_failure_error = "fixture auxiliary failure"
                return compacted, None
            return messages, None
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(sid, "telegram")
        for iteration in range(2):
            runner, adapter, event = _make_cooldown_runner(monkeypatch, tmp_path, CompressorAgent, db, sid)
            cfg = {"compression": {"enabled": True, "hygiene_failure_cooldown_seconds": 300}}
            if setting is not None:
                cfg["display"] = {"suppress_warning_notifications": setting}
            (tmp_path / "config.yaml").write_text(json.dumps(cfg))
            assert await runner._handle_message(event) == "ok"
            assert runner._run_agent.await_count == 1
            warnings = [sent for sent in adapter.sent if "Shortening the conversation history failed" in sent["content"]]
            assert len(warnings) == (1 if iteration == 0 and setting is not True else 0)
            assert len(calls) == 1  # fresh runner still obeys persisted cooldown
            cooldown = db.get_compression_failure_cooldown(sid)
            assert cooldown and cooldown["remaining_seconds"] > 0
            runner.session_store.rewrite_transcript.assert_not_called()
        # Let the durable cooldown expire, then exercise actual gateway adoption
        # of a committed fallback summary (provider compression is the fake edge).
        recovering = True
        db.clear_compression_failure_cooldown(sid)
        runner, adapter, event = _make_cooldown_runner(monkeypatch, tmp_path, CompressorAgent, db, sid)
        (tmp_path / "config.yaml").write_text(json.dumps(cfg))
        assert await runner._handle_message(event) == "ok"
        assert len(calls) == 2
        assert db.get_compression_failure_cooldown(sid) is None
        assert all(state.persistent.hygiene_failure_streak == 0 for state in runner._sessions.values())
        warnings = [sent for sent in adapter.sent if "Configured compression model" in sent["content"]]
        assert len(warnings) == (0 if setting is True else 1)
        assert db.get_messages(sid)[-1]["content"] == "retained summary"
        runner.session_store.rewrite_transcript.assert_not_called()
    finally:
        db.close()
