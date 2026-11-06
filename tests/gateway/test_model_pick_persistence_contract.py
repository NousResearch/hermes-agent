"""Classic and receipt-backed model picks persist across queued accounting writes."""
from types import SimpleNamespace

import pytest

from tests.gateway.test_messaging_session_picks import _compressed_chat, _mutate


@pytest.mark.asyncio
@pytest.mark.parametrize("picker", [False, True])
async def test_classic_model_record_persists_for_typed_and_picker_calls(tmp_path, picker):
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_cli.model_switch import ModelSwitchResult

    runner, source, route, viewer, _root, _fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        db.update_session_model("seg-1", "old")
        ctx = _ModelSwitchContext(session_key=route, source=source,
                                  config_path=tmp_path / "config.yaml", persist_global=False,
                                  current_model="old")
        result = ModelSwitchResult(success=True, new_model="new", target_provider="custom",
                                   base_url="http://127.0.0.1:9/v1")
        await runner._record_model_switch(result, ctx, source=source, one_turn=False, picker=picker)
        assert db.get_session("seg-1")["model"] == "new"
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_receipt_model_pick_drains_old_usage_before_route_commit(tmp_path, monkeypatch):
    from hermes_cli import model_selection_guards, model_switch
    import hermes_state_mutation_chat_pick as picks

    monkeypatch.setattr(model_switch, "switch_model", lambda **kwargs: model_switch.ModelSwitchResult(
        success=True, new_model=kwargs["raw_input"], target_provider="custom",
        base_url="http://127.0.0.1:9/v1"))
    monkeypatch.setattr(model_selection_guards, "selection_warnings", lambda *args, **kwargs: [])
    runner, _source, _route, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        db.update_session_model("seg-1", "old")
        assert db.flush_token_counts()
        # Queue at the actual receipt boundary: earlier request reads legitimately
        # flush accounting, while a running old turn may enqueue after those reads.
        original_commit = picks.commit_chat_pick
        def queued_commit(*args, **kwargs):
            db._token_writer_thread = SimpleNamespace(is_alive=lambda: True)
            try:
                db.queue_token_counts("seg-1", model="old", billing_provider="old-provider",
                                      api_call_count=1, input_tokens=7)
            finally:
                db._token_writer_thread = None
            return original_commit(*args, **kwargs)
        monkeypatch.setattr(picks, "commit_chat_pick", queued_commit)
        picked = await viewer.dispatch(_mutate(root, fences, "model", {"model": "picked"}, "queued-pick"))
        assert picked.get("result", {}).get("model") == "picked", picked
        assert db.flush_token_counts()
        row = db.get_session("seg-1")
        assert row["input_tokens"] == 7
        assert row["model"] == "picked"
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()
