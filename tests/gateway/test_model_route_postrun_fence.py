"""A running messaging turn cannot overwrite a later viewer model selection."""
import asyncio
from types import SimpleNamespace
import threading
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_messaging_session_picks import _compressed_chat, _mutate


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["no_pick", "pick_before_sync", "pick_during_sync", "reset", "reset_configured", "classic"])
async def test_postrun_snapshot_fenced_at_commit(tmp_path, monkeypatch, order):
    from gateway.run_turn_runner import TurnRunner
    from hermes_cli import model_selection_guards, model_switch
    from hermes_cli.cli_model_switch_mixin import stored_session_route

    monkeypatch.setattr(model_switch, "switch_model", lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model=k["raw_input"], target_provider="openrouter",
        base_url="https://openrouter.ai/api/v1"))
    monkeypatch.setattr(model_selection_guards, "selection_warnings", lambda *a, **k: [])
    runner, source, route, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        if order in {"reset", "reset_configured"}:
            if order == "reset_configured":
                import gateway.run as gateway_run
                monkeypatch.setattr(runner, "_channel_default_route", lambda *a: ("default", "openrouter"))
                monkeypatch.setattr(gateway_run, "_load_gateway_config",
                                    lambda: {"model": {"default": "default", "provider": "openrouter"}})
            first = await viewer.dispatch(_mutate(root, fences, "model", {"model": "picked"}, "first"))
            assert first.get("result", {}).get("model") == "picked", first
            fences = first["result"]
        old = SimpleNamespace(session_id="seg-1", model="old", provider="openrouter",
                              requested_provider="openrouter", base_url="https://openrouter.ai/api/v1",
                              api_mode="chat_completions", _fallback_activated=(order == "no_pick"))
        ctx = SimpleNamespace(agent_holder=[old], session_id="seg-1", session_key=route,
                              source=source, inbound_message_id="old-turn")
        turn = object.__new__(TurnRunner)
        turn._ctx, turn._runner = ctx, runner
        turn._route_sync_fence = turn._capture_route_sync_fence()
        if order == "pick_during_sync":
            entered, release = threading.Event(), threading.Event()
            original = db.update_session_meta
            def stalled(*a, **kw):
                entered.set()
                assert release.wait(10)
                return original(*a, **kw)
            monkeypatch.setattr(db, "update_session_meta", stalled)
            task = asyncio.create_task(asyncio.to_thread(turn._sync_session_after_run, []))
            assert await asyncio.to_thread(entered.wait, 5)
        if order == "classic":
            import gateway.session_setting_commands as settings
            from gateway.slash_commands_model import _ModelSwitchContext
            monkeypatch.setattr(settings, "local_authority_live", lambda *a: (None, None))
            monkeypatch.setattr(runner, "_switch_cached_agent_model", lambda *a: None)
            monkeypatch.setattr(runner, "_record_switch_metrics", lambda *a: None)
            monkeypatch.setattr(runner, "_model_switch_confirmation", AsyncMock(return_value="ok"))
            switch_ctx = _ModelSwitchContext(session_key=route, source=source, config_path=tmp_path / "config.yaml",
                                              persist_global=False, current_model="old")
            result = model_switch.ModelSwitchResult(success=True, new_model="picked", target_provider="openrouter",
                                                     base_url="https://openrouter.ai/api/v1")
            assert await runner._commit_model_switch_locked(result, switch_ctx, source=source, picker=True) == "ok"
            expected = "picked"
        elif order != "no_pick":
            target = "reset" if order in {"reset", "reset_configured"} else "picked"
            picked = await viewer.dispatch(_mutate(root, fences, "model", {"model": target}, "second"))
            assert "result" in picked, picked
            expected = picked["result"]["model"]
        if order == "pick_during_sync":
            release.set()
            await task
        else:
            await asyncio.to_thread(turn._sync_session_after_run, [])
        row = db.get_session("seg-1")
        if order == "no_pick":
            assert row["model"] == "old"
            assert '"fallback_active": true' in row["model_config"]
        else:
            assert row["model"] == expected
            restored = stored_session_route(row, current_model="ambient", current_provider="ambient")
            if expected:
                assert restored[0] == expected
            else:
                assert restored is None
                assert "picked" not in (row["model_config"] or "")
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_turn_started_between_classic_db_write_and_override_publication(tmp_path, monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    import gateway.session_setting_commands as settings
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_cli.model_switch import ModelSwitchResult

    runner, source, route, viewer, _root, _fences = await _compressed_chat(tmp_path)
    release = asyncio.Event()
    written = asyncio.Event()
    try:
        monkeypatch.setattr(settings, "local_authority_live", lambda *a: (None, None))
        monkeypatch.setattr(runner, "_switch_cached_agent_model", lambda *a: None)
        monkeypatch.setattr(runner, "_record_switch_metrics", lambda *a: None)
        monkeypatch.setattr(runner, "_model_switch_confirmation", AsyncMock(return_value="ok"))
        original = runner._session_db.update_session_model
        async def paused(*args, **kwargs):
            await original(*args, **kwargs)
            written.set()
            await release.wait()
        monkeypatch.setattr(runner._session_db, "update_session_model", paused)
        switch_ctx = _ModelSwitchContext(session_key=route, source=source, config_path=tmp_path / "config.yaml",
                                          persist_global=False, current_model="old")
        result = ModelSwitchResult(success=True, new_model="picked", target_provider="openrouter",
                                   base_url="https://openrouter.ai/api/v1")
        pick = asyncio.create_task(runner._commit_model_switch(result, switch_ctx, source=source, picker=True))
        await asyncio.wait_for(written.wait(), timeout=5)
        assert runner._session_model_override(route) is None
        ctx = SimpleNamespace(agent_holder=[None], session_id="seg-1", session_key=route,
                              source=source, inbound_message_id="old-turn")
        turn = object.__new__(TurnRunner)
        turn._ctx, turn._runner = ctx, runner
        capture = asyncio.create_task(turn.capture_route_sync_fence())
        await asyncio.sleep(0)
        assert not capture.done(), "turn capture must wait for model publication"
        release.set()
        assert await pick == "ok"
        await capture
        selected = runner._session_model_override(route)
        assert selected["model"] == "picked"
        ctx.agent_holder[0] = SimpleNamespace(session_id="seg-1", model=selected["model"], provider="openrouter",
                                              requested_provider="openrouter", base_url=selected["base_url"],
                                              api_mode="chat_completions")
        await asyncio.to_thread(turn._sync_session_after_run, [])
        assert runner.session_store._db.get_session("seg-1")["model"] == "picked"
    finally:
        release.set()
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_turn_capture_waits_for_receipt_commit(tmp_path, monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    from hermes_cli import model_selection_guards, model_switch
    import hermes_state_mutation_chat_pick as picks

    monkeypatch.setattr(model_switch, "switch_model", lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model=k["raw_input"], target_provider="openrouter",
        base_url="https://openrouter.ai/api/v1"))
    monkeypatch.setattr(model_selection_guards, "selection_warnings", lambda *a, **k: [])
    runner, source, route, viewer, root, fences = await _compressed_chat(tmp_path)
    entered, release = threading.Event(), threading.Event()
    try:
        original = picks.commit_chat_pick
        def paused(*a, **kw):
            entered.set()
            assert release.wait(10)
            return original(*a, **kw)
        monkeypatch.setattr(picks, "commit_chat_pick", paused)
        pick = asyncio.create_task(viewer.dispatch(_mutate(root, fences, "model", {"model": "picked"}, "receipt")))
        assert await asyncio.to_thread(entered.wait, 5)
        ctx = SimpleNamespace(session_id="seg-1", session_key=route, source=source)
        turn = object.__new__(TurnRunner)
        turn._ctx, turn._runner = ctx, runner
        capture = asyncio.create_task(turn.capture_route_sync_fence())
        await asyncio.sleep(0)
        assert not capture.done(), "turn capture must wait for receipt commit"
        release.set()
        result = await pick
        assert result.get("result", {}).get("model") == "picked", result
        await capture
        assert turn._route_sync_fence[1] == result["result"]["revision"]
    finally:
        release.set()
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_uncontested_compression_turn_syncs_new_tip(tmp_path, monkeypatch):
    from gateway.run_turn_runner import TurnRunner

    runner, source, route, viewer, _root, _fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        agent = SimpleNamespace(session_id="seg-1", model="fallback", provider="openrouter",
                                requested_provider="openrouter", base_url="https://openrouter.ai/api/v1",
                                api_mode="chat_completions", _fallback_activated=True)
        ctx = SimpleNamespace(agent_holder=[agent], session_id="seg-1", session_key=route, source=source,
                              inbound_message_id="old-turn", _run_still_current=lambda: True, run_generation=1)
        turn = object.__new__(TurnRunner)
        turn._ctx, turn._runner = ctx, runner
        turn._route_sync_fence = turn._capture_route_sync_fence()
        db.publish_compression_child(parent_session_id="seg-1", child_session_id="seg-2", source="telegram",
                                     messages=[{"role": "user", "content": "summary"}],
                                     require_compression_lease=False)
        agent.session_id = "seg-2"
        monkeypatch.setattr(turn, "_restore_telegram_thread_id_after_split", lambda *a: None)
        monkeypatch.setattr(runner, "_sync_telegram_topic_binding", lambda *a, **kw: None)
        await asyncio.to_thread(turn._sync_session_after_run, [])
        row = db.get_session("seg-2")
        assert row["model"] == "fallback"
        assert '"fallback_active": true' in row["model_config"]
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_no_authority_classic_pick_fences_running_turn(tmp_path, monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    import gateway.session_authorities as authorities
    import gateway.session_setting_commands as settings
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_cli.model_switch import ModelSwitchResult

    runner, source, route, viewer, _root, _fences = await _compressed_chat(tmp_path)
    try:
        monkeypatch.setattr(authorities, "active_authority", lambda *a: None)
        monkeypatch.setattr(settings, "local_authority_live", lambda *a: (None, None))
        monkeypatch.setattr(runner, "_switch_cached_agent_model", lambda *a: None)
        monkeypatch.setattr(runner, "_record_switch_metrics", lambda *a: None)
        monkeypatch.setattr(runner, "_model_switch_confirmation", AsyncMock(return_value="ok"))
        old = SimpleNamespace(session_id="seg-1", model="old", provider="openrouter",
                              requested_provider="openrouter", base_url="https://openrouter.ai/api/v1",
                              api_mode="chat_completions")
        ctx = SimpleNamespace(agent_holder=[old], session_id="seg-1", session_key=route,
                              source=source, inbound_message_id="old-turn")
        turn = object.__new__(TurnRunner)
        turn._ctx, turn._runner = ctx, runner
        await turn.capture_route_sync_fence()
        switch_ctx = _ModelSwitchContext(session_key=route, source=source, config_path=tmp_path / "config.yaml",
                                          persist_global=False, current_model="old")
        result = ModelSwitchResult(success=True, new_model="picked", target_provider="openrouter",
                                   base_url="https://openrouter.ai/api/v1")
        assert await runner._commit_model_switch(result, switch_ctx, source=source, picker=True) == "ok"
        await asyncio.to_thread(turn._sync_session_after_run, [])
        assert runner.session_store._db.get_session("seg-1")["model"] == "picked"
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_classic_pick_after_split_before_route_publication_targets_tip(tmp_path, monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    import gateway.session_setting_commands as settings
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_cli.model_switch import ModelSwitchResult
    from hermes_cli.cli_model_switch_mixin import stored_session_route

    runner, source, route, viewer, _root, _fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        db.update_session_model("seg-1", "old", provider="openrouter")
        old = SimpleNamespace(session_id="seg-1", model="old", provider="openrouter",
                              requested_provider="openrouter", base_url="https://openrouter.ai/api/v1",
                              api_mode="chat_completions")
        ctx = SimpleNamespace(agent_holder=[old], session_id="seg-1", session_key=route, source=source,
                              inbound_message_id="old-turn", _run_still_current=lambda: True, run_generation=1)
        turn = object.__new__(TurnRunner)
        turn._ctx, turn._runner = ctx, runner
        await turn.capture_route_sync_fence()
        db.publish_compression_child(parent_session_id="seg-1", child_session_id="seg-2", source="telegram",
                                     messages=[{"role": "user", "content": "summary"}],
                                     require_compression_lease=False)
        old.session_id = "seg-2"
        monkeypatch.setattr(settings, "local_authority_live", lambda *a: (None, None))
        monkeypatch.setattr(runner, "_switch_cached_agent_model", lambda *a: None)
        monkeypatch.setattr(runner, "_record_switch_metrics", lambda *a: None)
        monkeypatch.setattr(runner, "_model_switch_confirmation", AsyncMock(return_value="ok"))
        monkeypatch.setattr(turn, "_restore_telegram_thread_id_after_split", lambda *a: None)
        monkeypatch.setattr(runner, "_sync_telegram_topic_binding", lambda *a, **kw: None)
        switch_ctx = _ModelSwitchContext(session_key=route, source=source, config_path=tmp_path / "config.yaml",
                                          persist_global=False, current_model="old")
        result = ModelSwitchResult(success=True, new_model="picked", target_provider="openrouter",
                                   base_url="https://openrouter.ai/api/v1")
        assert await runner._commit_model_switch(result, switch_ctx, source=source, picker=True) == "ok"
        await asyncio.to_thread(turn._sync_session_after_run, [])
        row = db.get_session("seg-2")
        assert stored_session_route(row, current_model="ambient", current_provider="ambient")[0] == "picked"
        from gateway.session_local_migration import legacy_launch_params
        assert legacy_launch_params(row)[0]["model"] == "picked"
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_compression_target_metadata_change_after_sync_read_survives(tmp_path, monkeypatch):
    from gateway.run_turn_runner import TurnRunner

    runner, source, route, viewer, _root, _fences = await _compressed_chat(tmp_path)
    entered, release = threading.Event(), threading.Event()
    try:
        db = runner.session_store._db
        agent = SimpleNamespace(session_id="seg-1", model="fallback", provider="openrouter",
                                requested_provider="openrouter", base_url="https://openrouter.ai/api/v1",
                                api_mode="chat_completions", _fallback_activated=True)
        ctx = SimpleNamespace(agent_holder=[agent], session_id="seg-1", session_key=route, source=source,
                              inbound_message_id="old-turn", _run_still_current=lambda: True, run_generation=1)
        turn = object.__new__(TurnRunner)
        turn._ctx, turn._runner = ctx, runner
        await turn.capture_route_sync_fence()
        db.publish_compression_child(parent_session_id="seg-1", child_session_id="seg-2", source="telegram",
                                     messages=[{"role": "user", "content": "summary"}],
                                     require_compression_lease=False)
        agent.session_id = "seg-2"
        monkeypatch.setattr(turn, "_restore_telegram_thread_id_after_split", lambda *a: None)
        monkeypatch.setattr(runner, "_sync_telegram_topic_binding", lambda *a, **kw: None)
        original = db.update_session_meta
        def stalled(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)
        monkeypatch.setattr(db, "update_session_meta", stalled)
        task = asyncio.create_task(asyncio.to_thread(turn._sync_session_after_run, []))
        assert await asyncio.to_thread(entered.wait, 5)
        db.patch_session_model_config("seg-2", {"concurrent_note": "preserve"})
        release.set()
        await task
        assert '"concurrent_note": "preserve"' in db.get_session("seg-2")["model_config"]
    finally:
        release.set()
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("picker", [False, True])
async def test_classic_reset_without_default_clears_stored_route(tmp_path, monkeypatch, picker):
    import gateway.session_setting_commands as settings
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_cli.cli_model_switch_mixin import stored_session_route

    runner, source, route, viewer, _root, _fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        db.update_session_model("seg-1", "picked", provider="openrouter")
        runner._session_model_overrides[route] = {"model": "picked", "provider": "openrouter"}
        runner.session_store.set_model_override(route, {"model": "picked", "provider": "openrouter"})
        monkeypatch.setattr(settings, "local_authority_live", lambda *a: (None, None))
        monkeypatch.setattr(runner, "_channel_default_route", lambda *a: ("", ""))
        ctx = _ModelSwitchContext(session_key=route, source=source, config_path=tmp_path / "config.yaml",
                                  persist_global=False, current_model="picked")
        await runner._handle_model_reset(None, ctx, picker=picker)
        row = db.get_session("seg-1")
        assert row["model"] == ""
        assert stored_session_route(row, current_model="ambient", current_provider="ambient") is None
        assert "picked" not in (row["model_config"] or "")
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_receipt_reset_failed_barrier_never_writes_provisional_model(tmp_path, monkeypatch):
    import hermes_state_mutation_chat_pick as picks

    runner, _source, route, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        db.update_session_model("seg-1", "picked", provider="openrouter")
        saved = {"model": "picked", "provider": "openrouter"}
        runner._session_model_overrides[route] = dict(saved)
        runner.session_store.set_model_override(route, saved)
        before_root, before_tip = db.get_session(root), db.get_session("seg-1")
        original = picks.commit_chat_pick
        def failed(*args, **kwargs):
            flush = db.flush_token_counts
            db.flush_token_counts = lambda: False
            try:
                return original(*args, **kwargs)
            finally:
                db.flush_token_counts = flush
        monkeypatch.setattr(picks, "commit_chat_pick", failed)
        refusal = await viewer.dispatch(_mutate(root, fences, "model", {"model": "reset"}, "refused-reset"))
        assert refusal.get("error", {}).get("data", {}).get("reason") == "storage_unavailable", refusal
        assert (db.get_session(root), db.get_session("seg-1")) == (before_root, before_tip)
        assert runner._session_model_override(route) == saved
        assert runner.session_store.get_model_override(route) == saved
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()
