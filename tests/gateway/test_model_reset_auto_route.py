"""Automatic reset must not resurrect historic billing provenance as a runtime route."""
import json

import pytest

from tests.gateway.test_messaging_session_picks import _compressed_chat, _mutate
from tests.hermes_cli.test_resume_model_restore import _make_stub


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["typed", "picker", "receipt"])
@pytest.mark.parametrize("default_model", ["new-default", ""])
async def test_real_auto_normalizer_reset_keeps_old_billing_out_of_all_readers(tmp_path, monkeypatch, producer, default_model):
    import gateway.run as gateway_run
    from gateway.slash_commands_model import _ModelSwitchContext
    from gateway.session_local_migration import legacy_launch_params
    from hermes_cli.cli_model_switch_mixin import stored_session_route
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from tui_gateway.server import _stored_session_runtime_overrides

    cfg = {"model": {"default": default_model, "provider": "auto",
                     "base_url": "http://127.0.0.1:9877/v1", "api_mode": "chat_completions"}}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kwargs: {
        "base_url": "http://127.0.0.1:9877/v1", "api_mode": "chat_completions", "api_key": "profile-key"})
    runner, source, route, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        old_url = "http://127.0.0.1:9876/v1"
        db.update_session_model("seg-1", "old-model", provider="custom:work",
                                base_url=old_url, api_mode="anthropic_messages")
        db.update_token_counts("seg-1", model="old-model", billing_provider="custom:work",
                               billing_base_url=old_url, billing_mode="anthropic_messages",
                               input_tokens=5, api_call_count=1)
        old_override = {"model": "old-model", "provider": "custom:work", "base_url": old_url}
        runner._session_model_overrides[route] = dict(old_override)
        runner.session_store.set_model_override(route, old_override)
        # Real production normalizer: only configuration loading is replaced.
        assert runner._channel_default_route(source, tmp_path / "config.yaml") == (default_model, "")
        if producer == "receipt":
            result = await viewer.dispatch(_mutate(root, fences, "model", {"model": "reset"}, "reset"))
            assert "result" in result, result
        else:
            ctx = _ModelSwitchContext(session_key=route, source=source, config_path=tmp_path / "config.yaml",
                                      persist_global=False, current_model="old-model")
            await runner._handle_model_reset(None, ctx, picker=(producer == "picker"))
        row = db.get_session("seg-1")
        assert row["model"] == default_model
        assert (row["billing_provider"], row["billing_base_url"], row["input_tokens"]) == ("custom:work", old_url, 5)
        assert json.loads(row["model_config"] or "{}").get("provider") == "auto"
        cli = stored_session_route(row, current_model="ambient", current_provider="nous")
        legacy, legacy_mode = legacy_launch_params(row)
        # Auto is not a missing route to heal from a model catalog.
        monkeypatch.setattr("hermes_cli.runtime_provider.canonical_custom_identity",
                            lambda **kwargs: "custom:work")
        tui = _stored_session_runtime_overrides(row)
        profile_url = "http://127.0.0.1:9877/v1"
        stub = _make_stub(base_url=profile_url, api_mode="chat_completions")
        stub._restore_session_model(row)
        explicit = _make_stub(model="manual", base_url="http://127.0.0.1:9880/v1", _explicit_model_override=True)
        explicit._restore_session_model(row)
        assert (explicit.model, explicit.base_url) == ("manual", "http://127.0.0.1:9880/v1")
        assert not cli or (cli[1] != "custom:work" and cli[2] != old_url)
        assert legacy.get("provider") is None and legacy.get("base_url") is None and legacy_mode is None
        from gateway.session_policy import build_policy
        policy = build_policy({**legacy, "cwd": str(tmp_path)}, cfg, profile_terminal=False)
        assert policy.config()["model"]["base_url"] == "http://127.0.0.1:9877/v1"
        assert policy.config()["model"]["api_mode"] == "chat_completions"
        model, runtime = _resolve_policy_agent_runtime(runner, policy, fallback=False)
        assert model == (default_model or None)
        assert runtime["provider"] != "custom:work"
        assert runtime["base_url"] == profile_url
        assert runtime["api_mode"] == "chat_completions"
        assert runtime["api_key"] == "profile-key"
        assert "custom:work" not in str(tui) and old_url not in str(tui)
        assert stub.provider != "custom:work" and stub.base_url == profile_url
        assert stub.api_mode == "chat_completions"
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_auto_reset_receipt_retry_and_stale_fences(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    cfg = {"model": {"default": "new-default", "provider": "auto"}}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *args, **kwargs: cfg)
    runner, _source, route, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        db.update_session_model("seg-1", "old", provider="custom:work")
        runner._session_model_overrides[route] = {"model": "old", "provider": "custom:work"}
        runner.session_store.set_model_override(route, {"model": "old", "provider": "custom:work"})
        request = _mutate(root, fences, "model", {"model": "reset"}, "reset-retry")
        first = await viewer.dispatch(request)
        assert first.get("result", {}).get("model") == "new-default", first
        before = db.get_session(root), db.get_session("seg-1")
        replay = await viewer.dispatch(request)
        assert replay.get("result") == first["result"]
        assert (db.get_session(root), db.get_session("seg-1")) == before
        stale = await viewer.dispatch(_mutate(root, fences, "model", {"model": "reset"}, "reset-stale"))
        assert stale.get("error", {}).get("data", {}).get("reason") == "revision_conflict", stale
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_auto_reset_failed_receipt_barrier_restores_prior_state(tmp_path, monkeypatch):
    import gateway.run as gateway_run
    import hermes_state_mutation_chat_pick as picks

    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda *args, **kwargs: {"model": {"default": "new-default", "provider": "auto"}})
    runner, _source, route, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        db.update_session_model("seg-1", "old", provider="custom:work")
        prior = {"model": "old", "provider": "custom:work"}
        runner._session_model_overrides[route] = dict(prior)
        runner.session_store.set_model_override(route, prior)
        before = db.get_session(root), db.get_session("seg-1")
        original = picks.commit_chat_pick
        def refuse(*args, **kwargs):
            flush = db.flush_token_counts
            db.flush_token_counts = lambda: False
            try:
                return original(*args, **kwargs)
            finally:
                db.flush_token_counts = flush
        monkeypatch.setattr(picks, "commit_chat_pick", refuse)
        result = await viewer.dispatch(_mutate(root, fences, "model", {"model": "reset"}, "refused"))
        assert result.get("error", {}).get("data", {}).get("reason") == "storage_unavailable", result
        assert (db.get_session(root), db.get_session("seg-1")) == before
        assert runner._session_model_override(route) == prior
        assert runner.session_store.get_model_override(route) == prior
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


def test_unreset_legacy_billing_route_remains_restorable(tmp_path):
    from hermes_state import SessionDB
    from gateway.session_local_migration import legacy_launch_params
    from hermes_cli.cli_model_switch_mixin import stored_session_route

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("legacy", "cli", model="old")
        db.update_token_counts("legacy", model="old", billing_provider="custom:work",
                               billing_base_url="http://127.0.0.1:9876/v1", input_tokens=5, api_call_count=1)
        row = db.get_session("legacy")
        route = stored_session_route(row, current_model="ambient", current_provider="nous")
        legacy, _ = legacy_launch_params(row)
        assert route[1] == "custom:work"
        assert legacy["provider"] == "custom:work"
        assert legacy["base_url"] == "http://127.0.0.1:9876/v1"


@pytest.mark.parametrize("default_model", ["new-default", ""])
def test_auto_reset_route_survives_history_export_import(tmp_path, default_model):
    from hermes_state import SessionDB
    from gateway.session_local_migration import legacy_launch_params
    from hermes_cli.cli_model_switch_mixin import stored_session_route

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("source", "cli", model="old")
        db.update_token_counts("source", model="old", billing_provider="custom:work",
                               billing_base_url="http://127.0.0.1:9876/v1", input_tokens=5, api_call_count=1)
        db.update_session_model("source", default_model, provider="auto")
        payload = db.export_session("source")
        payload["id"] = "copy"
        imported = db.import_sessions([payload])
        assert imported["ok"] and imported["imported"] == 1, imported
        row = db.get_session("copy")
        assert row["billing_provider"] == "custom:work" and row["input_tokens"] == 5
        route = stored_session_route(row, current_model="ambient", current_provider="nous")
        params, mode = legacy_launch_params(row)
        if default_model:
            assert route[:3] == (default_model, "auto", None)
            assert params["model"] == default_model
        else:
            assert route is None and "model" not in params
        assert "provider" not in params and "base_url" not in params and mode is None


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["typed", "picker", "receipt"])
async def test_nonautomatic_channel_reset_preserves_channel_provider(tmp_path, monkeypatch, producer):
    import gateway.run as gateway_run
    from types import SimpleNamespace
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_cli.cli_model_switch_mixin import stored_session_route

    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda *args, **kwargs: {"model": {"default": "profile", "provider": "auto"}})
    runner, source, route, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        db.update_session_model("seg-1", "old", provider="custom:work")
        runner._session_model_overrides[route] = {"model": "old", "provider": "custom:work"}
        runner.session_store.set_model_override(route, {"model": "old", "provider": "custom:work"})
        monkeypatch.setattr(runner, "_channel_override_for",
                            lambda *args: SimpleNamespace(model="channel-model", provider="openrouter"))
        assert runner._channel_default_route(source, tmp_path / "config.yaml") == ("channel-model", "openrouter")
        if producer == "receipt":
            result = await viewer.dispatch(_mutate(root, fences, "model", {"model": "reset"}, "channel-reset"))
            assert "result" in result, result
        else:
            ctx = _ModelSwitchContext(session_key=route, source=source, config_path=tmp_path / "config.yaml",
                                      persist_global=False, current_model="old")
            await runner._handle_model_reset(None, ctx, picker=(producer == "picker"))
        row = db.get_session("seg-1")
        assert row["model"] == "channel-model"
        assert stored_session_route(row, current_model="ambient", current_provider="nous")[:2] == (
            "channel-model", "openrouter")
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()
