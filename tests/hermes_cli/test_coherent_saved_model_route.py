"""Restore complete runtime snapshots without combining them with an old usage model."""
import json
from types import SimpleNamespace

from gateway.session_local_migration import legacy_launch_params
from hermes_cli.cli_model_switch_mixin import stored_session_route
from hermes_state import SessionDB
from tui_gateway.server import _runtime_model_config, _stored_session_runtime_overrides


def test_late_accounting_does_not_mix_saved_route_on_resume(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli", model="old")
        db.update_session_model("s", "picked", provider="custom:feather", base_url="http://127.0.0.1:9/v1")
        db.update_token_counts("s", model="old", billing_provider="old-provider", api_call_count=1, input_tokens=7)
        row = db.get_session("s")
        assert (row["model"], row["input_tokens"]) == ("old", 7)
        assert stored_session_route(row, current_model="ambient", current_provider="ambient")[:3] == (
            "picked", "custom:feather", "http://127.0.0.1:9/v1")
        params, _ = legacy_launch_params(row)
        assert (params["model"], params["provider"], params["base_url"]) == (
            "picked", "custom:feather", "http://127.0.0.1:9/v1")


def test_first_accounted_fallback_without_complete_snapshot_still_resumes_actual_route(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli", model="primary")
        db.update_token_counts("s", model="fallback", billing_provider="openrouter",
                               billing_base_url="https://openrouter.ai/api/v1", api_call_count=1, input_tokens=7)
        row = db.get_session("s")
        assert row["input_tokens"] == 7
        assert stored_session_route(row, current_model="ambient", current_provider="ambient")[:3] == (
            "fallback", "openrouter", None)
        usage = db._read_one("SELECT model, billing_provider, input_tokens FROM session_model_usage WHERE session_id='s'")
        assert tuple(usage) == ("fallback", "openrouter", 7)


def test_new_tui_snapshot_supersedes_stale_nested_gateway_route():
    agent = SimpleNamespace(model="new", provider="openrouter", requested_provider="openrouter",
                            base_url="https://openrouter.ai/api/v1", api_mode="chat_completions",
                            reasoning_config=None, service_tier=None)
    config = _runtime_model_config(agent, {"gateway_runtime": {"model": "old", "provider": "old-provider"}})
    row = {"model": "new", "model_config": json.dumps(config)}
    assert stored_session_route(row, current_model="ambient", current_provider="ambient")[:3] == (
        "new", "openrouter", "https://openrouter.ai/api/v1")
    desktop = _stored_session_runtime_overrides(row)["model_override"]
    assert (desktop["model"], desktop["provider"]) == ("new", "openrouter")


def test_newer_gateway_snapshot_replaces_same_provider_model(tmp_path):
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli", model="old")
        db.update_session_model("s", "old", provider="openrouter", base_url="https://openrouter.ai/api/v1")
        runner = SimpleNamespace(_session_db=SimpleNamespace(_db=db))
        agent = SimpleNamespace(model="new", provider="openrouter", requested_provider="openrouter",
                                base_url="https://openrouter.ai/api/v1", api_mode="chat_completions")
        GatewayTurnPrepareMixin._sync_session_model_from_agent(runner, "s", agent)
        row = db.get_session("s")
        assert stored_session_route(row, current_model="ambient", current_provider="ambient")[:3] == (
            "new", "openrouter", "https://openrouter.ai/api/v1")
        desktop = _stored_session_runtime_overrides(row)["model_override"]
        assert (desktop["model"], desktop["provider"]) == ("new", "openrouter")
        params, _ = legacy_launch_params(row)
        assert (params["model"], params["provider"]) == ("new", "openrouter")


def test_same_complete_snapshot_restores_same_route_for_late_usage_and_first_fallback(tmp_path):
    with SessionDB(tmp_path / "selected.db") as selected, SessionDB(tmp_path / "fallback.db") as fallback:
        selected.create_session("s", "cli", model="old")
        selected.update_session_model("s", "picked", provider="custom:feather", base_url="http://127.0.0.1:9/v1")
        snapshot = json.loads(selected.get_session("s")["model_config"])
        fallback.create_session("s", "cli", model="picked", model_config=snapshot)
        for db in (selected, fallback):
            db.update_token_counts("s", model="old", billing_provider="old-provider", api_call_count=1, input_tokens=7)
            row = db.get_session("s")
            assert (row["model"], row["billing_provider"], row["input_tokens"]) == ("old", "old-provider", 7)
            assert stored_session_route(row, current_model="ambient", current_provider="ambient")[:3] == (
                "picked", "custom:feather", "http://127.0.0.1:9/v1")
            usage = db._read_one("SELECT model, billing_provider, input_tokens FROM session_model_usage WHERE session_id='s'")
            assert tuple(usage) == ("old", "old-provider", 7)


def test_model_only_update_advances_existing_complete_snapshot(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli", model="old")
        db.update_session_model("s", "one", provider="openrouter", base_url="https://openrouter.ai/api/v1")
        db.update_session_model("s", "two")
        row = db.get_session("s")
        assert stored_session_route(row, current_model="ambient", current_provider="ambient")[:3] == (
            "two", "openrouter", "https://openrouter.ai/api/v1")


def test_legacy_complete_snapshot_does_not_borrow_old_billing_endpoint():
    row = {"source": "cli", "model": "old", "billing_base_url": "https://old.invalid/v1",
           "model_config": json.dumps({"model": "new", "provider": "openrouter"})}
    params, mode = legacy_launch_params(row)
    assert (params["model"], params["provider"]) == ("new", "openrouter")
    assert "base_url" not in params
    assert mode is None


def test_complete_top_level_wins_over_conflicting_incomplete_nested_route():
    row = {"source": "cli", "model": "old", "model_config": json.dumps({
        "model": "new", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "gateway_runtime": {"provider": "old-provider", "base_url": "https://old.invalid/v1"}})}
    assert stored_session_route(row, current_model="ambient", current_provider="ambient")[:3] == (
        "new", "openrouter", "https://openrouter.ai/api/v1")
    params, _ = legacy_launch_params(row)
    assert (params["model"], params["provider"], params["base_url"]) == (
        "new", "openrouter", "https://openrouter.ai/api/v1")


def test_complete_nested_route_wins_over_complete_top_level_route():
    row = {"model": "old", "model_config": json.dumps({
        "model": "top", "provider": "top-provider",
        "gateway_runtime": {"model": "nested", "provider": "nested-provider"}})}
    assert stored_session_route(row, current_model="ambient", current_provider="ambient")[:2] == (
        "nested", "nested-provider")
