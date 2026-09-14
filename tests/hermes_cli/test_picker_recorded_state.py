"""S: real inventory/provider modules, synthetic local/global stores only."""
import json
import os
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import inventory
from agent.credential_pool import load_pool as real_load_pool
real_apply_pricing = inventory._apply_pricing


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    from hermes_cli import auth
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "alpha"
    profile.mkdir(parents=True)
    home = tmp_path / "fakehome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("CODEX_HOME", str(home / "codex"))
    from hermes_constants import get_default_hermes_root
    assert get_default_hermes_root() == root
    assert "PYTEST_CURRENT_TEST" in os.environ
    assert root / "auth.json" != home / ".hermes" / "auth.json"
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("no network")))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda *a, **k: ["gpt-5.3-codex"])
    monkeypatch.setattr("hermes_cli.models._credential_fingerprint", lambda *a: "synthetic")
    monkeypatch.setattr("hermes_cli.models._load_provider_models_cache", lambda: {
        "openai-codex": {"fp": "synthetic", "at": 2000000000, "models": ["gpt-5.3-codex"]}})
    monkeypatch.setattr("hermes_cli.model_switch_providers._build_curated_lists", lambda *a: {})
    monkeypatch.setattr("hermes_cli.model_switch_providers._collect_authed_provider_slugs", lambda *a: [])
    for name in ("_apply_pricing", "_apply_capabilities", "_apply_featured", "_prewarm_pricing_async"):
        monkeypatch.setattr(inventory, name, lambda *a, **k: None)
    monkeypatch.setattr(inventory, "_local_runtime_row", lambda *a: None)
    monkeypatch.setattr(inventory, "_moa_provider_row", lambda *a: None)
    # Other provider discovery is outside this fixture; exercise the real Codex ladder.
    from hermes_cli.providers import HERMES_OVERLAYS
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {"openai-codex": HERMES_OVERLAYS["openai-codex"]})
    # A picker may not add a mutating load/availability/refresh path.
    monkeypatch.setattr("agent.credential_pool.load_pool", Mock(side_effect=AssertionError("mutating pool load")))
    return root, profile


def store(path, state, *, suppression=False, singleton=False):
    entries = [] if state == "empty" else [{
        "id": "synthetic-account", "source": "device_code", "auth_type": "oauth",
        "access_token": "synthetic-access", "refresh_token": "synthetic-refresh",
        "last_status": state, "last_status_at": 2000000000,
        "last_error_reset_at": 4102444800 if state == "exhausted" else None,
    }]
    path.write_text(json.dumps({"providers": {"openai-codex": {"tokens": {
        "access_token": "synthetic-access", "refresh_token": "synthetic-refresh"}}} if singleton else {},
                                "credential_pool": {"openai-codex": entries},
                                "suppressed_sources": {"openai-codex": ["device_code"]} if suppression else {}}))


def context(**kwargs):
    return inventory.ConfigContext("", "", "", kwargs.get("user_providers", {}), [], kwargs.get("excluded", []))


@pytest.mark.parametrize("owner", ["local", "global"])
@pytest.mark.parametrize("state", ["exhausted", "dead", "ok", "empty"])
@pytest.mark.parametrize("surface", ["options", "aux", "cli"])
@pytest.mark.parametrize("singleton", [False, True])
def test_real_picker_payload(fleet, monkeypatch, owner, state, surface, singleton):
    root, profile = fleet
    target = (profile if owner == "local" else root) / "auth.json"
    store(target, state, singleton=singleton and state != "empty")
    before = target.read_bytes()
    ctx = context()
    if surface == "options":
        rows = inventory.build_model_options_payload(ctx)["providers"]
    elif surface == "aux":
        monkeypatch.setattr(inventory, "load_picker_context", lambda: ctx)
        rows = inventory.build_aux_picker_rows()
    else:
        from hermes_cli.cli_model_switch_mixin import _show_model_picker
        cli = SimpleNamespace(model="other", provider="", _open_model_picker=Mock())
        _show_model_picker(cli, ctx, False)
        rows = cli._open_model_picker.call_args.args[0] if cli._open_model_picker.called else []
    codex = [r for r in rows if r["slug"] == "openai-codex"]
    assert bool(codex) == (state != "empty")
    if codex:
        row = codex[0]
        assert row["models"] == ["gpt-5.3-codex"]
        assert row["credential_present"] is True
        assert row["availability_source"] == "recorded_pool"
        assert row["auth_state"] == ("invalid" if state == "dead" else "present")
        assert row["available"] is (None if state == "ok" else False)
        if state == "exhausted":
            assert "recorded" in row["warning"]
            assert "2100-01-01" in row["warning"]
            assert "not checked" in row["warning"]
            rendered = inventory.format_aux_picker_entries([row])[0][1]
            assert row["warning"] in rendered
    assert target.read_bytes() == before
    if owner == "global":
        assert not (profile / "auth.json").exists()


@pytest.mark.parametrize("policy", ["excluded", "disabled"])
def test_policy_filters(fleet, policy):
    root, profile = fleet
    store(root / "auth.json", "exhausted")
    ctx = context(excluded=["openai-codex"] if policy == "excluded" else [],
                  user_providers={"openai-codex": {"enabled": False}} if policy == "disabled" else {})
    for unconfigured in (False, True):
        rows = inventory.build_model_options_payload(ctx, include_unconfigured=unconfigured)["providers"]
        assert not any(r["slug"] == "openai-codex" for r in rows)


def test_suppression_is_not_reinterpreted(fleet):
    root, profile = fleet
    store(root / "auth.json", "exhausted")
    store(profile / "auth.json", "empty", suppression=True)
    from hermes_cli.auth import is_source_suppressed
    assert is_source_suppressed("openai-codex", "device_code")
    assert any(r["slug"] == "openai-codex" for r in inventory.build_model_options_payload(context())["providers"])


def test_runtime_default_not_changed(fleet):
    root, _ = fleet
    store(root / "auth.json", "exhausted")
    # Existing runtime-facing default does not opt into persisted-presence fallback.
    assert not any(r["slug"] == "openai-codex" for r in inventory.build_models_payload(context())["providers"])


def test_real_runtime_pool_stays_unavailable_after_picker(fleet, monkeypatch):
    root, profile = fleet
    store(root / "auth.json", "exhausted", singleton=True)
    monkeypatch.setattr("agent.credential_pool.load_pool", real_load_pool)
    probe = Mock(return_value=False)
    monkeypatch.setattr("hermes_cli.auth._probe_codex_quota_restored", probe)
    assert real_load_pool("openai-codex").has_available() is False
    before = (root / "auth.json").read_bytes()
    probe.reset_mock()
    rows = inventory.build_model_options_payload(context())["providers"]
    assert any(r["slug"] == "openai-codex" for r in rows)
    probe.assert_not_called()
    assert (root / "auth.json").read_bytes() == before
    assert not (profile / "auth.json").exists()
    assert real_load_pool("openai-codex").has_available() is False


@pytest.mark.parametrize("cached", [False, True])
def test_cooldown_catalog_does_not_refresh_auth_on_cache_miss(fleet, monkeypatch, cached):
    root, _ = fleet
    store(root / "auth.json", "exhausted")
    if not cached:
        monkeypatch.setattr("hermes_cli.models._load_provider_models_cache", lambda: {})
    live = Mock(side_effect=AssertionError("live catalog could refresh OAuth"))
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", live)
    row = next(r for r in inventory.build_model_options_payload(context())["providers"]
               if r["slug"] == "openai-codex")
    assert "gpt-5.3-codex" in row["models"]
    live.assert_not_called()


def test_cooldown_pricing_does_not_query_tier_or_usage(fleet, monkeypatch):
    root, _ = fleet
    store(root / "auth.json", "exhausted")
    from hermes_cli.picker_state import recorded_pool_state
    row = {"slug": "openai-codex", "models": ["gpt-5.3-codex"], **recorded_pool_state("openai-codex")}
    lookup = Mock(side_effect=AssertionError("pricing lookup"))
    monkeypatch.setattr("hermes_cli.models_pricing.get_pricing_for_provider", lookup)
    real_apply_pricing([row], force_fresh_nous_tier=True, cached_only=False)
    lookup.assert_not_called()
    assert row["pricing_pending"] is True


@pytest.mark.parametrize("transport", ["api", "rpc"])
def test_serving_handlers_preserve_recorded_state(fleet, monkeypatch, transport):
    root, _ = fleet
    store(root / "auth.json", "exhausted")
    monkeypatch.setattr(inventory, "load_picker_context", lambda: context())
    if transport == "rpc":
        import tui_gateway.server as server
        monkeypatch.setattr(server, "_model_picker_context", lambda agent: context())
        payload = server._methods["model.options"](1, {})["result"]
    else:
        import asyncio
        from gateway.platforms.api_server import APIServerAdapter
        adapter = SimpleNamespace(_check_auth=lambda request: None)
        response = asyncio.run(APIServerAdapter._handle_model_options(adapter, SimpleNamespace(query={})))
        assert response.status == 200
        payload = json.loads(response.text)
    row = next(r for r in payload["providers"] if r["slug"] == "openai-codex")
    assert row["models"] == ["gpt-5.3-codex"]
    assert row["available"] is False
    assert "not checked" in row["warning"]


@pytest.mark.parametrize("stage", ["provider", "model"])
def test_cli_warning_is_really_rendered(fleet, stage):
    from cli import HermesCLI
    root, _ = fleet
    store(root / "auth.json", "exhausted")
    row = next(r for r in inventory.build_model_options_payload(context())["providers"]
               if r["slug"] == "openai-codex")
    cli = object.__new__(HermesCLI)
    cli._model_picker_state = {"stage": stage, "selected": 0, "providers": [row],
                               "provider_data": row, "model_list": row["models"]}
    text = "".join(fragment[1] for fragment in cli._get_model_picker_display_fragments())
    assert "recorded limit" in text
    assert "2100-01-01" in text
    text = " ".join(text.replace("│", " ").split())
    assert "not checked" in text
    assert "gpt-5.3-codex" in text if stage == "model" else row["name"] in text


def test_protected_path_guard_still_rejects(fleet, monkeypatch):
    from hermes_cli import auth
    from pathlib import Path
    protected = Path(os.environ["HOME"]) / ".hermes"
    protected.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(protected))
    with pytest.raises(RuntimeError):
        auth._auth_file_path()


@pytest.mark.parametrize("reset", [None, "bad", 1, "nan", "2100-01-01T00:00:00Z"])
def test_recorded_state_does_not_invent_reset_or_remote_health(fleet, reset):
    from hermes_cli.picker_state import recorded_pool_state
    root, _ = fleet
    target = root / "auth.json"
    target.write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"access_token": "fake", "last_status": "exhausted", "last_error_reset_at": reset}]}}))
    state = recorded_pool_state("openai-codex")
    assert state["available"] is (False if reset == "2100-01-01T00:00:00Z" else None)
    assert "not checked" in state["warning"]
    assert "access_token" not in state
