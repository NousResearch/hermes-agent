"""Phase-2 tests for the junie-acp provider: recognition + interactive flow.

Covers what the runtime-path tests (tests/agent/test_junie_acp_client.py) do
not: that `hermes model` / setup machinery recognizes junie-acp and that the
interactive model flow writes the right config.
"""
from __future__ import annotations

import pytest

from hermes_cli import models as M
from hermes_cli import providers as P
from hermes_cli.provider_catalog import provider_catalog_by_slug


def test_junie_acp_is_canonical_provider():
    slugs = {e.slug for e in M.CANONICAL_PROVIDERS}
    assert "junie-acp" in slugs


def test_junie_acp_in_provider_catalog_with_label():
    by = provider_catalog_by_slug()
    assert "junie-acp" in by
    assert by["junie-acp"].label
    assert by["junie-acp"].description


@pytest.mark.parametrize("alias", ["junie", "jetbrains-junie-acp", "junie-acp-agent"])
def test_junie_aliases_resolve(alias):
    assert P.ALIASES.get(alias) == "junie-acp"
    full = P.resolve_provider_full(alias)
    assert full is not None


def test_junie_acp_static_catalog_is_sentinel_only():
    # The STATIC reverse-map must stay sentinel-only so detect_provider_for_model
    # doesn't mis-resolve real claude/gemini/gpt ids to junie-acp (commit
    # 09fb55ac6). Live discovery (below) is layered on top without touching it.
    assert M._PROVIDER_MODELS["junie-acp"] == ["junie-acp"]


def test_junie_acp_falls_back_to_sentinel_when_discovery_unavailable(monkeypatch):
    # No Junie CLI / not authed / offline -> fetch returns None -> sentinel only.
    monkeypatch.setattr("agent.junie_acp_client.fetch_junie_models", lambda **_: None)
    ids = M.provider_model_ids("junie-acp", force_refresh=True)
    assert ids == ["junie-acp"]


def test_junie_acp_live_models_are_merged_sentinel_first(monkeypatch):
    # When Junie advertises models over ACP, the picker surfaces them — with the
    # sentinel kept first — without duplicating it if Junie also returns it.
    monkeypatch.setattr(
        "agent.junie_acp_client.fetch_junie_models",
        lambda **_: ["claude-fable-5", "claude-opus-4-8", "junie-acp"],
    )
    ids = M.provider_model_ids("junie-acp", force_refresh=True)
    assert ids[0] == "junie-acp"
    assert ids == ["junie-acp", "claude-fable-5", "claude-opus-4-8"]


def test_junie_acp_provider_model_ids(monkeypatch):
    monkeypatch.setattr("agent.junie_acp_client.fetch_junie_models", lambda **_: None)
    ids = M.provider_model_ids("junie-acp")
    assert "junie-acp" in ids


def test_junie_acp_flow_writes_config(monkeypatch):
    """The interactive flow persists provider/base_url/api_mode correctly."""
    import hermes_cli.auth as auth
    from hermes_cli.model_setup_flows import _model_flow_junie_acp
    from hermes_cli.config import load_config

    monkeypatch.setattr(
        auth, "get_external_process_provider_status",
        lambda pid: {"resolved_command": "/usr/bin/junie", "command": "junie",
                     "base_url": "acp://junie"},
    )
    monkeypatch.setattr(
        auth, "resolve_external_process_provider_credentials",
        lambda pid: {"base_url": "acp://junie", "command": "/usr/bin/junie",
                     "args": ["--acp=true"]},
    )
    monkeypatch.setattr(auth, "_prompt_model_selection", lambda *a, **k: "gemini-3-flash-preview")

    _model_flow_junie_acp({}, current_model="")

    cfg = load_config()
    assert isinstance(cfg["model"], dict)
    assert cfg["model"]["provider"] == "junie-acp"
    assert cfg["model"]["base_url"] == "acp://junie"
    assert cfg["model"]["api_mode"] == "chat_completions"


def test_junie_acp_flow_missing_cli_does_not_write(monkeypatch):
    """When the Junie CLI can't be resolved, the flow bails without writing."""
    import hermes_cli.auth as auth
    from hermes_cli.model_setup_flows import _model_flow_junie_acp
    from hermes_cli.config import load_config

    monkeypatch.setattr(
        auth, "get_external_process_provider_status",
        lambda pid: {"command": "junie", "base_url": "acp://junie"},
    )

    def _raise(pid):
        raise RuntimeError("Could not find the CLI command 'junie'")

    monkeypatch.setattr(auth, "resolve_external_process_provider_credentials", _raise)
    # Selection must never be reached.
    monkeypatch.setattr(
        auth, "_prompt_model_selection",
        lambda *a, **k: pytest.fail("should not prompt when CLI is missing"),
    )

    _model_flow_junie_acp({}, current_model="")

    cfg = load_config()
    model = cfg.get("model")
    if isinstance(model, dict):
        assert model.get("provider") != "junie-acp"
