"""Tests for per-request reasoning_effort and fast_mode honour in the API server.

When API consumers (e.g. Mission Control, CCC task dispatcher, external
UIs) include ``reasoning_effort`` or ``fast_mode`` fields in their
request body, the API server must apply them for that call only, without
mutating ``config.yaml``.

Mirrors the layered pattern of ``test_api_server_model_override.py``:
exercise ``_create_agent`` directly so the override plumbing is covered
without spinning up an aiohttp test client (which the rest of the
gateway test suite avoids for the same reasons).
"""

from __future__ import annotations

from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    """Build a minimal APIServerAdapter with stubbed internals."""
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._model_name = "claude-opus-4-6"
    adapter._session_db = None
    return adapter


def _stub_runtime(monkeypatch, *, model="claude-opus-4-6", gateway_reasoning=None):
    """Patch the heavy imports that _create_agent() pulls in.

    *model* lets a test pin the resolved runtime model (e.g. an OpenAI
    name vs. an Anthropic name) so ``resolve_fast_mode_overrides`` picks
    the right provider-specific override shape.
    """
    monkeypatch.setattr(
        "gateway.platforms.api_server.APIServerAdapter._ensure_session_db",
        lambda self: None,
    )

    mock_agent_cls = MagicMock()
    mock_agent_cls.return_value = MagicMock()

    def _fake_resolve_runtime():
        return {
            "api_key": "sk-test",
            "base_url": "https://api.example.com",
            "provider": "anthropic",
            "api_mode": None,
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    def _fake_resolve_model():
        return model

    def _fake_load_config():
        return {"providers": {}}

    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", _fake_resolve_runtime)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", _fake_resolve_model)
    monkeypatch.setattr("gateway.run._load_gateway_config", _fake_load_config)
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda: gateway_reasoning),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda cfg, platform: ["terminal"],
    )
    monkeypatch.setattr("run_agent.AIAgent", mock_agent_cls)

    return mock_agent_cls


def _kwargs(mock_cls):
    """Return the kwargs of the last AIAgent(...) call."""
    return mock_cls.call_args.kwargs


# ---------------------------------------------------------------------------
# reasoning_effort
# ---------------------------------------------------------------------------

class TestCreateAgentReasoningEffortOverride:
    """_create_agent honours reasoning_effort_override for this call only."""

    def test_no_override_uses_gateway_default(self, monkeypatch):
        mock_cls = _stub_runtime(
            monkeypatch, gateway_reasoning={"enabled": True, "effort": "medium"}
        )
        adapter = _make_adapter()

        adapter._create_agent()

        assert _kwargs(mock_cls)["reasoning_config"] == {"enabled": True, "effort": "medium"}

    def test_valid_override_replaces_gateway_default(self, monkeypatch):
        mock_cls = _stub_runtime(
            monkeypatch, gateway_reasoning={"enabled": True, "effort": "medium"}
        )
        adapter = _make_adapter()

        adapter._create_agent(reasoning_effort_override="high")

        assert _kwargs(mock_cls)["reasoning_config"] == {"enabled": True, "effort": "high"}

    def test_none_disables_thinking(self, monkeypatch):
        mock_cls = _stub_runtime(
            monkeypatch, gateway_reasoning={"enabled": True, "effort": "medium"}
        )
        adapter = _make_adapter()

        adapter._create_agent(reasoning_effort_override="none")

        assert _kwargs(mock_cls)["reasoning_config"] == {"enabled": False}

    def test_unknown_effort_falls_back_to_gateway_default(self, monkeypatch):
        mock_cls = _stub_runtime(
            monkeypatch, gateway_reasoning={"enabled": True, "effort": "medium"}
        )
        adapter = _make_adapter()

        adapter._create_agent(reasoning_effort_override="ludicrous")

        assert _kwargs(mock_cls)["reasoning_config"] == {"enabled": True, "effort": "medium"}

    def test_empty_string_does_not_override(self, monkeypatch):
        mock_cls = _stub_runtime(
            monkeypatch, gateway_reasoning={"enabled": True, "effort": "low"}
        )
        adapter = _make_adapter()

        adapter._create_agent(reasoning_effort_override="")

        assert _kwargs(mock_cls)["reasoning_config"] == {"enabled": True, "effort": "low"}


# ---------------------------------------------------------------------------
# fast_mode
# ---------------------------------------------------------------------------

class TestCreateAgentFastModeOverride:
    """_create_agent honours fast_mode_override based on model support."""

    def test_no_override_leaves_request_overrides_none(self, monkeypatch):
        mock_cls = _stub_runtime(monkeypatch)
        adapter = _make_adapter()

        adapter._create_agent()

        assert _kwargs(mock_cls)["request_overrides"] is None

    def test_false_leaves_request_overrides_none(self, monkeypatch):
        mock_cls = _stub_runtime(monkeypatch)
        adapter = _make_adapter()

        adapter._create_agent(fast_mode_override=False)

        assert _kwargs(mock_cls)["request_overrides"] is None

    def test_anthropic_model_injects_speed_fast(self, monkeypatch):
        from hermes_cli.models import resolve_fast_mode_overrides, _is_anthropic_fast_model

        # Pick a known Anthropic fast-mode-capable model.  We probe
        # resolve_fast_mode_overrides directly so the test is decoupled
        # from the catalog as it evolves.
        anthropic_model = "claude-opus-4-6"
        if not _is_anthropic_fast_model(anthropic_model):
            anthropic_model = "claude-sonnet-4-5"
        expected = resolve_fast_mode_overrides(anthropic_model)
        assert expected is not None, "Anthropic fast-mode model picker is stale"

        mock_cls = _stub_runtime(monkeypatch, model=anthropic_model)
        adapter = _make_adapter()

        adapter._create_agent(fast_mode_override=True)

        assert _kwargs(mock_cls)["request_overrides"] == expected

    def test_openai_model_injects_service_tier_priority(self, monkeypatch):
        from hermes_cli.models import resolve_fast_mode_overrides, _is_openai_fast_model

        openai_model = "gpt-5"
        if not _is_openai_fast_model(openai_model):
            openai_model = "gpt-5.5"
        expected = resolve_fast_mode_overrides(openai_model)
        if expected is None:
            # The fast-mode catalog may not include every gpt-5* alias in
            # all forks — skip rather than hard-fail to keep this test
            # forward-compatible with upstream catalog updates.
            import pytest
            pytest.skip("No OpenAI fast-mode model available in this catalog")

        mock_cls = _stub_runtime(monkeypatch, model=openai_model)
        adapter = _make_adapter()

        adapter._create_agent(fast_mode_override=True)

        assert _kwargs(mock_cls)["request_overrides"] == expected

    def test_unsupported_model_silently_skips_fast_mode(self, monkeypatch):
        """Toggling fast_mode against a model without fast-mode support
        must not crash — it should just leave request_overrides empty."""
        mock_cls = _stub_runtime(monkeypatch, model="some-obscure-model-name")
        adapter = _make_adapter()

        adapter._create_agent(fast_mode_override=True)

        assert _kwargs(mock_cls)["request_overrides"] is None
