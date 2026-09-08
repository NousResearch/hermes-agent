"""Model-context warm-up inside the gateway boot warm-up (#105986).

The startup warm-up primed the import graph and tool schemas, but the default
route's context-window metadata was only resolved on the first inbound turn —
a blocking catalog HTTP probe (codex OAuth, OpenRouter metadata) inside AIAgent
construction, between the submit ACK and the inference request. The warm-up now
resolves the default route's model context up front with the same route /
credential rules as the turn itself, so the probe's caches are primed before
the inbound gate opens.
"""

import gateway.run as gateway_run
import gateway.run_model_context as model_context


def _quiet_tool_side(monkeypatch, tool_count):
    import model_tools

    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda quiet_mode=False: ["t"] * tool_count)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: {"agent": {"environment_probe": False}})


def test_model_context_warmup_primes_default_route(monkeypatch):
    """Warm-up resolves the default gateway route's model context exactly once."""
    resolved: list = []

    def fake_resolve(model=None, route=None, *, warmup=False):
        resolved.append((model, route, warmup))
        return model_context._GatewayModelContext(
            model="m", provider="p", base_url="", context_length=128000, context_source="detected")

    monkeypatch.setattr(model_context, "_resolve_gateway_model_context", fake_resolve)
    _quiet_tool_side(monkeypatch, 3)

    assert gateway_run._warm_turn_machinery_sync() == 3
    assert resolved == [(None, None, True)]


def test_model_context_warmup_failure_is_non_fatal(monkeypatch):
    """A resolver failure degrades to lazy init — warm-up still returns the tool count."""

    def boom(model=None, route=None, *, warmup=False):
        raise RuntimeError("catalog unreachable")

    monkeypatch.setattr(model_context, "_resolve_gateway_model_context", boom)
    _quiet_tool_side(monkeypatch, 7)

    assert gateway_run._warm_turn_machinery_sync() == 7
