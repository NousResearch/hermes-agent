"""Model-context warm-up inside the gateway boot warm-up (#105986).

The startup warm-up used to prime only the import graph, tool schemas and
context files. The first inbound turn then paid a blocking catalog HTTP probe
(codex OAuth, OpenRouter metadata) inside AIAgent construction — between the
submit ACK and the actual inference request. The warm-up now resolves the
default route's context metadata up front with the same route/credential rules
as normal execution, so the probe's process-local caches are primed before the
inbound gate opens.
"""

from unittest.mock import patch

import gateway.run as gateway_run


def test_model_context_warmup_primes_default_route(monkeypatch):
    """Warm-up resolves the default route's model context once, best-effort."""
    resolved: list = []

    def fake_resolve(model=None):
        resolved.append(model)
        return "primed"

    monkeypatch.setattr(gateway_run, "_resolve_gateway_model_context", fake_resolve)
    import model_tools

    monkeypatch.setattr(
        model_tools, "get_tool_definitions", lambda quiet_mode=False: []
    )

    count = gateway_run._warm_turn_machinery_sync()
    assert count == 0
    assert resolved == [None]


def test_model_context_warmup_failure_is_non_fatal(monkeypatch):
    """A resolver failure degrades to lazy init — warm-up still returns the tool count."""

    def boom(_model=None):
        raise RuntimeError("catalog unreachable")

    monkeypatch.setattr(gateway_run, "_resolve_gateway_model_context", boom)
    import model_tools

    monkeypatch.setattr(
        model_tools, "get_tool_definitions", lambda quiet_mode=False: ["t"] * 7
    )

    count = gateway_run._warm_turn_machinery_sync()
    assert count == 7


def test_model_context_warmup_runs_after_tool_schemas(monkeypatch):
    """Schema materialization is never skipped because of metadata resolution order."""
    order: list = []

    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model_context",
        lambda model=None: order.append("model-context"),
    )
    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda quiet_mode=False: order.append("tool-schemas") or [],
    )

    gateway_run._warm_turn_machinery_sync()
    assert order == ["tool-schemas", "model-context"]
