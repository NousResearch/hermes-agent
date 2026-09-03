"""ACP must expose a configured context engine's toolset marker.

Regression test: an ACP agent with a plugin context engine received none of that
engine's tools, because ``acp_adapter.session._make_agent`` built
``enabled_toolsets`` directly instead of consulting the shared per-platform
resolver, so the ``context_engine`` marker was never present and the gate in
``agent_init`` skipped attaching the engine's schemas.

The seeding rule mirrors ``hermes_cli.tools_config._get_platform_tools``: add the
marker when a non-default engine is configured, withhold it for the built-in
compressor.
"""

import pytest

from acp_adapter.session import _expand_acp_enabled_toolsets


def _acp_seed(config):
    """The seeding rule as applied in ``_make_agent``."""
    seed = ["hermes-acp"]
    context_cfg = config.get("context") or {}
    if not isinstance(context_cfg, dict):
        context_cfg = {}
    engine = str(context_cfg.get("engine") or "compressor").strip().lower()
    if engine and engine != "compressor":
        seed.append("context_engine")
    return _expand_acp_enabled_toolsets(seed, mcp_server_names=[])


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"context": {"engine": "lcm"}}, True),
        ({"context": {"engine": "LCM"}}, True),           # case-insensitive
        ({"context": {"engine": "  lcm  "}}, True),       # whitespace tolerant
        ({"context": {"engine": "compressor"}}, False),   # built-in unchanged
        ({"context": {"engine": ""}}, False),
        ({"context": {}}, False),
        ({}, False),
        ({"context": "not-a-dict"}, False),               # malformed config is safe
    ],
)
def test_context_engine_marker_matches_resolver_semantics(config, expected):
    assert ("context_engine" in _acp_seed(config)) is expected


def test_marker_grants_no_tools_by_itself():
    """``context_engine`` is a marker toolset; it must not widen the tool surface."""
    from toolsets import resolve_toolset

    assert resolve_toolset("context_engine") == []
    assert not set(resolve_toolset("context_engine")) & set(resolve_toolset("hermes-acp"))


def test_mcp_server_toolsets_still_expand():
    """Seeding the marker must not disturb existing ``mcp-*`` expansion."""
    out = _expand_acp_enabled_toolsets(
        ["hermes-acp", "context_engine"], mcp_server_names=["example"]
    )
    assert "hermes-acp" in out
    assert "context_engine" in out
    assert "mcp-example" in out
