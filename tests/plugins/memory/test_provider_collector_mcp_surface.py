"""The memory provider collector's ctx forwards the MCP plugin surface (#152373).

A provider that reads from an MCP server used to build its own ``PluginContext`` out of private
core imports (flagged by ``hermes plugins validate`` rule 9). The collector now forwards
``call_mcp`` — with the per-server allowlist keyed to the PROVIDER's name, so the forwarded path
stays default-deny — and the ``get_portable_mcp_servers`` availability read. ``__getattr__``
strictness is untouched: unknown attributes still raise.
"""

from unittest.mock import MagicMock

import pytest

from plugins.memory import _ProviderCollector


def _patch_config(monkeypatch, entries: dict) -> None:
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config",
        lambda *a, **k: {"plugins": {"entries": entries}},
    )


def test_call_mcp_forwards_to_the_real_context_with_exact_arguments(monkeypatch):
    collector = _ProviderCollector(name="memtomem")
    forwarded = MagicMock()
    monkeypatch.setattr(collector, "_plugin_context", lambda: forwarded)

    collector.call_mcp("knowledge_rag", "query", {"q": "runbook"}, timeout=7)

    forwarded.call_mcp.assert_called_once_with(
        "knowledge_rag", "query", arguments={"q": "runbook"}, timeout=7)


def test_call_mcp_is_default_deny_under_the_providers_own_name(monkeypatch):
    """The forwarded call must not be ambient authority: the allowlist keys to the provider's
    name, and an unlisted server raises the standard PermissionError naming the config key."""
    _patch_config(monkeypatch, {"memtomem": {"mcp_allowlist": ["other-server"]}})
    collector = _ProviderCollector(name="memtomem")

    with pytest.raises(PermissionError, match=r"plugins\.entries\.memtomem\.mcp_allowlist"):
        collector.call_mcp("knowledge_rag", "query")


def test_get_portable_mcp_servers_forwards_for_the_availability_check(monkeypatch):
    collector = _ProviderCollector(name="memtomem")
    forwarded = MagicMock()
    forwarded.get_portable_mcp_servers.return_value = {"knowledge_rag": {"command": "x"}}
    monkeypatch.setattr(collector, "_plugin_context", lambda: forwarded)

    assert collector.get_portable_mcp_servers() == {"knowledge_rag": {"command": "x"}}


def test_unknown_attributes_still_raise_loudly():
    """The explicit forwards do not loosen __getattr__: a typo fails at the call site."""
    collector = _ProviderCollector(name="memtomem")
    with pytest.raises(AttributeError):
        collector.call_mcp_typo
