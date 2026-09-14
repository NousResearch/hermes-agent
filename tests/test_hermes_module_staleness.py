"""Tests for ``hermes_module_staleness`` — the shared stale-module purge/heal.

Field failure (2026-09-14T02:52:31Z gateway ImportError): an in-place checkout pull added
``gateway.session.profile_from_session_key_namespace`` and ``gateway.run``'s import of it while a
long-lived desktop ``hermes serve`` process cached ``gateway.session`` from before. The TUI's lazy
``from gateway.run import GatewayRunner`` then raised on every turn.

The updater already purged for its own process; these tests pin the extraction of that purge into
``hermes_module_staleness`` plus the new :func:`import_symbol` retry used by long-lived processes.
"""

from __future__ import annotations

import sys
import types

import pytest

from hermes_module_staleness import (
    PURGE_PROTECTED,
    PURGE_PROTECTED_PREFIX,
    import_symbol,
    purge_stale_modules,
)


def _fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__stale_sentinel__ = True
    return mod


def test_purge_evicts_hermes_prefixed_modules():
    registry = {
        name: _fake_module(name)
        for name in (
            "hermes_cli.cli_output",
            "gateway.status",
            "tools.ansi_strip",
            "tui_gateway.server",
            "agent.memory_store",
        )
    }
    expected = set(registry)
    purged = purge_stale_modules(modules=registry)
    assert set(purged) == expected
    assert registry == {}


def test_purge_protects_executing_and_updater_modules():
    registry = {
        name: _fake_module(name)
        for name in ("hermes_cli", "hermes_cli.main", "hermes_cli.update_receipt", "gateway.session")
    }
    assert "hermes_cli.update_receipt".startswith(PURGE_PROTECTED_PREFIX)
    purged = purge_stale_modules(modules=registry)
    assert purged == ["gateway.session"]
    assert set(registry) == {"hermes_cli", "hermes_cli.main", "hermes_cli.update_receipt"}


def test_purge_leaves_prefix_lookalikes_alone():
    """``gatewayd``/``toolshed`` start with a prefix string but are not the package."""
    registry = {name: _fake_module(name) for name in ("gatewayd", "toolshed", "agents_external")}
    assert purge_stale_modules(modules=registry) == []
    assert set(registry) == {"gatewayd", "toolshed", "agents_external"}


def test_purge_never_raises_on_none_entries():
    registry = {"gateway.ghost": None}
    assert purge_stale_modules(modules=registry) == []


def test_import_symbol_heals_stale_cache(tmp_path, monkeypatch):
    """The pre-pull cached module lacks the symbol; the on-disk source has it."""
    (tmp_path / "stale_fixture_pkg.py").write_text("NEW_SYMBOL = 'fresh'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    stale = types.ModuleType("stale_fixture_pkg")
    monkeypatch.setitem(sys.modules, "stale_fixture_pkg", stale)
    assert not hasattr(stale, "NEW_SYMBOL")

    assert import_symbol("stale_fixture_pkg", "NEW_SYMBOL", retry_prefixes=("stale_fixture_pkg",)) == "fresh"


def test_import_symbol_heals_nested_stale_dependency(tmp_path, monkeypatch):
    """Field-failure variant (2026-09-06): the stale module is one the imported module ITSELF
    imports — ``from gateway.run import ...`` died on ``agent.session_activity`` missing
    ``format_iteration_progress``. The implicated sibling must be rebuilt, not just the target."""
    (tmp_path / "stale_inner_fixture.py").write_text("NEEDED = 'fresh'\n", encoding="utf-8")
    (tmp_path / "stale_outer_fixture.py").write_text(
        "from stale_inner_fixture import NEEDED\nOK = 'OK'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    stale_inner = types.ModuleType("stale_inner_fixture")  # cached pre-update: no NEEDED
    monkeypatch.setitem(sys.modules, "stale_inner_fixture", stale_inner)
    monkeypatch.delitem(sys.modules, "stale_outer_fixture", raising=False)

    healed = import_symbol(
        "stale_outer_fixture", "OK", retry_prefixes=("stale_outer_fixture", "stale_inner_fixture"))

    assert healed == "OK"
    assert sys.modules["stale_inner_fixture"] is not stale_inner


def test_import_symbol_evicts_only_implicated_modules(tmp_path, monkeypatch):
    """Surgical eviction: unrelated modules under the same retry prefix must survive."""
    (tmp_path / "stale_outer_fixture.py").write_text(
        "from stale_inner_fixture import NEEDED\nOK = 'OK'\n", encoding="utf-8")
    (tmp_path / "stale_inner_fixture.py").write_text("NEEDED = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "stale_inner_fixture", types.ModuleType("stale_inner_fixture"))
    bystander = types.ModuleType("stale_bystander_fixture")
    monkeypatch.setitem(sys.modules, "stale_bystander_fixture", bystander)

    assert import_symbol(
        "stale_outer_fixture", "OK",
        retry_prefixes=("stale_outer_fixture", "stale_inner_fixture", "stale_bystander_fixture")) == "OK"
    assert sys.modules.get("stale_bystander_fixture") is bystander


def test_import_symbol_returns_cached_symbol_without_purging(monkeypatch):
    import html

    monkeypatch.setattr(
        "hermes_module_staleness.purge_stale_modules",
        lambda *a, **k: pytest.fail("purge must not run when the import already resolves"),
    )
    assert import_symbol("html", "escape") is html.escape


def test_import_symbol_reraises_genuinely_missing_module(monkeypatch):
    # retry_prefixes scoped away from real packages so the retry purge is a no-op in the runner.
    monkeypatch.setattr(
        "hermes_module_staleness.purge_stale_modules",
        lambda *a, **k: pytest.fail("unrelated ImportError must not purge Hermes packages"),
    )
    with pytest.raises(ImportError):
        import_symbol("definitely_not_a_hermes_module_xyz", "anything", retry_prefixes=("nonexistent_xyz",))
