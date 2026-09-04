"""Tests for _purge_stale_hermes_modules — the class fix for stale
sys.modules breaking the gateway auto-restart after `hermes update`.

Field failure (2026-08-20, Teknium's Linux box): `hermes update` pulled a
checkout where hermes_cli/gateway.py newly imports `line_input` from
hermes_cli.cli_output, but the updater process had cli_output cached from
before that symbol existed. The function-level `from hermes_cli.gateway
import ...` in the restart phase raised ImportError, the whole phase
aborted, and the running gateway kept serving pre-update code.

The old mitigation (_UPDATE_RUNTIME_RELOAD_MODULES) reloaded 3 hardcoded
modules — re-fixed per symptom. The purge evicts EVERY cached module under
the Hermes package prefixes so later imports rebuild a self-consistent
module graph from the updated checkout.
"""

from __future__ import annotations

import sys
import types

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    """Snapshot & restore sys.modules around each test.

    The purge under test evicts real Hermes modules from the cache; later
    tests in the same process may hold references to the evicted module
    objects (e.g. `patch.object` targets), so put the originals back.
    """
    snapshot = dict(sys.modules)
    yield
    for name, mod in snapshot.items():
        sys.modules[name] = mod
    for name in list(sys.modules):
        if name not in snapshot:
            del sys.modules[name]


def _fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__stale_sentinel__ = True
    return mod


def test_purge_evicts_hermes_prefixed_modules():
    victims = [
        "hermes_cli.cli_output",
        "hermes_cli.gateway",
        "gateway.status",
        "tools.ansi_strip",
        "tui_gateway.server",
        "agent.memory_store",
    ]
    added = []
    for name in victims:
        if name not in sys.modules:
            sys.modules[name] = _fake_module(name)
            added.append(name)
    try:
        cli_main._purge_stale_hermes_modules()
        for name in victims:
            mod = sys.modules.get(name)
            assert mod is None or not getattr(mod, "__stale_sentinel__", False), (
                f"{name} survived the purge"
            )
    finally:
        for name in added:
            sys.modules.pop(name, None)


def test_purge_protects_executing_modules():
    # The updater's own modules must survive — they're running this code.
    cli_main._purge_stale_hermes_modules()
    assert sys.modules.get("hermes_cli.update_cmd") is update_cmd
    assert sys.modules.get("hermes_cli.main") is cli_main
    assert "hermes_cli" in sys.modules


def test_purge_leaves_prefix_lookalikes_alone():
    # `gateway_foo` starts with the string prefix "gateway" but is NOT the
    # gateway package — the root-segment check must spare it.
    lookalikes = ["gatewayd", "toolshed", "agents_external"]
    added = []
    for name in lookalikes:
        if name not in sys.modules:
            sys.modules[name] = _fake_module(name)
            added.append(name)
    try:
        cli_main._purge_stale_hermes_modules()
        for name in lookalikes:
            assert name in sys.modules, f"{name} was wrongly purged"
    finally:
        for name in added:
            sys.modules.pop(name, None)


def test_purge_never_raises_on_weird_sys_modules():
    # Entries with None values (import machinery quirk) must not break it.
    sys.modules["hermes_cli._purge_test_none"] = None  # type: ignore[assignment]
    try:
        cli_main._purge_stale_hermes_modules()
    finally:
        sys.modules.pop("hermes_cli._purge_test_none", None)


def test_stale_symbol_scenario_end_to_end():
    """Reproduce the field failure shape: a cached module missing a symbol
    that freshly-imported code needs — purge, then re-import resolves it."""
    name = "hermes_cli.cli_output"
    real = sys.modules.get(name)
    # Install a stale stand-in WITHOUT line_input (pre-d0132b582 world).
    stale = types.ModuleType(name)
    sys.modules[name] = stale
    try:
        # The failure mode: importing the symbol from the stale cache dies.
        try:
            from hermes_cli.cli_output import line_input  # noqa: F401
            raised = False
        except ImportError:
            raised = True
        assert raised, "precondition: stale module must lack line_input"

        cli_main._purge_stale_hermes_modules()

        # Post-purge, the import resolves against real on-disk source.
        from hermes_cli.cli_output import line_input  # noqa: F401
    finally:
        sys.modules.pop(name, None)
        if real is not None:
            sys.modules[name] = real


# --- the pytest-side antidote to the purge -------------------------------
#
# The purge is correct in production (the interpreter is about to exec the new
# checkout) and poisonous inside pytest, where the session continues. The
# antidote is `conftest._restore_purged_hermes_modules`; these tests pin its
# CONTRACT, because the first version of it restored `sys.modules` only and so
# rebuilt the same split identity one level up — a later test's
# `monkeypatch.setattr("pkg.sub.thing", ...)` then patched a copy the code under
# test never reached. Reproduced in this suite with `hermes_cli.config` after
# `test_state_db_guard` deletes and re-imports it.


def _install_fake_package(monkeypatch):
    """A parent package with one submodule, wired the way import wires it."""
    pkg = types.ModuleType("hermes_cli_fakepkg")
    sub = types.ModuleType("hermes_cli_fakepkg.sub")
    pkg.sub = sub
    monkeypatch.setitem(sys.modules, "hermes_cli_fakepkg", pkg)
    monkeypatch.setitem(sys.modules, "hermes_cli_fakepkg.sub", sub)
    return pkg, sub


def test_restore_snapshot_rebinds_the_parent_attribute(monkeypatch):
    """Restoring must fix BOTH names a submodule answers to."""
    from tests.hermes_cli.conftest import _restore_module_snapshot

    pkg, sub = _install_fake_package(monkeypatch)
    snapshot = {"hermes_cli_fakepkg.sub": sub}

    # What a partial purge + re-import leaves behind, and the shape that
    # actually bit: only the SUBMODULE was evicted, so the import system
    # rebound the fresh copy onto the SAME, still-cached parent package.
    new_sub = types.ModuleType("hermes_cli_fakepkg.sub")
    sys.modules["hermes_cli_fakepkg.sub"] = new_sub
    pkg.sub = new_sub

    _restore_module_snapshot(snapshot)

    assert sys.modules["hermes_cli_fakepkg.sub"] is sub
    # The regression: without rebinding, this is still `new_sub`, so
    # `from hermes_cli_fakepkg import sub` hands out the copy the test built.
    assert pkg.sub is sub, "parent package still points at the post-purge copy"


def test_restore_snapshot_leaves_untouched_modules_alone(monkeypatch):
    """Identity already correct ⇒ no write, so a legitimately shared object
    is never clobbered and a read-only parent never raises."""
    pkg, sub = _install_fake_package(monkeypatch)

    class Frozen(types.ModuleType):
        def __setattr__(self, name, value):  # pragma: no cover - guard only
            raise AttributeError("read-only parent")

    frozen = Frozen("hermes_cli_fakepkg")
    monkeypatch.setitem(sys.modules, "hermes_cli_fakepkg", frozen)

    from tests.hermes_cli.conftest import _restore_module_snapshot

    # `sub` is unchanged in sys.modules, so restore must not touch the parent.
    _restore_module_snapshot({"hermes_cli_fakepkg.sub": sub})
    assert sys.modules["hermes_cli_fakepkg.sub"] is sub


def test_restore_snapshot_survives_a_parent_that_refuses_setattr(monkeypatch):
    """A parent that rejects attribute writes must not fail the whole run:
    the dict entry is the load-bearing half, the attribute is best-effort."""
    from tests.hermes_cli.conftest import _restore_module_snapshot

    class Frozen(types.ModuleType):
        def __setattr__(self, name, value):
            raise AttributeError("read-only parent")

    frozen = Frozen("hermes_cli_fakepkg")
    sub = types.ModuleType("hermes_cli_fakepkg.sub")
    monkeypatch.setitem(sys.modules, "hermes_cli_fakepkg", frozen)
    monkeypatch.setitem(sys.modules, "hermes_cli_fakepkg.sub",
                        types.ModuleType("hermes_cli_fakepkg.sub"))

    _restore_module_snapshot({"hermes_cli_fakepkg.sub": sub})

    assert sys.modules["hermes_cli_fakepkg.sub"] is sub
