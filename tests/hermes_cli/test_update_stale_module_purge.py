"""Tests for _purge_stale_hermes_modules — the class fix for stale
sys.modules breaking the gateway auto-restart after `hermes update`.

Field failure (2026-08-20, Teknium's Linux box): `hermes update` pulled a
checkout where hermes_cli/gateway.py newly imports `line_input` from
hermes_cli.cli_output, but the updater process had cli_output cached from
before that symbol existed. The function-level `from hermes_cli.gateway
import ...` in the restart phase raised ImportError, the whole phase
aborted, and the running gateway kept serving pre-update code.

The old mitigation (_UPDATE_RUNTIME_RELOAD_MODULES) reloaded 3 hardcoded
modules — re-fixed per symptom. The purge evicts EVERY cached module whose
top-level name is a ``.py`` file or package in the checkout root (minus
``tests``) so later imports rebuild a self-consistent module graph from the
updated checkout.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    """Snapshot & restore sys.modules around each test.

    The purge under test evicts real Hermes modules from the cache; later
    tests in the same process may hold references to the evicted module
    objects (e.g. `patch.object` targets), so put the originals back.

    The eviction also drops the stale submodule attribute a purged module left on its PARENT
    package (see `_evict_module`), which `sys.modules` alone does not cover: after the test,
    `hermes_cli.X` could hold a freshly imported copy while the cache holds the original, and a
    later `patch("hermes_cli.X.fn")` would miss the module the code under test resolves. Restore
    the checkout-owned package namespaces too.
    """
    from hermes_cli.update_cmd_maint import _stale_purge_prefixes

    snapshot = dict(sys.modules)
    owned = _stale_purge_prefixes()
    namespaces = {
        name: (mod, dict(vars(mod)))
        for name, mod in snapshot.items()
        if mod is not None
        and name.split(".", 1)[0] in owned
        and getattr(mod, "__path__", None) is not None  # packages only: they carry submodules
    }
    yield
    for name, mod in snapshot.items():
        sys.modules[name] = mod
    for name in list(sys.modules):
        if name not in snapshot:
            del sys.modules[name]
    for name, (mod, attrs) in namespaces.items():
        if sys.modules.get(name) is not mod:
            continue
        current = vars(mod)
        for key in [key for key in current if key not in attrs]:
            del current[key]
        current.update(attrs)


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


def test_purge_preserves_active_update_receipt(tmp_path, monkeypatch):
    """A receipt begun before the post-pull purge must still be finalizable."""
    import hermes_cli.update_receipt as receipt

    receipt_dir = tmp_path / "update_receipts"
    monkeypatch.setattr(receipt, "_receipt_dir", lambda: receipt_dir)
    receipt._current = None
    post_purge_receipt = receipt
    try:
        receipt.begin_update_receipt()
        receipt.record_step("git_pull", True, "updated checkout")

        cli_main._purge_stale_hermes_modules()
        post_purge_receipt = importlib.import_module("hermes_cli.update_receipt")
        path = post_purge_receipt.finalize_update_receipt("success")

        assert path is not None and path.is_file()
        latest = json.loads((receipt_dir / "latest.json").read_text(encoding="utf-8"))
        assert latest["outcome"] == "success"
        assert latest["steps"][0]["name"] == "git_pull"
    finally:
        receipt._current = None
        post_purge_receipt._current = None


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


def test_purge_keeps_plan_record_class_identity():
    # The pre-update plan is built BEFORE the purge; reconciliation after it filters with
    # ``isinstance(r, RuntimeRecord)``. An evicted ``update_inventory`` yields a fresh class,
    # every record fails the check, and the plan-vs-execution report goes silently empty.
    from hermes_cli.update_inventory import RuntimeRecord as before

    cli_main._purge_stale_hermes_modules()
    from hermes_cli.update_inventory import RuntimeRecord as after
    assert after is before


def test_stale_top_level_utils_scenario_end_to_end():
    """The 2026-09-12 field failure: `hermes update` from a pre-`base_url_origin`
    checkout kept the old top-level `utils` cached, and the restart phase's import of
    `hermes_cli.gateway` died on `from utils import base_url_origin`."""
    stale = types.ModuleType("utils")
    real = sys.modules.get("utils")
    sys.modules["utils"] = stale
    try:
        try:
            from utils import base_url_origin  # noqa: F401
            raised = False
        except ImportError:
            raised = True
        assert raised, "precondition: stale utils must lack base_url_origin"

        cli_main._purge_stale_hermes_modules()

        from utils import base_url_origin  # noqa: F401
    finally:
        sys.modules.pop("utils", None)
        if real is not None:
            sys.modules["utils"] = real


def test_purge_protects_hermes_logging():
    # A second copy of hermes_logging starts a second QueueListener over the same log
    # files while the first keeps running: its listener/handler state is module-global.
    real = sys.modules.get("hermes_logging")
    sentinel = _fake_module("hermes_logging")
    sys.modules["hermes_logging"] = sentinel
    try:
        cli_main._purge_stale_hermes_modules()
        assert sys.modules.get("hermes_logging") is sentinel
    finally:
        sys.modules.pop("hermes_logging", None)
        if real is not None:
            sys.modules["hermes_logging"] = real


def test_purge_drops_stale_package_attribute_so_from_import_rereads_source():
    """2026-09-16 field failure (#111689): `hermes_cli.main` imports `main_dashboard` at CLI
    start, so the updater process holds it as an ATTRIBUTE of the (protected) `hermes_cli`
    package. Evicting only the sys.modules entry left `from hermes_cli import main_dashboard`
    handing the PRE-pull module to the dashboard cleanup, which then died on a symbol the pull
    had added (`AttributeError ... has no attribute '_loaded_launchd_backend_jobs'`).
    """
    import hermes_cli

    name = "hermes_cli.main_dashboard"
    real = sys.modules.get(name)
    stale = types.ModuleType(name)  # pre-pull stand-in: lacks the symbol the cleanup needs
    stale.__file__ = getattr(real, "__file__", "<stale>")
    stale.__stale_sentinel__ = True
    sys.modules[name] = stale
    setattr(hermes_cli, "main_dashboard", stale)
    try:
        cli_main._purge_stale_hermes_modules()

        assert not hasattr(stale, "_loaded_launchd_backend_jobs")
        from hermes_cli import main_dashboard as pulled

        assert pulled is not stale, "call-time import was handed the pre-pull module"
        assert getattr(pulled, "__stale_sentinel__", False) is False
        assert hasattr(pulled, "_loaded_launchd_backend_jobs")
    finally:
        sys.modules.pop(name, None)
        if real is not None:
            sys.modules[name] = real
            setattr(hermes_cli, "main_dashboard", real)


def test_dashboard_cleanup_survives_a_pre_pull_main_dashboard():
    """End-to-end shape of the 2026-09-16 field failure: the post-update dashboard cleanup runs
    after the purge, and `_kill_stale_dashboard_processes` resolves its helpers then. With the
    pre-pull `main_dashboard` still reachable, the cleanup aborted on
    `AttributeError: module 'hermes_cli.main_dashboard' has no attribute
    '_loaded_launchd_backend_jobs'` — after the code update had already succeeded.
    """
    import hermes_cli
    from hermes_cli import dashboard_procs

    name = "hermes_cli.main_dashboard"
    real = sys.modules.get(name)
    stale = types.ModuleType(name)
    stale.__file__ = getattr(real, "__file__", "<stale>")
    stale.__stale_sentinel__ = True
    # The pre-pull module DOES scan processes; it only lacks the launchd symbol the pull added,
    # so the cleanup reaches the launchd snapshot line and dies there (the reported traceback).
    stale._find_stale_dashboard_pids = lambda **_kw: [999999]
    sys.modules[name] = stale
    setattr(hermes_cli, "main_dashboard", stale)
    try:
        cli_main._purge_stale_hermes_modules()

        # The scan/kill pass must not touch real processes: the fresh module's pid scan is host
        # dependent (a dashboard may or may not be running), so only the kill step is stubbed.
        with patch.object(dashboard_procs, "_kill_pids_posix", side_effect=lambda pids, k, f: None):
            result = dashboard_procs._kill_stale_dashboard_processes("regression")

        assert isinstance(result, dict), "cleanup aborted instead of returning its result"
        assert hasattr(sys.modules["hermes_cli.main_dashboard"], "_loaded_launchd_backend_jobs"), \
            "cleanup resolved the pre-pull module"
    finally:
        sys.modules.pop(name, None)
        if real is not None:
            sys.modules[name] = real
            setattr(hermes_cli, "main_dashboard", real)
