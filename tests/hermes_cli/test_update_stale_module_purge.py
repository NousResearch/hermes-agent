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

import importlib
import json
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


def test_purge_prefixes_cover_checkout_top_level_modules():
    # The static tuple listed 5 packages and no top-level module, so `utils`,
    # `hermes_constants` and friends stayed cached through every update.
    prefixes = update_cmd._stale_purge_prefixes()
    for name in ("utils", "hermes_constants", "hermes_bootstrap", "plugins", "providers"):
        assert name in prefixes, f"{name} is not covered by the purge"
    for name in ("hermes_cli", "gateway", "tools", "tui_gateway", "agent"):
        assert name in prefixes


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


def test_purge_spares_the_tests_package():
    # pytest resolves fixtures through the identity of already-imported test modules.
    sentinel = _fake_module("tests.hermes_cli._purge_probe")
    sys.modules["tests.hermes_cli._purge_probe"] = sentinel
    try:
        cli_main._purge_stale_hermes_modules()
        assert sys.modules.get("tests.hermes_cli._purge_probe") is sentinel
    finally:
        sys.modules.pop("tests.hermes_cli._purge_probe", None)


def test_purge_prefixes_follow_project_root(tmp_path, monkeypatch):
    # The scan must read main.PROJECT_ROOT, not this module's own __file__: tests and the
    # installer both relocate the checkout, and a divergent root silently purges nothing.
    (tmp_path / "zzz_probe.py").write_text("", encoding="utf-8")
    (tmp_path / "zzz_pkg").mkdir()
    (tmp_path / "zzz_pkg" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)

    prefixes = update_cmd._stale_purge_prefixes()
    assert {"zzz_probe", "zzz_pkg"} <= prefixes
    assert set(update_cmd._STALE_PURGE_PREFIXES) <= prefixes


def test_purge_prefixes_fall_back_when_root_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path / "does-not-exist")
    assert update_cmd._stale_purge_prefixes() == frozenset(update_cmd._STALE_PURGE_PREFIXES)
