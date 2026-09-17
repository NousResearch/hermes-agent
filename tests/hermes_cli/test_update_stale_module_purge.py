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
    the submodule bindings of the checkout-owned packages too — only those, so any other package
    global a test leaks still shows up as pollution.
    """
    from hermes_cli.update_cmd_maint import _stale_purge_prefixes

    snapshot = dict(sys.modules)
    owned = _stale_purge_prefixes()
    bindings = {
        mod: {k: v for k, v in vars(mod).items() if isinstance(v, types.ModuleType)}
        for name, mod in snapshot.items()
        if mod is not None
        and name.split(".", 1)[0] in owned
        and "__path__" in vars(mod)  # packages only: they carry submodules
    }
    yield
    for name, mod in snapshot.items():
        sys.modules[name] = mod
    for name in list(sys.modules):
        if name not in snapshot:
            del sys.modules[name]
    for mod, attrs in bindings.items():
        current = vars(mod)
        for key in [k for k, v in current.items() if isinstance(v, types.ModuleType) and k not in attrs]:
            del current[key]
        current.update(attrs)


def _fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__stale_sentinel__ = True
    return mod


def _install_stale_main_dashboard(**attrs) -> types.ModuleType:
    """A pre-pull ``main_dashboard`` stand-in, bound the way ``hermes_cli.main``'s eager import
    leaves it: in ``sys.modules`` AND as an attribute of the protected ``hermes_cli`` package."""
    import hermes_cli

    stale = _fake_module("hermes_cli.main_dashboard")
    vars(stale).update(attrs)
    sys.modules["hermes_cli.main_dashboard"] = stale
    hermes_cli.main_dashboard = stale
    return stale


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


def test_receipt_write_survives_a_mixed_module_graph(tmp_path, monkeypatch, capsys):
    """#112465 / #112558: the activation run wrote NO receipt while the no-op run did. The
    receipt is written by the pre-pull interpreter after the purge; resolving the receipt dir
    through ``hermes_cli.config`` re-executed the pulled config.py against a stale top-level
    ``utils`` (``from utils import file_signature`` → ImportError) and the whole write was
    swallowed at debug level. The receipt path must not depend on any purgeable module, and a
    write failure must be visible."""
    import hermes_cli
    import hermes_cli.update_receipt as receipt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Pre-pull world: a `utils` without the symbols the pulled config.py imports, and the
    # purged (evicted + unbound) hermes_cli.config so any import of it re-executes source.
    real_utils = sys.modules.get("utils")
    sys.modules["utils"] = types.ModuleType("utils")
    evicted = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("hermes_cli.config")}
    stale_attr = vars(hermes_cli).pop("config", None)
    receipt._current = None
    try:
        receipt.begin_update_receipt()
        receipt.record_step("git_pull", True)
        path = receipt.finalize_update_receipt("partial")
    finally:
        receipt._current = None
        sys.modules.pop("utils", None)
        if real_utils is not None:
            sys.modules["utils"] = real_utils
        for k in [k for k in sys.modules if k.startswith("hermes_cli.config")]:
            del sys.modules[k]
        sys.modules.update(evicted)
        if stale_attr is not None:
            hermes_cli.config = stale_attr

    assert path is not None and path.is_file(), "receipt lost to the mixed sys.modules graph"
    assert path.parent == tmp_path / "logs" / "update_receipts"
    assert json.loads(path.read_text(encoding="utf-8"))["outcome"] == "partial"


def test_receipt_write_failure_is_visible(tmp_path, monkeypatch, capsys):
    """A begun-but-unwritten receipt is the run operators must post-mortem; the failure was
    logged at DEBUG only, indistinguishable from "no receipt expected" (#112465)."""
    import hermes_cli.update_receipt as receipt

    def _boom():
        raise OSError("disk says no")

    monkeypatch.setattr(receipt, "_receipt_dir", _boom)
    receipt._current = None
    try:
        receipt.begin_update_receipt()
        assert receipt.finalize_update_receipt("success") is None
    finally:
        receipt._current = None
    assert "receipt not written: disk says no" in capsys.readouterr().out


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
    """Field failure #112604: `hermes_cli.main` imports `main_dashboard` at CLI start, so the
    updater process holds it as an ATTRIBUTE of the (protected) `hermes_cli` package. Evicting
    only the sys.modules entry left `from hermes_cli import main_dashboard` handing the PRE-pull
    module to the dashboard cleanup, which then died on a symbol the pull had added
    (`AttributeError ... has no attribute '_loaded_launchd_backend_jobs'`).
    """
    stale = _install_stale_main_dashboard()

    cli_main._purge_stale_hermes_modules()

    assert not hasattr(stale, "_loaded_launchd_backend_jobs")
    from hermes_cli import main_dashboard as pulled

    assert pulled is not stale, "call-time import was handed the pre-pull module"
    assert getattr(pulled, "__stale_sentinel__", False) is False
    assert hasattr(pulled, "_loaded_launchd_backend_jobs")


def test_dashboard_cleanup_survives_a_pre_pull_main_dashboard():
    """End-to-end shape of #112604: the post-update dashboard cleanup runs after the purge, and
    `_kill_stale_dashboard_processes` resolves its helpers then. With the pre-pull
    `main_dashboard` still reachable, the cleanup aborted on
    `AttributeError: module 'hermes_cli.main_dashboard' has no attribute
    '_loaded_launchd_backend_jobs'` — after the code update had already succeeded.
    """
    # The pre-pull module DOES scan processes; it only lacks the launchd symbol the pull added,
    # so a cleanup handed this module reaches the launchd snapshot line and dies there.
    _install_stale_main_dashboard(_find_stale_dashboard_pids=lambda **_kw: [999999])

    cli_main._purge_stale_hermes_modules()

    # The pulled scanner reads the host's process table through `dashboard_procs`; the stale
    # stand-in never does. Stubbing the table keeps the test off real processes AND records
    # which module the cleanup resolved.
    from hermes_cli import dashboard_procs
    table_reads = []
    with patch.object(dashboard_procs, "_iter_process_table", lambda: table_reads.append(1) or []):
        result = dashboard_procs._kill_stale_dashboard_processes("regression")

    assert isinstance(result, dict), "cleanup aborted instead of returning its result"
    assert table_reads, "cleanup was handed the pre-pull module instead of the pulled one"
