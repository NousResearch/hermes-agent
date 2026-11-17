"""Profiles a plugin runs elsewhere: the host never runs them.

A host-side plugin that runs a profile's backend outside this host (for example in a sandbox)
marks it under ``<root>/profiles_run_elsewhere/<profile>.json``. These tests drive the readers
the host services go through: ``profiles_to_serve`` and its callers, ``ensure_gateway_runtime``,
the Kanban dispatcher, the Desktop cron ticker, the dashboard's profile-scoped requests, and the
profile listing inside a backend that is denied the profiles folder.
"""
from __future__ import annotations

import errno
import json
import os
import threading
from pathlib import Path

import pytest

from hermes_cli import profiles_run_elsewhere as pre


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root with the default profile and two named profiles, ``ops`` and ``dev``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    for name in ("ops", "dev"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _names(pairs):
    return [name for name, _home in pairs]


def _fail_listing_marker_folder(monkeypatch, error: OSError):
    real_listdir = os.listdir

    def listdir(path):
        if Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            raise error
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)


# ── marker API ─────────────────────────────────────────────────────────────────────────────────


def test_mark_writes_the_documented_file_and_clear_removes_it(root):
    entry = pre.mark_profile_run_elsewhere("Ops", "sandbox-plugin")
    path = root / "profiles_run_elsewhere" / "ops.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["plugin"] == "sandbox-plugin" and data["since"].endswith("Z")
    assert (entry.profile, entry.plugin, entry.path) == ("ops", "sandbox-plugin", path)
    assert pre.run_elsewhere("ops").plugin == "sandbox-plugin"
    assert pre.run_elsewhere("dev") is None

    # Marking again for the same plugin keeps ``since``.
    assert pre.mark_profile_run_elsewhere("ops", "sandbox-plugin").since == data["since"]

    assert pre.clear_profile_run_elsewhere("ops", "sandbox-plugin") is True
    assert not path.exists() and pre.run_elsewhere("ops") is None
    assert pre.clear_profile_run_elsewhere("ops", "sandbox-plugin") is False


def test_one_plugin_cannot_take_over_or_clear_another_plugins_marker(root):
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    with pytest.raises(ValueError, match="sandbox-plugin"):
        pre.mark_profile_run_elsewhere("ops", "other")
    with pytest.raises(ValueError, match="sandbox-plugin"):
        pre.clear_profile_run_elsewhere("ops", "other")
    assert pre.run_elsewhere("ops").plugin == "sandbox-plugin"


@pytest.mark.parametrize("profile, plugin", [("../evil", "sandbox-plugin"), ("ops", ""), ("ops", None)])
def test_mark_rejects_bad_profile_ids_and_plugin_names(root, profile, plugin):
    with pytest.raises(ValueError):
        pre.mark_profile_run_elsewhere(profile, plugin)
    assert not (root / "profiles_run_elsewhere").exists()


def test_an_unreadable_marker_still_marks_the_profile(root):
    folder = root / "profiles_run_elsewhere"
    folder.mkdir()
    (folder / "ops.json").write_text("{not json", encoding="utf-8")
    entry = pre.run_elsewhere("ops")
    assert entry is not None and entry.plugin is None
    assert "'ops'" in entry.message() and str(folder / "ops.json") in entry.message()


def test_temp_files_and_other_names_are_not_markers(root):
    folder = root / "profiles_run_elsewhere"
    folder.mkdir()
    (folder / ".ops_abc123.tmp").write_text("{}", encoding="utf-8")
    (folder / "dev.json.bak").write_text("{}", encoding="utf-8")
    (folder / "Not An Id.json").write_text("{}", encoding="utf-8")
    assert pre.run_elsewhere("ops") is None and pre.run_elsewhere("dev") is None


def test_a_home_resolves_to_its_own_profile_and_root(root):
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert pre.run_elsewhere_for_home(root / "profiles" / "ops").profile == "ops"
    assert pre.run_elsewhere_for_home(root / "profiles" / "dev") is None
    assert pre.run_elsewhere_for_home(root) is None
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin")
    assert pre.run_elsewhere_for_home(root).profile == "default"


# ── profiles_to_serve and its callers ──────────────────────────────────────────────────────────


def test_profiles_to_serve_skips_marked_profiles_unless_asked(root):
    from hermes_cli.profiles import profiles_to_serve

    assert _names(profiles_to_serve(True)) == ["default", "dev", "ops"]
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert _names(profiles_to_serve(True)) == ["default", "dev"]
    assert _names(profiles_to_serve(True, include_run_elsewhere=True)) == ["default", "dev", "ops"]
    # The standalone/parked opt-ins do not bring a marked profile back.
    assert _names(profiles_to_serve(True, include_standalone=True, include_parked=True)) == ["default", "dev"]

    pre.mark_profile_run_elsewhere("default", "sandbox-plugin")
    assert _names(profiles_to_serve(True)) == ["dev"]

    pre.clear_profile_run_elsewhere("ops", "sandbox-plugin")
    pre.clear_profile_run_elsewhere("default", "sandbox-plugin")
    assert _names(profiles_to_serve(True)) == ["default", "dev", "ops"]


def test_an_unreadable_marker_folder_serves_nothing(root, monkeypatch):
    """Any error listing the folder other than "missing" or "denied" fails closed."""
    from hermes_cli.profiles import profiles_to_serve

    (root / "profiles_run_elsewhere").mkdir()
    _fail_listing_marker_folder(monkeypatch, OSError(errno.EIO, "I/O error"))
    assert profiles_to_serve(True) == []
    entry = pre.run_elsewhere("dev")
    assert entry is not None and entry.folder_error and "cannot tell" in entry.message()


def test_a_process_denied_the_marker_folder_refuses_nothing(root, monkeypatch):
    """The plugin denies the folder to the backend it runs elsewhere; that backend keeps running."""
    from hermes_cli.profiles import profiles_to_serve

    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    _fail_listing_marker_folder(monkeypatch, PermissionError(errno.EACCES, "Access is denied"))
    assert pre.run_elsewhere("ops") is None
    assert _names(profiles_to_serve(True)) == ["default", "dev", "ops"]


def test_installed_profile_enumerators_keep_marked_profiles(root, monkeypatch):
    """Callers that look for live processes or count servable homes still see a marked profile."""
    from gateway import host_attach
    from tui_gateway import launch_profile_policy

    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    probed = []
    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home",
                        lambda home: probed.append(Path(home).name) or None)
    list(host_attach._coexisting_gateways(None))
    assert "ops" in probed

    servable = launch_profile_policy._servable_profile_homes()
    assert (root / "profiles" / "ops").resolve() in servable


def test_marked_profiles_keep_a_multi_profile_host_multiplexing(root, monkeypatch):
    """Running elsewhere is reversible like parking: a host whose other profiles are all marked
    still counts them, so it keeps its multiplexer (and the rescan that serves them again)."""
    from hermes_cli import gateway_migrate, gateway_multiplex_mode as mode

    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    pre.mark_profile_run_elsewhere("dev", "sandbox-plugin")
    monkeypatch.setattr(mode, "_standalone_launcher", lambda: False)
    monkeypatch.setattr(gateway_migrate, "_host_supports_migration", lambda: "host check reached")
    assert mode.implicit_multiplex_blocker() == "host check reached"


def test_worktree_sweep_reads_the_pins_of_a_marked_profile(root, tmp_path):
    """The sweep's view of resumable-subagent pins only reads, so a marked profile's subagents keep
    their trees; the gateway-shutdown stamp writes, so it leaves a marked profile's store alone."""
    import sqlite3
    import time

    from tools import delegate_tool_resumable as resumable

    def record(db, sid, document, *, state, pinned):
        assert resumable._write(db, lambda ledger: resumable._upsert(
            ledger, sid, document, state=state, pinned=pinned, expires_at=time.time() + 3600, expected=None))

    trees = {}
    for name in ("ops", "dev"):
        db = root / "profiles" / name / "state.db"
        trees[name] = tmp_path / "repo" / ".worktrees" / f"subagent-{name}"
        record(db, f"sa-{name}", {"cwd": str(trees[name]), "pin_paths": [str(trees[name])]},
               state="resumable", pinned=True)
        record(db, f"run-{name}", {"owner_pid": os.getpid()}, state="running", pinned=False)
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")

    pins = resumable.pinned_worktrees()
    assert resumable.pin_reason(str(trees["ops"]), pins) and resumable.pin_reason(str(trees["dev"]), pins)

    assert resumable.note_gateway_shutdown() == 1  # dev's running record; ops's store is not written
    causes = {}
    for name in ("ops", "dev"):
        conn = sqlite3.connect(root / "profiles" / name / "state.db")  # health: allow HX015 -- temp-home store
        try:
            causes[name] = conn.execute("SELECT stop_cause FROM resumable_subagents WHERE subagent_id=?",
                                        (f"run-{name}",)).fetchone()[0]
        finally:
            conn.close()
    assert causes == {"ops": None, "dev": resumable.GATEWAY_SHUTDOWN}


def test_dashboard_refuses_requests_scoped_to_a_marked_profile(root):
    from fastapi import HTTPException
    from hermes_cli import web_server_cron
    from hermes_cli.web_routers import profiles as profiles_router

    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert "ops" not in _names(profiles_router._profile_targets("test"))
    assert "ops" not in [p["name"] for p in web_server_cron._cron_profile_dicts()]
    with pytest.raises(HTTPException) as refused:
        web_server_cron._cron_profile_home("ops")
    assert refused.value.status_code == 409
    assert "'ops'" in refused.value.detail and "sandbox-plugin" in refused.value.detail
    assert web_server_cron._cron_profile_home("dev")[0] == "dev"


def test_desktop_cron_ticker_never_ticks_a_marked_profile(root, monkeypatch):
    from cron import scheduler_provider as sp
    from hermes_cli import web_server

    monkeypatch.setattr("hermes_cli.profiles._check_gateway_running", lambda home: False)
    monkeypatch.setattr("hermes_cli.profiles._served_by_running_multiplexer", lambda name: False)
    monkeypatch.setattr("hermes_logging.enable_profile_log_routing", lambda homes: None)
    captured = {}

    class _Provider:
        name = "fake"

        def start(self, stop_event, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: _Provider())
    monkeypatch.setattr(sp, "InProcessCronScheduler", _Provider)
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    assert _names(captured["profile_homes"]()) == ["default", "dev"]
    gate = captured["profile_gate"]
    assert gate("dev", root / "profiles" / "dev") is True
    # The gate holds on its own too: the single-store fallback ticks through it.
    assert gate("ops", root / "profiles" / "ops") is False


# ── profile listing inside a backend denied the profiles folder ────────────────────────────────


def _deny_profiles_folder(monkeypatch, profiles_root: Path):
    real_iterdir = Path.iterdir

    def iterdir(self):
        if self == profiles_root:
            raise PermissionError(errno.EACCES, "Access is denied", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)


def test_denied_profiles_folder_still_raises_for_a_default_profile_process(root, monkeypatch):
    """Only a process that runs a named profile of this root is the backend a plugin denied the
    folder. A default-profile process gets the error, as it did before the fallback existed, so a
    caller that fails closed on a profiles folder it cannot list still does: the host's eager
    multi-profile activation cannot prove the host is single-profile, and arms the credential guard."""
    from agent import secret_scope
    from hermes_cli import profiles
    from tui_gateway import launch_profile_policy

    _deny_profiles_folder(monkeypatch, profiles._get_profiles_root())
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    assert launch_profile_policy.activate_multi_profile_hosting_eagerly() is True
    assert secret_scope.is_multiplex_active()

    with pytest.raises(PermissionError):
        profiles._iter_named_profile_dirs()
    with pytest.raises(PermissionError):
        profiles.profiles_to_serve(True)
    assert profiles.list_profile_names() == ["default"]  # the name-only listing always tolerated it


def test_denied_profiles_folder_still_lists_the_backends_own_profile(root, monkeypatch):
    """A named profile's backend sees its own profile, even with the tombstone folder denied too."""
    import hermes_constants
    from agent import secret_scope
    from hermes_cli import profiles
    from tui_gateway import launch_profile_policy

    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    _deny_profiles_folder(monkeypatch, profiles._get_profiles_root())

    def tombstone_denied(home):
        raise PermissionError(errno.EACCES, "Access is denied")

    monkeypatch.setattr(hermes_constants, "named_profile_is_deleted", tombstone_denied)
    assert [p.name for p in profiles._iter_named_profile_dirs()] == ["ops"]
    assert [p.name for p in profiles.list_profiles(lazy_skill_count=True)] == ["default", "ops"]

    # Two homes (the root's default and its own), so that backend arms the credential guard too.
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setattr("hermes_logging.enable_profile_log_routing", lambda homes: None)
    assert launch_profile_policy.activate_multi_profile_hosting_eagerly() is True


# ── ensure_gateway_runtime ─────────────────────────────────────────────────────────────────────


def test_ensure_refuses_a_marked_profile_before_discovery_or_start(root, monkeypatch):
    from hermes_cli import gateway_runtime as runtime

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("a profile a plugin runs elsewhere must not be discovered or started")

    monkeypatch.setattr(runtime, "discover_gateway_endpoint", must_not_run)
    monkeypatch.setattr("hermes_cli.gateway_runtime_start.spawn_unmanaged_gateway", must_not_run)
    monkeypatch.setattr("hermes_cli.gateway_runtime_service.start_existing_gateway_service", must_not_run)
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin")

    for home, name in ((root / "profiles" / "ops", "ops"), (root, "default")):
        result = runtime.ensure_gateway_runtime(home, timeout=5)
        assert (result.state, result.reason_code, result.endpoint) == (
            "inaccessible", "profile_runs_elsewhere", None)
        assert f"'{name}'" in result.detail and "sandbox-plugin" in result.detail


def test_ensure_still_runs_for_an_unmarked_profile(root, monkeypatch):
    from hermes_cli import gateway_runtime as runtime

    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    observed = runtime.GatewayDiscovery("inaccessible", reason_code="authorization")
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: observed)
    assert runtime.ensure_gateway_runtime(root / "profiles" / "dev", timeout=5) is observed


def test_gateway_ensure_command_prints_the_refusal(root, monkeypatch, capsys):
    from argparse import Namespace
    from hermes_cli.gateway_runtime_cli import cmd_gateway_ensure

    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin", root=root)
    with pytest.raises(SystemExit) as exited:
        cmd_gateway_ensure(Namespace(timeout="5"))
    assert exited.value.code == 7
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "inaccessible" and payload["reason_code"] == "profile_runs_elsewhere"
    assert "'ops'" in payload["detail"] and "sandbox-plugin" in payload["detail"]


# ── Kanban ─────────────────────────────────────────────────────────────────────────────────────


def test_dispatcher_never_claims_a_card_for_a_marked_profile(root, all_assignees_spawnable):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.init_db()
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    with kbc.connect() as conn:
        marked = kb.create_task(conn, title="sandboxed", assignee="ops")
        assert kbd.has_spawnable_ready(conn) is False
        res = kbd.dispatch_once(conn, dry_run=True)
        assert res.skipped_nonspawnable == [marked] and res.spawned == []
        control = kb.create_task(conn, title="on this host", assignee="dev")
        res = kbd.dispatch_once(conn, dry_run=True)
        assert [t for t, _a, _w in res.spawned] == [control]
