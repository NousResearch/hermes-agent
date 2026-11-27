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


def test_dashboard_file_requests_never_dial_a_marked_profiles_ssh_workspace(root, monkeypatch):
    from fastapi import HTTPException
    from hermes_cli import ssh_workspace_fs
    from hermes_cli.web_routers import files as file_routes

    for name in ("ops", "dev"):
        (root / "profiles" / name / "config.yaml").write_text(
            "terminal:\n  backend: ssh\n  ssh_host: ssh.example\n  ssh_user: dev\n", encoding="utf-8")
    dialed = []
    monkeypatch.setattr(ssh_workspace_fs, "get_ssh_workspace_fs", lambda key, _terminal: dialed.append(key) or key)
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    with pytest.raises(HTTPException) as refused:
        file_routes._fs_backend("ops")
    assert refused.value.status_code == 409 and "sandbox-plugin" in refused.value.detail
    assert dialed == []
    assert file_routes._fs_backend("dev") == "dev" and dialed == ["dev"]


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


def test_an_external_cron_reconcile_still_counts_the_marked_profiles(root, monkeypatch):
    """The external registry is not profile-scoped: reconciling it toward one profile's jobs would
    cancel the armed one-shots of the profiles the plugins run elsewhere."""
    from cron import scheduler_provider as sp
    from hermes_cli import web_server_cron

    reconciled = []

    class _External:
        name = "external"

        def on_jobs_changed(self):
            reconciled.append(True)

    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: _External())
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    pre.mark_profile_run_elsewhere("dev", "sandbox-plugin")
    web_server_cron._notify_cron_provider_for_profile("default")
    assert reconciled == []


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


def test_ensure_refuses_a_marked_profile_whose_directory_is_a_link_out_of_the_root(root, tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli.gateway_runtime_attach import attach_gateway

    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: pytest.fail("discovered"))
    outside = tmp_path / "other-disk" / "ops"
    outside.parent.mkdir()
    (root / "profiles" / "ops").rename(outside)
    _link(root / "profiles" / "ops", outside)
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    result = runtime.ensure_gateway_runtime(root / "profiles" / "ops", timeout=5)
    assert (result.state, result.reason_code) == ("inaccessible", "profile_runs_elsewhere")
    attached = attach_gateway(root / "profiles" / "ops", start=True, timeout=5)  # resolves before it ensures
    assert (attached.receipt.reason_code, attached.ticket) == ("profile_runs_elsewhere", None)


def test_a_gateway_client_refuses_a_marked_profile_whose_directory_is_a_link_out_of_the_root(root, tmp_path,
                                                                                             monkeypatch):
    """``connect_gateway`` resolves this process's home before it ensures a gateway for it. ``hermes
    sessions discard`` and ``adopt`` and the cron and Kanban clients call it, and none of them passes
    the command line's own refusal first."""
    import asyncio
    from hermes_cli import gateway_client
    from hermes_cli import gateway_runtime as runtime

    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: pytest.fail("discovered"))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    outside = tmp_path / "other-disk" / "ops"
    outside.parent.mkdir()
    (root / "profiles" / "ops").rename(outside)
    _link(root / "profiles" / "ops", outside)
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))

    async def connect():
        async with gateway_client.connect_gateway():
            pytest.fail("connected")

    with pytest.raises(gateway_client.GatewayUnavailableError) as refused:
        asyncio.run(connect())
    assert (refused.value.state, refused.value.reason_code) == ("inaccessible", "profile_runs_elsewhere")


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


# ── a second name for a marked profile's home ──────────────────────────────────────────────────


def _link(link: Path, target: Path) -> None:
    """Make *link* lead to the folder *target*: a symlink, or on Windows a junction when this
    account may not create symlinks. Skips where the OS offers neither."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        if os.name != "nt":
            pytest.skip(f"cannot create a directory symlink here: {exc}")
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


@pytest.fixture
def alias(root):
    """``profiles/alias``, a link to the home of ``ops``: the same profile under a second name."""
    _link(root / "profiles" / "alias", root / "profiles" / "ops")
    return root / "profiles" / "alias"


def test_a_link_to_a_marked_profiles_home_is_not_served_either(root, alias):
    """The served list asks by home: marking ``ops`` takes its other name out with it."""
    from hermes_cli.profiles import profiles_to_serve

    assert _names(profiles_to_serve(True)) == ["default", "alias", "dev", "ops"]
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert pre.run_elsewhere_for_home(alias).profile == "ops"
    assert _names(profiles_to_serve(True)) == ["default", "dev"]
    assert _names(profiles_to_serve(True, include_run_elsewhere=True)) == ["default", "alias", "dev", "ops"]

    pre.clear_profile_run_elsewhere("ops", "sandbox-plugin")
    assert _names(profiles_to_serve(True)) == ["default", "alias", "dev", "ops"]


def test_dashboard_refuses_requests_scoped_to_a_link_to_a_marked_profiles_home(root, alias):
    from fastapi import HTTPException
    from hermes_cli import web_server_cron

    assert web_server_cron._cron_profile_home("alias")[0] == "alias"
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    with pytest.raises(HTTPException) as refused:
        web_server_cron._cron_profile_home("alias")
    assert refused.value.status_code == 409
    assert "'ops'" in refused.value.detail and "sandbox-plugin" in refused.value.detail


def test_dispatcher_never_claims_a_card_for_a_link_to_a_marked_profiles_home(root, alias, all_assignees_spawnable):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.init_db()
    assert pre.profile_exists_here("alias") is True
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert pre.profile_exists_here("alias") is False and pre.profile_exists_here("dev") is True
    with kbc.connect() as conn:
        linked = kb.create_task(conn, title="sandboxed, under its other name", assignee="alias")
        res = kbd.dispatch_once(conn, dry_run=True)
        assert res.skipped_nonspawnable == [linked] and res.spawned == []


def test_the_served_list_of_a_large_host_is_answered_from_stats(root, monkeypatch):
    """The list asks about every profile at each rescan. The kept answers hold a whole host's
    profiles, so a rescan with nothing changed reads no home and no marker folder again."""
    from hermes_cli.profiles import profiles_to_serve

    for index in range(300):
        home = root / "profiles" / f"p{index:03d}"
        home.mkdir()
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    pre.mark_profile_run_elsewhere("p007", "sandbox-plugin")
    first = _names(profiles_to_serve(True))
    assert len(first) == 302 and "p007" not in first  # default, dev, ops and 299 of the 300

    read_again = []
    readings = pre._home_readings
    monkeypatch.setattr(pre, "_home_readings", lambda home: read_again.append(home) or readings(home))
    assert _names(profiles_to_serve(True)) == first
    assert read_again == []


# ── a marked name that is itself a link: its directory under the directory's own name ──────────


def _repoint(link: Path, target: Path) -> None:
    """Point the existing link *link* at the folder *target*."""
    os.rmdir(link) if os.name == "nt" else os.unlink(link)  # the link alone, not what it leads to
    _link(link, target)


@pytest.fixture
def linked(root):
    """``profiles/linked``, a link to the home of ``dev``; the tests mark ``linked``, not ``dev``."""
    _link(root / "profiles" / "linked", root / "profiles" / "dev")
    return root / "profiles" / "dev"


def test_the_directory_of_a_marked_link_is_not_served_under_its_own_name(root, linked):
    """The backend elsewhere runs in one directory. Marking a name that is a link to it takes the
    directory's own name out of the served list as well."""
    from hermes_cli.profiles import profiles_to_serve

    assert _names(profiles_to_serve(True)) == ["default", "dev", "linked", "ops"]
    pre.mark_profile_run_elsewhere("linked", "sandbox-plugin")
    assert pre.run_elsewhere("dev") is None  # no marker carries the directory's own name
    assert _names(profiles_to_serve(True)) == ["default", "ops"]
    assert _names(profiles_to_serve(True, include_run_elsewhere=True)) == ["default", "dev", "linked", "ops"]

    pre.clear_profile_run_elsewhere("linked", "sandbox-plugin")
    assert _names(profiles_to_serve(True)) == ["default", "dev", "linked", "ops"]


def test_no_agent_is_built_in_the_directory_of_a_marked_link(root, linked):
    pre.require_home_runs_here(linked)
    pre.mark_profile_run_elsewhere("linked", "sandbox-plugin")
    assert pre.run_elsewhere_for_home(linked).profile == "linked"
    with pytest.raises(pre.ProfileRunsElsewhere, match="'linked'.*sandbox-plugin"):
        pre.require_home_runs_here(linked)
    pre.require_home_runs_here(root / "profiles" / "ops")


def test_no_env_value_is_read_from_the_directory_of_a_marked_link(root, linked):
    from agent import secret_scope

    (linked / ".env").write_text("LINKED_KEY=written-by-the-backend-elsewhere\n", encoding="utf-8")
    assert secret_scope.load_env_file(linked / ".env") == {"LINKED_KEY": "written-by-the-backend-elsewhere"}
    pre.mark_profile_run_elsewhere("linked", "sandbox-plugin")
    assert secret_scope.load_env_file(linked / ".env") == {}
    assert "LINKED_KEY" not in secret_scope.build_profile_secret_scope(linked)


def test_dashboard_refuses_requests_scoped_to_the_directory_of_a_marked_link(root, linked):
    from fastapi import HTTPException
    from hermes_cli import web_server_cron

    assert web_server_cron._cron_profile_home("dev")[0] == "dev"
    pre.mark_profile_run_elsewhere("linked", "sandbox-plugin")
    with pytest.raises(HTTPException) as refused:
        web_server_cron._cron_profile_home("dev")
    assert refused.value.status_code == 409
    assert "'linked'" in refused.value.detail and "sandbox-plugin" in refused.value.detail


def test_dispatcher_never_claims_a_card_for_the_directory_of_a_marked_link(root, linked, all_assignees_spawnable):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.init_db()
    assert pre.profile_exists_here("dev") is True
    pre.mark_profile_run_elsewhere("linked", "sandbox-plugin")
    assert pre.profile_exists_here("dev") is False and pre.profile_exists_here("ops") is True
    with kbc.connect() as conn:
        card = kb.create_task(conn, title="sandboxed, under its directory's name", assignee="dev")
        res = kbd.dispatch_once(conn, dry_run=True)
        assert res.skipped_nonspawnable == [card] and res.spawned == []


def test_a_marked_link_to_the_default_home_marks_the_default_profile(root):
    _link(root / "profiles" / "linked", root)
    assert pre.run_elsewhere_for_home(root) is None and not pre.home_runs_elsewhere(root)
    pre.mark_profile_run_elsewhere("linked", "sandbox-plugin")
    assert pre.run_elsewhere_for_home(root).profile == "linked"
    assert pre.home_runs_elsewhere(root)
    assert not pre.home_runs_elsewhere(root / "profiles" / "ops")


def test_a_marker_for_a_link_is_seen_at_the_next_asking_about_its_directory(root, linked):
    """The kept answer for the directory's own name learns of a marker under another name."""
    assert not pre.home_runs_elsewhere(linked)
    pre.mark_profile_run_elsewhere("linked", "sandbox-plugin")
    assert pre.home_runs_elsewhere(linked)
    pre.clear_profile_run_elsewhere("linked", "sandbox-plugin")
    assert not pre.home_runs_elsewhere(linked)


def test_a_marked_link_pointed_at_another_directory_is_seen_at_the_next_asking(root, linked, monkeypatch):
    """No marker is written or removed when a link is pointed elsewhere, and the marker folder's
    time does not move: the directory each marked profile's home leads to is part of what is
    compared. Asked long after the folder last changed, so only stats are left to tell."""
    link, ops = root / "profiles" / "linked", root / "profiles" / "ops"
    pre.mark_profile_run_elsewhere("linked", "sandbox-plugin")
    assert pre.home_runs_elsewhere(linked) and not pre.home_runs_elsewhere(ops)
    later = pre.time.time_ns() + 10 * pre._FOLDER_SETTLES_NS
    monkeypatch.setattr(pre.time, "time_ns", lambda: later)
    assert pre.home_runs_elsewhere(linked) and not pre.home_runs_elsewhere(ops)

    _repoint(link, ops)
    assert not pre.home_runs_elsewhere(linked) and pre.home_runs_elsewhere(ops)
    assert pre.run_elsewhere_for_home(linked) is None and pre.run_elsewhere_for_home(ops).profile == "linked"
    _repoint(link, linked)
    assert pre.home_runs_elsewhere(linked) and not pre.home_runs_elsewhere(ops)


def test_a_marker_for_a_link_is_seen_when_the_folders_time_does_not_move(root, linked):
    """Two changes of the marker folder can share one tick of the filesystem's clock, and then the
    folder's time says nothing about the second: a folder that changed a moment ago is listed at
    every asking."""
    folder = pre.run_elsewhere_dir(root)
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert not pre.home_runs_elsewhere(linked)
    before = os.stat(folder)
    marker = folder / "linked.json"
    for change, expected in ((lambda: marker.write_text("{}", encoding="utf-8"), True), (marker.unlink, False)):
        change()
        os.utime(folder, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = os.stat(folder)
        assert (after.st_mode, after.st_size, after.st_mtime_ns) == (before.st_mode, before.st_size, before.st_mtime_ns)
        assert pre.home_runs_elsewhere(linked) is expected


def test_a_listing_taken_just_after_the_folder_changed_is_taken_again_however_late(root, linked, monkeypatch):
    """Whether the folder's time can be relied on is judged when the folder is listed. An answer
    from a listing taken in the tick of the folder's last change is not kept as final, even when
    nobody asks again until long after."""
    folder = pre.run_elsewhere_dir(root)
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert not pre.home_runs_elsewhere(linked)
    before = os.stat(folder)
    (folder / "linked.json").write_text("{}", encoding="utf-8")  # a second change, in the same tick
    os.utime(folder, ns=(before.st_atime_ns, before.st_mtime_ns))
    later = pre.time.time_ns() + 10 * pre._FOLDER_SETTLES_NS
    monkeypatch.setattr(pre.time, "time_ns", lambda: later)
    assert pre.home_runs_elsewhere(linked)


def test_a_marker_folder_that_has_been_quiet_is_not_listed_again(root, linked, monkeypatch):
    """Value reads come by the hundred: once the folder's time can be relied on, each asking costs
    stats (the home, its marker paths, the home of each marked profile), no listing and no resolve."""
    pre.mark_profile_run_elsewhere("linked", "sandbox-plugin")
    # The clock stands at the folder's own time: "a moment after the marker was written" for as
    # long as the first loop takes, however slowly this test runs.
    written = pre.run_elsewhere_dir().stat().st_mtime_ns
    monkeypatch.setattr(pre.time, "time_ns", lambda: written)
    homes = (linked, root / "profiles" / "ops", root)
    answers = [pre.home_runs_elsewhere(home) for home in homes]
    assert answers == [True, False, False]

    listings, resolves = [], []
    real_listdir, real_readings = os.listdir, pre._home_readings

    def listdir(path):
        if Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            listings.append(path)
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)
    monkeypatch.setattr(pre, "_home_readings", lambda home: resolves.append(home) or real_readings(home))
    for _ in range(50):  # a moment after the marker was written: listed, not resolved
        assert [pre.home_runs_elsewhere(home) for home in homes] == answers
    assert resolves == [] and len(listings) == 150
    later = pre.time.time_ns() + 10 * pre._FOLDER_SETTLES_NS
    monkeypatch.setattr(pre.time, "time_ns", lambda: later)
    assert [pre.home_runs_elsewhere(home) for home in homes] == answers  # the last listing, now kept
    del listings[:]
    for _ in range(50):
        assert [pre.home_runs_elsewhere(home) for home in homes] == answers
    assert resolves == [] and listings == []


def test_homes_that_are_not_directories_are_never_the_same_directory(root, monkeypatch):
    """Two homes that do not exist share an error, and on a filesystem without file ids every
    directory shares the id zero: neither makes a home the directory of a marked profile."""
    pre.mark_profile_run_elsewhere("gone", "sandbox-plugin")  # a marker whose profile has no directory
    absent = root / "profiles" / "absent"
    assert pre.run_elsewhere_for_home(absent) is None and not pre.home_runs_elsewhere(absent)

    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    monkeypatch.setattr(pre, "_home_target", lambda home: (7, 0))
    dev = root / "profiles" / "dev"
    assert pre.run_elsewhere_for_home(dev) is None and not pre.home_runs_elsewhere(dev)
    assert pre.home_runs_elsewhere(root / "profiles" / "ops")  # by its own name, as before
