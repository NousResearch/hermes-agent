"""The backend migration engine on the fake host: apply, rollback, restore, status, the run lock's
callers, --detach and --expect (the dry-run plan's tests stay in test_gateway_migrate_backends_apply.py,
which holds the shared fake host)."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import stat
import time
from pathlib import Path

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli.gateway_migrate_backends import BackendItem
from hermes_cli.gateway_migrate_backends_host import SystemHost
from tests.hermes_cli.test_gateway_migrate_backends_apply import (
    _BACKEND_PID,
    _GATEWAY_PID,
    _PASSWORD,
    _SECRET,
    FakeHost,
    Killed,
    _args,
    _config_files,
    _files,
    _nothing_applied,
    _plan,
)


def test_a_run_from_inside_the_old_unit_is_blocked_and_pointed_at_detach(home, capsys, monkeypatch):
    monkeypatch.setattr("hermes_cli.update_cmd_posix_pause._pid_cgroup",
                        lambda pid="self": "/fake/hermes-dashboard.service/shell")

    code, out = _plan(FakeHost(home), capsys)

    (blocker,) = out["plan"]["blockers"]
    assert code == engine.EXIT_BLOCKED and "runs inside hermes-dashboard.service" in blocker
    assert "--detach" in blocker


@pytest.mark.parametrize(("flags", "message"), [({"dry_run": True, "address": "nope"}, "--address must be HOST:PORT")])
def test_the_cli_refuses_a_malformed_address(flags, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        engine.cmd_backends(_args(**flags))
    assert exit_info.value.code == 2 and message in capsys.readouterr().err


@pytest.mark.parametrize(("bind", "expected"), [("0.0.0.0", "0.0.0.0"), ("127.0.0.1", "127.0.0.1"),
                                                ("192.168.7.20", "192.168.7.20"), ("example.lan", None)])
def test_never_widens_exposure(home, bind, expected):
    host = FakeHost(home, host=bind)
    before = _config_files(home)

    code = engine.cmd_apply(host, _args())

    if expected is None:  # a bind the gateway cannot take: refused, nothing touched
        assert code == engine.EXIT_BLOCKED
        assert host.calls == [] and _config_files(home) == before
        assert not engine.manifest_path(home).exists()
    else:
        assert code == engine.EXIT_OK
        settings = host.webapp_settings()
        assert (settings.host, settings.port) == (expected, 9119)


def test_a_specific_ip_backend_is_probed_where_it_listens_and_verified_on_ip_and_loopback(home):
    host = FakeHost(home, host="192.168.7.20")  # the old backend answers on that IP only

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    manifest = json.loads(engine.manifest_path(home).read_text(encoding="utf-8"))
    assert manifest["dial"]["old_host"] == "192.168.7.20"
    assert manifest["dial"]["verify_hosts"] == ["192.168.7.20", "127.0.0.1"]
    after_stop = host.probes[host.probes.index(("192.168.7.20", 9119)) + 1:]
    assert ("127.0.0.1", 9119) in after_stop  # the loopback companion was verified too


def test_never_prints_a_secret_and_carries_credentials_into_a_private_env(home, capsys):
    host = FakeHost(home)

    assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_OK
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    assert engine.print_status(host, as_json=True) == engine.EXIT_OK

    out = capsys.readouterr()
    manifest = engine.manifest_path(home).read_text(encoding="utf-8")
    for secret in (_PASSWORD, _SECRET):
        assert secret not in out.out and secret not in out.err and secret not in manifest
    env = (home / ".env").read_text(encoding="utf-8")
    assert _PASSWORD in env and _SECRET in env
    assert stat.S_IMODE((home / ".env").stat().st_mode) == 0o600
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in out.out  # the NAME is reported


def test_manifest_exists_before_the_first_change(home):
    host = FakeHost(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    assert host.calls and all(host.manifest_seen_before)
    manifest = json.loads(engine.manifest_path(home).read_text(encoding="utf-8"))
    for key in ("config", "env"):
        backup = manifest["backups"][key]
        assert stat.S_IMODE(os.stat(backup).st_mode) == 0o600
    assert manifest["old_backend"]["unit"] == "hermes-dashboard.service"
    assert manifest["gateway"]["installed"] is False and manifest["state"] == "done"


@pytest.mark.parametrize("gateway_before", ["not_installed", "running"])
def test_failed_verify_rolls_back_to_the_gateways_prior_state(home, gateway_before):
    running = gateway_before == "running"
    host = FakeHost(home, gateway_installed=running, gateway_active=running)
    host.gateway_lies = True  # the gateway never answers on the old address
    before_files, before_gw, before_linger = _config_files(home), dict(host.gw), host.linger_on

    code = engine.cmd_apply(host, _args())

    assert code == engine.EXIT_ROLLED_BACK
    assert host.gw == before_gw and host.linger_on == before_linger
    assert _config_files(home) == before_files
    assert host.old == {"enabled": True, "active": True}
    assert engine.status_payload(host)["state"] == "rolled_back"


def test_interrupted_run_is_rolled_back_by_the_next_invocation(home):
    host = FakeHost(home)
    before_files = _config_files(home)
    host.fail_at["gateway_start"] = Killed()

    with pytest.raises(Killed):
        engine.cmd_apply(host, _args())
    assert engine.status_payload(host)["state"] == "running"
    assert host.old["active"] is False  # the old backend was already stopped when the run died

    host.fail_at.clear()
    code = engine.cmd_apply(host, _args())

    assert code == engine.EXIT_ROLLED_BACK
    assert host.old == {"enabled": True, "active": True}
    assert host.gw == {"installed": False, "enabled": False, "active": False}
    assert _config_files(home) == before_files
    assert engine.status_payload(host)["state"] == "rolled_back"


def test_the_plan_lists_the_steps_the_manifest_and_status_report(home, capsys):
    host = FakeHost(home)

    assert engine.cmd_apply(host, _args(dry_run=True, json=True)) == engine.EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["steps"] == out["plan"]["steps"]
    assert all(s["title"] for s in out["steps"])
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    assert [s["name"] for s in out["steps"]] == [s["name"] for s in engine.status_payload(host)["steps"]]


# --- round 4: what cannot be read blocks (exit 3), nothing is guessed


def test_the_engines_own_steps_pin_a_hermes_root_that_is_not_the_standard_one(home, capsys, monkeypatch):
    monkeypatch.setattr("hermes_cli.gateway_migrate_backends_plan._is_standard_root", lambda root: False)
    host = FakeHost(home)
    host.gateway_lies = True
    host.fail_at["old_start"] = RuntimeError("could not start")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    steps = engine.status_payload(host)["manual_steps"]
    assert steps[0].startswith("HERMES_HOME=") and steps[0].endswith(" gateway stop")
    from hermes_cli.gateway_migrate_backends_lock import status_command
    assert status_command(home).startswith("HERMES_HOME=")


def test_the_gateway_install_output_is_shown_only_when_it_fails(capfd):
    import subprocess
    import sys

    from hermes_cli.gateway_migrate_backends_host import run_quietly

    def noisy(fail):
        print("Next steps: hermes gateway start")
        subprocess.run([sys.executable, "-c", "print('Created symlink from a child')"], check=True)
        if fail:
            raise RuntimeError("install failed")
        return "installed"

    assert run_quietly(lambda: noisy(False)) == "installed"
    assert capfd.readouterr().out == ""
    with pytest.raises(RuntimeError):
        run_quietly(lambda: noisy(True))
    out = capfd.readouterr().out
    assert "Next steps" in out and "Created symlink from a child" in out


@pytest.mark.parametrize(("fault", "step"), [("set_env", "write_config"), ("install_gateway", "gateway_service"),
                                             ("old_stop", "stop_old_backend"), ("gateway_start", "start_gateway"),
                                             ("verify", "verify")])
def test_a_failure_at_any_step_rolls_everything_back(home, fault, step):
    host = FakeHost(home)
    before = _config_files(home)
    if fault == "verify":
        host.gateway_lies = True
    else:
        host.fail_at[fault] = RuntimeError(f"{fault} broke")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    assert _config_files(home) == before and host.old == {"enabled": True, "active": True}
    assert host.gw == {"installed": False, "enabled": False, "active": False}
    status = engine.status_payload(host)
    assert status["step"] == step and f"{step} failed" in status["message"]
    assert engine.read_manifest(home)["entered"][-1] == step  # the journal ends at the step that failed


def test_no_environment_variable_changes_what_apply_does(home, monkeypatch):
    monkeypatch.setenv("HERMES_MIGRATE_TEST_FAIL_AT", "start_gateway")  # the removed test hook
    host = FakeHost(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK


def _interrupt(host):
    host.fail_at["gateway_start"] = Killed()
    with pytest.raises(Killed):
        engine.cmd_apply(host, _args())
    host.fail_at.clear()


def test_detach_hands_restore_and_an_interrupted_rollback_to_the_background_run(home, capsys):
    host = FakeHost(home)
    _interrupt(host)
    capsys.readouterr()

    assert engine.cmd_apply(host, _args(detach=True, json=True)) == engine.EXIT_OK
    assert engine.cmd_restore(host, _args(apply=False, restore=True, detach=True)) == engine.EXIT_OK

    # Each background run names this home's profile: a sticky `hermes profile use` cannot redirect it.
    assert host.spawned == [["-p", "default", "gateway", "migrate", "--backends", "--apply", "--yes", "--detached-run"],
                            ["-p", "default", "gateway", "migrate", "--backends", "--restore", "--yes",
                             "--detached-run"]]
    assert engine.status_payload(host)["state"] == "running"  # nothing rolled back in the caller
    assert json.loads(capsys.readouterr().out.splitlines()[0])["detached"] is True
    # The background run itself (--detached-run) rolls the interrupted run back.
    assert engine.cmd_restore(host, _args(apply=False, restore=True, detached_run=True)) == engine.EXIT_ROLLED_BACK
    assert host.old == {"enabled": True, "active": True}


def test_dry_run_json_stays_json_when_an_unfinished_run_exists(home, capsys):
    host = FakeHost(home)
    _interrupt(host)
    capsys.readouterr()

    assert engine.cmd_apply(host, _args(dry_run=True, json=True)) == engine.EXIT_BLOCKED

    out = json.loads(capsys.readouterr().out)
    assert out["unfinished_run"] is True and out["run"]["state"] == "running"


def test_a_refused_detached_run_is_readable_in_status_and_leaves_the_manifest_alone(home):
    from hermes_cli.backend_migration_marker import backend_retired_by_migration
    host = FakeHost(home)
    host.turns = 1  # a turn is running on the old backend: the background run refuses

    assert engine.cmd_apply(host, _args(detached_run=True)) == engine.EXIT_BLOCKED

    status = engine.status_payload(host)
    assert status["state"] == "failed" and "turn(s) are running" in status["message"]
    assert status["steps"] == [{"name": "preflight", "state": "failed"}] and status["started_at"] > 0
    assert not engine.manifest_path(home).exists() and backend_retired_by_migration(home) is False
    host.turns = 0
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK  # a later run replaces the refusal
    assert engine.status_payload(host)["state"] == "done"


def test_a_rollback_that_cannot_bring_the_old_backend_back_exits_5_with_numbered_steps(home, capsys):
    host = FakeHost(home)
    host.gateway_lies = True  # verify fails ...
    host.fail_at["old_start"] = RuntimeError("unit file broken")  # ... and the old unit will not start

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    out = capsys.readouterr().out
    status = engine.status_payload(host)
    assert status["state"] == "failed" and "rollback incomplete" in status["message"]
    assert status["manual_steps"][0] == " ".join(["hermes", "-p", "default", "gateway", "stop"])
    # From the recorded state: it was enabled and running, so exactly that comes back.
    assert status["manual_steps"][-2:] == ["systemctl --user enable hermes-dashboard.service",
                                           "systemctl --user start hermes-dashboard.service"]
    numbered = [f"  {n}. {s}" for n, s in enumerate(status["manual_steps"], 1)]
    assert "\n".join(numbered) in out  # the printed steps are exactly the ones --status reports
    assert host.gw == {"installed": False, "enabled": False, "active": False}  # the rest was undone


def test_a_hand_started_backends_command_line_is_never_in_the_manual_steps(home, capsys):
    host = FakeHost(home, supervisor="manual")
    host.argv += ["--token", _SECRET]
    host.gateway_lies = True  # verify fails ...
    host.fail_at["old_start"] = RuntimeError("cannot respawn")  # ... and the rollback cannot restart it

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    steps = engine.status_payload(host)["manual_steps"]
    assert steps and any("the way you started it before" in s for s in steps)
    assert _SECRET not in capsys.readouterr().out and not any(_SECRET in s for s in steps)


def test_a_gateway_without_the_webapp_page_fails_verify_and_rolls_back(home):
    host = FakeHost(home)
    host.bundle_broken = True  # /api/status is fine, the page is not

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    status = engine.status_payload(host)
    assert status["state"] == "rolled_back" and "not the Webapp page" in status["message"]
    assert host.old == {"enabled": True, "active": True}


class _Site(SystemHost):
    """The real page check against a real local HTTP server serving a temp bundle."""

    def __init__(self, home, dist):
        super().__init__(home)
        self.dist = dist

    def webapp_dist(self):
        return self.dist


@pytest.mark.parametrize(("answer", "ok"), [("index", True), ("login", True), ("unavailable", False),
                                             ("login-other-asset", False)])
def test_the_webapp_page_check_needs_this_bundle_served(home, tmp_path, answer, ok):
    import http.server
    import threading

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    index = '<!doctype html><html><head><script type="module" src="/assets/index-abc.js"></script></head></html>'
    (dist / "index.html").write_text(index, encoding="utf-8")
    (dist / "assets" / "index-abc.js").write_text("console.log('webapp')", encoding="utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # http.server API
            if self.path == "/" and answer == "index":
                body, code, extra = index.encode(), 200, {}
            elif self.path == "/" and answer == "unavailable":
                body, code, extra = b"<h1>Hermes is unavailable</h1>", 404, {}
            elif self.path == "/":
                body, code, extra = b"", 302, {"Location": "/login?next=%2F"}
            elif self.path == "/assets/index-abc.js":
                body = b"console.log('webapp')" if answer == "login" else b"console.log('another build')"
                code, extra = 200, {}
            else:
                body, code, extra = b"", 404, {}
            self.send_response(code)
            for k, v in extra.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # http.server API
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        problem = _Site(home, dist).webapp_page_problem("127.0.0.1", server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
    assert (problem is None) is ok, problem
    (dist / "index.html").rename(dist / "index.html.moved")  # a bundle broken after the build
    assert "cannot be served" in str(_Site(home, dist).webapp_page_problem("127.0.0.1", 9))


@pytest.mark.parametrize("ending", ["restore", "rollback", "interrupted"])
def test_the_old_unit_gets_a_restart_guard_and_the_update_marker_follows_every_state(home, ending):
    from pathlib import Path

    from hermes_cli.backend_migration_marker import backend_retired_by_migration
    host = FakeHost(home)
    if ending == "rollback":
        host.gateway_lies = True
    if ending == "interrupted":
        host.fail_at["gateway_start"] = Killed()
        with pytest.raises(Killed):
            engine.cmd_apply(host, _args())
        host.fail_at.clear()
    else:
        expected = engine.EXIT_OK if ending == "restore" else engine.EXIT_ROLLED_BACK
        assert engine.cmd_apply(host, _args()) == expected
    manifest = json.loads(engine.manifest_path(home).read_text(encoding="utf-8"))
    dropin = Path(manifest["old_unit_dropin"])
    assert dropin.parent == host.user_unit_dir() / "hermes-dashboard.service.d"
    if ending == "restore":  # migrated: the guard is in place and `hermes update` keeps the unit down
        assert "[Service]\nRestartPreventExitStatus=78\n" in dropin.read_text(encoding="utf-8")
        assert "daemon_reload_user" in host.calls
        assert manifest["state"] == "done" and backend_retired_by_migration(home) is True
        assert engine.restore(host, yes=True) == engine.EXIT_OK
        # the restore is its own run: a UI that asked for it after the migration recognizes it
        assert engine.status_payload(host)["started_at"] > manifest["finished_at"]
    elif ending == "interrupted":  # still `running`: kept down until the next run rolls it back
        assert dropin.exists() and backend_retired_by_migration(home) is True
        assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    assert not dropin.exists() and not dropin.parent.exists()
    assert engine.status_payload(host)["state"] == ("restored" if ending == "restore" else "rolled_back")
    assert backend_retired_by_migration(home) is False  # the old unit is managed normally again


def test_the_host_changes_in_contract_order(home):
    host = FakeHost(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    assert host.calls == ["set_env", "set_config", "install_gateway", "old_stop", "old_disable",
                          "daemon_reload_user", "gateway_start"]


class _NoBundleHost(FakeHost):
    """The Webapp bundle is missing and the build does not produce it."""

    def bundle_missing(self):
        return "apps/desktop/dist-webapp/index.html is missing"

    def build_bundle(self):
        self._change("build_bundle")


def test_a_bundle_that_cannot_be_built_stops_the_run_before_anything_else_changes(home):
    host = _NoBundleHost(home)
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    assert host.calls[0] == "build_bundle"
    assert not {"set_env", "set_config", "install_gateway", "old_stop", "old_disable"} & set(host.calls)
    assert _config_files(home) == before and host.old == {"enabled": True, "active": True}
    assert "webapp_bundle failed" in engine.status_payload(host)["message"]


def test_a_turn_on_the_old_backend_blocks_apply_unless_force_interrupt(home):
    host = FakeHost(home)
    host.turns = 1
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED
    assert host.calls == [] and _config_files(home) == before and not engine.manifest_path(home).exists()

    assert engine.cmd_apply(host, _args(force_interrupt=True)) == engine.EXIT_OK
    assert engine.status_payload(host)["state"] == "done"


def test_a_second_apply_after_a_migration_has_nothing_to_do(home, capsys):
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    host.calls.clear()
    capsys.readouterr()

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    assert host.calls == [] and "already migrated" in capsys.readouterr().out
    assert engine.status_payload(host)["state"] == "done"


def test_without_a_signing_key_one_is_generated_and_the_public_url_is_carried(home, capsys):
    class NoSecretHost(FakeHost):
        def process_env(self, pid):
            env = super().process_env(pid)
            env.pop("HERMES_DASHBOARD_BASIC_AUTH_SECRET")
            return env

    (home / "config.yaml").write_text("dashboard:\n  public_url: https://hermes.example.test\n", encoding="utf-8")
    host = NoSecretHost(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    env = dict(line.split("=", 1) for line in (home / ".env").read_text(encoding="utf-8").splitlines() if "=" in line)
    secret = env["HERMES_DASHBOARD_BASIC_AUTH_SECRET"]
    assert len(secret) >= 32 and secret not in capsys.readouterr().out
    from hermes_cli.config import load_config
    assert load_config()["webapp"]["public_url"] == "https://hermes.example.test"


@pytest.mark.parametrize("gateway_before", ["installed_disabled_stopped", "installed_enabled_stopped"])
def test_a_rollback_leaves_an_installed_gateway_exactly_as_it_was(home, gateway_before):
    host = FakeHost(home, gateway_installed=True, gateway_active=False)
    host.gw["enabled"] = gateway_before == "installed_enabled_stopped"
    host.gateway_lies = True
    before_gw = dict(host.gw)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    assert host.gw == before_gw and host.old == {"enabled": True, "active": True}


def test_a_hand_started_backend_migrates_and_a_restore_starts_it_again(home):
    host = FakeHost(home, supervisor="manual")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    manifest = engine.read_manifest(home)
    assert "old_unit_dropin" not in manifest and "daemon_reload_user" not in host.calls

    assert engine.restore(host, yes=True) == engine.EXIT_OK
    assert host.calls[-1] == "old_start" and host.old["active"] is True


def test_status_json_has_the_contract_shape(home, capsys):
    host = FakeHost(home)
    keys = {"state", "step", "steps", "message", "old_address", "new_address", "started_at", "finished_at",
            "manual_steps"}

    assert engine.print_status(host, as_json=True) == engine.EXIT_OK
    none = json.loads(capsys.readouterr().out)
    assert set(none) == keys and none["state"] == "none" and none["steps"] == []

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    capsys.readouterr()
    assert engine.print_status(host, as_json=True) == engine.EXIT_OK
    done = json.loads(capsys.readouterr().out)
    assert set(done) == keys and done["state"] == "done" and done["step"] == "verify"
    assert all(set(s) == {"name", "state"} and s["state"] in ("pending", "running", "done", "failed", "undone")
               for s in done["steps"])
    assert isinstance(done["started_at"], float) and isinstance(done["finished_at"], float)
    assert done["old_address"] == done["new_address"] == "0.0.0.0:9119" and done["manual_steps"] == []


def test_restore_without_a_finished_migration_is_blocked_and_changes_nothing(home, capsys):
    host = FakeHost(home)
    before = _config_files(home)

    assert engine.cmd_restore(host, _args(apply=False, restore=True)) == engine.EXIT_BLOCKED

    assert host.calls == [] and _config_files(home) == before and not engine.manifest_path(home).exists()
    assert "Nothing to restore" in capsys.readouterr().out


class _TwoBackendsHost(FakeHost):
    def backends(self):
        first = super().backends()
        if not first:
            return []
        second = dataclasses.replace(first[0], pid=_BACKEND_PID + 1, unit="hermes-serve.service",
                                     argv=[a if a != "9119" else "9129" for a in self.argv])
        return [first[0], second] if self.old["active"] else first


def test_several_backends_for_this_home_block_even_when_address_picks_one(home, capsys):
    host = _TwoBackendsHost(home)
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED
    assert host.calls == [] and _config_files(home) == before and not engine.manifest_path(home).exists()
    assert "--address" in capsys.readouterr().out

    capsys.readouterr()
    # Picking one does not make the other one go away: it would stay a second session owner.
    assert engine.cmd_apply(host, _args(address="0.0.0.0:9119")) == engine.EXIT_BLOCKED
    out = capsys.readouterr().out
    assert "Another old backend also serves this home: systemd hermes-serve.service" in out
    assert host.calls == [] and _config_files(home) == before and not engine.manifest_path(home).exists()


def test_a_webapp_already_enabled_elsewhere_blocks_and_explains_keep_webapp_address(home, capsys):
    (home / "config.yaml").write_text("webapp:\n  enabled: true\n  host: 127.0.0.1\n  port: 9120\n", encoding="utf-8")
    host = FakeHost(home)
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED

    assert host.calls == [] and _config_files(home) == before and not engine.manifest_path(home).exists()
    assert "--keep-webapp-address" in capsys.readouterr().out


# --------------------------------------------------------------------------- rollback undoes only what it changed


def test_a_failed_bundle_build_leaves_a_running_gateway_and_the_old_backend_untouched(home):
    host = _NoBundleHost(home, gateway_installed=True, gateway_active=True)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    assert host.calls == ["build_bundle"]  # no gateway stop/start, no old enable/start: nothing was changed
    assert host.gw == {"installed": True, "enabled": True, "active": True}


class _StuckGatewayHost(FakeHost):
    """The migrated gateway will not let go of the old address."""

    def gateway_verb(self, verb):
        if verb == "stop":
            self._change("gateway_stop")
            return  # "stopped", yet still running and listening
        super().gateway_verb(verb)

    def uninstall_gateway(self):
        self._change("uninstall_gateway")
        self.gw.update(installed=False, enabled=False)  # the unit is gone, its process is not

    def pid_listens_on(self, pid, port):
        return pid == _GATEWAY_PID and self.gw["active"]  # whatever config.yaml now says


@pytest.mark.parametrize("cause", ["gateway_stop_fails", "listener_stays", "config_restore_fails"])
def test_the_old_backend_is_never_started_beside_a_migrated_gateway(home, monkeypatch, capsys, cause):
    host = _StuckGatewayHost(home) if cause == "listener_stays" else FakeHost(home)
    host.gateway_lies = True  # verify fails after the gateway started on the old address
    if cause == "gateway_stop_fails":
        host.fail_at["gateway_stop"] = RuntimeError("stop timed out")
    if cause == "config_restore_fails":
        real = engine._restore_file

        def restore_file(home_, name, backup, mode=None):
            if name == "config.yaml":
                raise OSError("disk full")
            real(home_, name, backup, mode)
        monkeypatch.setattr(engine, "_restore_file", restore_file)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    assert "old_start" not in host.calls and "old_enable" not in host.calls
    assert host.old["active"] is False  # one session owner, never two
    status = engine.status_payload(host)
    assert status["state"] == "failed" and status["manual_steps"]


@pytest.mark.parametrize("state", ["linked", "linked-runtime", "generated", "masked", "indirect"])
def test_a_unit_whose_enablement_cannot_be_put_back_exactly_is_refused_before_any_change(home, capsys, state):
    host = FakeHost(home)
    host.unit_file_state = state
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED

    assert host.calls == [] and _config_files(home) == before and not engine.manifest_path(home).exists()
    assert f"hermes-dashboard.service is {state}" in capsys.readouterr().out


@pytest.mark.parametrize(("state", "tail"), [
    ("enabled-runtime", ["systemctl --user enable --runtime hermes-dashboard.service",
                         "systemctl --user start hermes-dashboard.service"]),
    ("static", ["systemctl --user daemon-reload", "systemctl --user start hermes-dashboard.service"])])
def test_the_manual_steps_put_back_the_recorded_unit_state(home, state, tail):
    host = FakeHost(home)
    host.unit_file_state = state
    host.gateway_lies = True
    host.fail_at["old_start"] = RuntimeError("unit file broken")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    steps = engine.status_payload(host)["manual_steps"]
    assert [s.rsplit("&& ", 1)[-1] for s in steps[-2:]] == tail
    assert not any("enable --now" in s for s in steps)


def test_a_hand_started_backend_without_a_start_time_is_refused(home, capsys):
    class NoStartTime(FakeHost):
        def start_time(self, pid):
            return None

    host = NoStartTime(home, supervisor="manual")

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED

    assert host.calls == [] and "cannot prove it would stop that process" in capsys.readouterr().out


# --------------------------------------------------------------------------- the manifest on disk


@pytest.mark.parametrize("content", ["{not json", "[]", '{"state": "running"}'])
def test_an_unreadable_manifest_is_never_mistaken_for_no_migration(home, capsys, content):
    host = FakeHost(home)
    (home / "backend_migration").mkdir()
    backup = home / "backend_migration" / "config.yaml.bak"
    backup.write_text("from the earlier run\n", encoding="utf-8")
    engine.manifest_path(home).write_text(content, encoding="utf-8")
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED
    assert engine.cmd_restore(host, _args(apply=False, restore=True)) == engine.EXIT_BLOCKED

    assert host.calls == [] and _config_files(home) == before
    assert backup.read_text(encoding="utf-8") == "from the earlier run\n"  # the backups stay what they were
    assert engine.manifest_path(home).read_text(encoding="utf-8") == content
    assert "move" in capsys.readouterr().out
    assert engine.status_payload(host)["state"] == "failed"


def test_secret_files_are_private_from_creation(home, monkeypatch):
    os.chmod(home / ".env", 0o640)
    host = FakeHost(home)
    host.gateway_lies = True
    monkeypatch.setattr(os, "chmod", lambda *a, **k: None)  # no tightening after the fact counts
    old_umask = os.umask(0)  # a file created without an explicit mode would be world-readable
    try:
        assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK
    finally:
        os.umask(old_umask)
    manifest = engine.read_manifest(home)
    for path in (manifest["backups"]["config"], manifest["backups"]["env"], engine.manifest_path(home)):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600, path
    assert manifest["backups"]["env_mode"] == 0o640


def test_a_rollback_puts_env_back_with_its_own_mode(home):
    os.chmod(home / ".env", 0o640)
    host = FakeHost(home)
    host.gateway_lies = True

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    assert stat.S_IMODE((home / ".env").stat().st_mode) == 0o640


# --------------------------------------------------------------------------- where the run itself lives


def test_a_background_run_that_landed_inside_the_old_unit_refuses(home):
    from hermes_cli.update_cmd_posix_pause import _pid_cgroup
    host = FakeHost(home)
    host.unit_cgroup = _pid_cgroup()  # this very process is in the old backend's unit

    assert engine.cmd_apply(host, _args(detached_run=True)) == engine.EXIT_BLOCKED

    assert host.calls == [] and not engine.manifest_path(home).exists()
    assert engine.status_payload(host)["state"] == "failed"  # the refusal is readable


def test_a_run_inside_the_gateway_unit_is_refused_before_any_change(home, capsys):
    host = FakeHost(home, gateway_installed=True, gateway_active=True)
    host.caller_unit = "hermes-gateway.service"

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED
    assert host.calls == [] and "runs inside hermes-gateway.service" in capsys.readouterr().out

    host.caller_unit = None
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    host.calls.clear()
    host.caller_unit = "hermes-gateway.service"
    assert engine.cmd_restore(host, _args(apply=False, restore=True)) == engine.EXIT_BLOCKED
    assert host.calls == [] and engine.status_payload(host)["state"] == "done"


def test_a_background_run_that_never_answers_is_unconfirmed_not_started(home, monkeypatch, capsys):
    from hermes_cli import gateway_migrate_backends_lock as run_lock

    class Silent(FakeHost):
        def spawn_detached(self, flags, log):
            self.spawned.append(flags)
            return {"pid": 778}  # started, never writes its acknowledgement

    monkeypatch.setattr(run_lock, "ACK_WAIT_SECONDS", 0.2)
    host = Silent(home)

    code = engine.cmd_apply(host, _args(detach=True, json=True))

    out = json.loads(capsys.readouterr().out)
    assert code == 7 and out["outcome"] == "unconfirmed" and out["confirmed"] is False


def test_a_background_run_that_cannot_leave_the_callers_unit_is_refused(home, capsys):
    class Refusing(FakeHost):
        def spawn_detached(self, flags, log):
            raise RuntimeError("cannot start the run outside hermes-dashboard.service")

    assert engine.cmd_apply(Refusing(home), _args(detach=True, json=True)) == engine.EXIT_BLOCKED
    assert json.loads(capsys.readouterr().out)["outcome"] == "refused"


class _SpawnHost(SystemHost):
    """The real spawn path; only systemd-run / the new-session child are stand-ins (they ack at once)."""

    def __init__(self, home, kind):
        super().__init__(home)
        self.kind = kind

    def gateway_kind(self):
        return self.kind

    def own_unit(self):
        return None


@pytest.mark.parametrize("kind", ["systemd", None])
def test_the_background_run_keeps_this_profile_over_a_sticky_one(tmp_path, monkeypatch, kind):
    import subprocess

    from hermes_cli import gateway_migrate_backends_host as host_module
    from hermes_cli import gateway_migrate_backends_lock as run_lock
    from hermes_cli._parser import scan_profile_flag
    root = tmp_path / "root"
    (root / "profiles" / "coder").mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    (root / "active_profile").write_text("coder\n", encoding="utf-8")  # `hermes profile use coder`
    monkeypatch.setenv("HERMES_HOME", str(root))
    launched: list[list[str]] = []

    def ack(argv):
        launched.append(list(argv))
        token = argv[argv.index("--detached-ack") + 1]
        run_lock.write_ack(root, token, locked=True)

    def run(cmd, **kw):
        ack(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    class Child:
        def __init__(self, argv, **kw):
            ack(argv)
            self.pid = 4321

    from types import SimpleNamespace

    from hermes_platform import resolver
    monkeypatch.setattr(resolver, "locate_command", lambda name, *a, **kw: SimpleNamespace(command="/usr/bin/" + name))
    monkeypatch.setattr(host_module.subprocess, "run", run)
    monkeypatch.setattr(host_module.subprocess, "Popen", Child)
    monkeypatch.setattr(host_module.SystemHost, "backends", lambda self: [])  # the caller's read-only pre-check

    assert engine.cmd_apply(_SpawnHost(root, kind), _args(detach=True)) == engine.EXIT_OK

    (argv,) = launched
    assert ("systemd-run" in argv[0]) is (kind == "systemd")
    flags = argv[argv.index("gateway") - 2:]
    assert flags[:2] == ["-p", "default"] and scan_profile_flag(flags)[0] == "default"


# --------------------------------------------------------------------------- --expect: the confirmed backend only


def _token(host, capsys):
    assert engine.cmd_apply(host, _args(dry_run=True, json=True)) == engine.EXIT_OK
    return json.loads(capsys.readouterr().out)["expect_token"]


def test_apply_with_the_confirmed_token_migrates(home, capsys):
    host = FakeHost(home)
    token = _token(host, capsys)

    assert token and engine.cmd_apply(host, _args(expect=token)) == engine.EXIT_OK


def test_a_replaced_hand_started_backend_is_not_stopped_with_an_old_token(home, capsys):
    class Restarted(FakeHost):
        started = 1000

        def start_time(self, pid):
            return self.started

    host = Restarted(home, supervisor="manual")
    token = _token(host, capsys)
    host.started = 2000  # restarted on the same port since the user confirmed the plan
    before = _config_files(home)

    assert engine.cmd_apply(host, _args(expect=token, json=True)) == engine.EXIT_PLAN_CHANGED

    out = json.loads(capsys.readouterr().out)
    assert out["plan_changed"] is True and out["expect_token"] != token
    assert host.calls == [] and _config_files(home) == before and not engine.manifest_path(home).exists()


def test_a_managed_units_restart_keeps_the_token(home, capsys):
    class Restarted(FakeHost):
        def start_time(self, pid):
            return 99999  # the unit restarted: a new process, the same backend

    assert _token(Restarted(home), capsys) == _token(FakeHost(home), capsys)


def test_a_background_run_checks_the_token_itself_and_leaves_the_reason(home, capsys):
    host = FakeHost(home)

    assert engine.cmd_apply(host, _args(detach=True, expect="feedface")) == engine.EXIT_OK
    assert host.spawned[-1][-2:] == ["--expect", "feedface"]
    assert engine.cmd_apply(host, _args(detached_run=True, expect="feedface")) == engine.EXIT_PLAN_CHANGED

    assert host.calls == [] and engine.status_payload(host)["state"] == "failed"


# --------------------------------------------------------------------------- fix round 2


class _StaticHost(FakeHost):
    """A `static` unit (no [Install]): disabling changes nothing, so the scan keeps listing it stopped."""

    unit_file_state = "static"

    def backends(self):
        item = BackendItem(supervisor="systemd", running=self.old["active"],
                           pid=_BACKEND_PID if self.old["active"] else None, argv=list(self.argv),
                           unit="hermes-dashboard.service", scope="user",
                           unit_path="/home/u/.config/systemd/user/hermes-dashboard.service", state="static",
                           hermes_home=str(self.home), this_home=True)
        return [item]


def test_a_second_apply_never_replans_over_a_finished_migration_or_its_backups(home, capsys):
    host = _StaticHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    manifest = engine.read_manifest(home)
    backups = {k: open(manifest["backups"][k], "rb").read() for k in ("config", "env")}
    host.calls.clear()
    capsys.readouterr()

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    assert host.calls == [] and "already migrated" in capsys.readouterr().out
    assert {k: open(manifest["backups"][k], "rb").read() for k in ("config", "env")} == backups
    assert engine.read_manifest(home)["started_at"] == manifest["started_at"]  # the same, finished run


def test_a_listener_probe_that_cannot_tell_keeps_the_old_backend_down(home):
    host = FakeHost(home)
    host.gateway_lies = True
    host.listener_unknown = True
    host.stray_listener = True

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    assert "old_start" not in host.calls and host.old["active"] is False
    assert "cannot tell whether the gateway still listens" in engine.status_payload(host)["message"]


def test_recovering_an_unfinished_run_inside_the_gateway_unit_is_refused(home, capsys):
    host = FakeHost(home)
    _interrupt(host)
    host.calls.clear()
    host.caller_unit = "hermes-gateway.service"

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED

    assert host.calls == [] and engine.read_manifest(home)["state"] == "running"
    assert "--restore" in capsys.readouterr().out


def test_an_expect_token_never_recovers_an_unfinished_run(home, capsys):
    host = FakeHost(home)
    _interrupt(host)
    host.calls.clear()
    capsys.readouterr()

    assert engine.cmd_apply(host, _args(expect="feedface", json=True)) == engine.EXIT_PLAN_CHANGED

    out = json.loads(capsys.readouterr().out)
    assert out["unfinished_run"] is True and out["plan_changed"] is True
    assert host.calls == [] and engine.read_manifest(home)["state"] == "running"


class _RespawnHost(FakeHost):
    """A hand-started backend whose restart gets a new pid, like a real one."""

    def __init__(self, home, **kw):
        super().__init__(home, supervisor="manual", **kw)
        self.current_pid = _BACKEND_PID
        self.respawns = 0
        self.die_after_respawn = False

    def backends(self):
        items = super().backends()
        for item in items:
            item.pid = self.current_pid if self.old["active"] else None
        return items

    def same_process(self, pid, start_time):
        return self.old["active"] and pid == self.current_pid

    def old_start(self, rec, launcher_env, argv=None):
        self._change("old_start")
        self.respawns += 1
        self.current_pid = 5000 + self.respawns
        self.old["active"] = True
        if self.die_after_respawn:
            raise Killed()  # the rollback dies right after it started the replacement
        return self.current_pid


def test_an_interrupted_rollback_never_starts_a_second_hand_started_backend(home):
    host = _RespawnHost(home)
    host.gateway_lies = True
    host.die_after_respawn = True
    with pytest.raises(Killed):
        engine.cmd_apply(host, _args())
    assert host.respawns == 1
    host.die_after_respawn = False

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK  # the next run finishes the rollback

    assert host.respawns == 1  # the one already listening on the address is kept
    assert engine.read_manifest(home)["old_backend_respawned"]["pid"] == 5001


def test_the_replacement_of_a_hand_started_backend_is_recorded(home):
    host = _RespawnHost(home)
    host.gateway_lies = True

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK

    assert engine.read_manifest(home)["old_backend_respawned"] == {"pid": 5001, "start_time": 6001}


def test_a_restarted_hand_started_backend_names_this_profile(tmp_path, monkeypatch):
    from hermes_cli import gateway_migrate_backends_host as host_module
    root = tmp_path / "root"
    (root / "profiles" / "coder").mkdir(parents=True)
    (root / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    (root / "active_profile").write_text("coder\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    started: list[list[str]] = []

    class Child:
        def __init__(self, argv, **kw):
            started.append(list(argv))
            self.pid = 4321

    monkeypatch.setattr(host_module.subprocess, "Popen", Child)
    raw = ["/venv/bin/hermes", "dashboard", "--no-open", "--port", "9119", "--auth-token", "tok-raw"]
    rec = {"supervisor": "manual", "argv": [*raw[:-1], "<redacted>"]}  # the report's argv is redacted

    SystemHost(root)._respawn_manual(rec, {"PATH": "/usr/bin"}, raw)

    (argv,) = started
    i = argv.index("dashboard")
    assert argv[i - 2:i] == ["-p", "default"]
    assert argv[-2:] == ["--auth-token", "tok-raw"]  # started from the raw command line, never the redacted one
    with pytest.raises(RuntimeError, match="start it again by hand"):
        SystemHost(root)._respawn_manual(rec, {"PATH": "/usr/bin"}, None)


@pytest.mark.parametrize("name", [".env", "config.yaml"])
def test_a_symlinked_env_or_config_is_refused_before_any_change(home, tmp_path, capsys, name):
    shared = tmp_path / f"shared-{name.strip('.')}"
    shared.write_bytes((home / name).read_bytes())
    os.chmod(shared, 0o644)
    (home / name).unlink()
    (home / name).symlink_to(shared)
    host = FakeHost(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED

    assert host.calls == [] and not engine.manifest_path(home).exists()
    assert "is a symbolic link" in capsys.readouterr().out
    assert (home / name).is_symlink() and stat.S_IMODE(shared.stat().st_mode) == 0o644


def test_restore_refuses_when_a_file_became_a_symlink(home, tmp_path):
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    shared = tmp_path / "shared-env"
    shared.write_bytes((home / ".env").read_bytes())
    (home / ".env").unlink()
    (home / ".env").symlink_to(shared)
    host.calls.clear()

    assert engine.restore(host, yes=True) == engine.EXIT_BLOCKED
    assert host.calls == [] and (home / ".env").is_symlink() and engine.read_manifest(home)["state"] == "done"


def test_restore_without_any_migration_refuses_and_writes_nothing(home, capsys):
    before = _files(home)

    with pytest.raises(SystemExit) as exit_info:
        engine.cmd_backends(_args(apply=False, restore=True))

    assert exit_info.value.code == engine.EXIT_BLOCKED and "Nothing to restore" in capsys.readouterr().out
    assert _files(home) == before  # not even the run lock's directory


# --------------------------------------------------------------------------- review 3


def test_a_hand_started_backend_whose_environment_cannot_be_read_is_never_retired(home, capsys, monkeypatch):
    host = FakeHost(home, supervisor="manual")
    monkeypatch.setattr(host, "process_env", lambda pid: None)
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED

    assert "recorded environment is missing or unreadable" in capsys.readouterr().out
    _nothing_applied(host, home, before)
    assert host.old["active"] is True
    assert not list((home / "backend_migration").glob("launcher-*.json"))


@pytest.mark.parametrize("lost", ["launcher_env", "launcher_argv"])
def test_a_rollback_without_the_launch_snapshot_leaves_the_old_backend_down(home, lost):
    host = FakeHost(home, supervisor="manual")
    host.gateway_lies = True  # verify fails after the old backend was stopped: rollback
    real_save = engine.Run.save

    def lose_snapshot(run):  # the private copy disappears after it was read back (disk, a cleanup tool)
        if run.data["step"] == "verify":
            with contextlib.suppress(FileNotFoundError):
                os.unlink(run.data["backups"][lost])
        real_save(run)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(engine.Run, "save", lose_snapshot)
        code = engine.cmd_apply(host, _args())

    assert code == engine.EXIT_ROLLBACK_INCOMPLETE
    assert "old_start" not in host.calls and host.old["active"] is False  # never the caller's environment
    status = engine.status_payload(host)
    assert "missing or unreadable" in status["message"] and status["manual_steps"]


def test_a_restore_without_the_launch_snapshot_changes_nothing_and_prints_steps(home, capsys):
    host = FakeHost(home, supervisor="manual")
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    os.unlink(engine.read_manifest(home)["backups"]["launcher_env"])
    host.calls.clear()
    capsys.readouterr()

    assert engine.restore(host, yes=True) == engine.EXIT_ROLLBACK_INCOMPLETE

    out = capsys.readouterr().out
    assert host.calls == [] and host.gw["active"] is True  # checked before the gateway is stopped
    assert "Nothing changed" in out and "1. " in out and engine.read_manifest(home)["state"] == "done"


def test_a_verified_restore_removes_the_credential_copies_and_a_failed_one_keeps_them(home):
    host = FakeHost(home, supervisor="manual")
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    paths = [engine.read_manifest(home)["backups"][k] for k in ("launcher_argv", "launcher_env")]
    assert all(os.path.exists(p) for p in paths)

    host.fail_at["old_start"] = RuntimeError("cannot start")
    assert engine.restore(host, yes=True) == engine.EXIT_ROLLBACK_INCOMPLETE
    assert all(os.path.exists(p) for p in paths)  # the next recovery needs them

    host.fail_at.clear()
    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK  # recovers the failed run
    assert not any(os.path.exists(p) for p in paths)
    assert engine.read_manifest(home)["backups"]["launcher_env"] is None


@pytest.mark.parametrize("listening", [True, False])
def test_an_unknown_gateway_identity_needs_a_free_port_before_the_old_backend_starts(home, listening):
    host = FakeHost(home)
    host.gateway_lies = True
    host.listener_unknown = True  # the identity record cannot be read: gateway_listens() is None
    host.stray_listener = listening  # else nothing at all listens on the port once the old one stopped

    code = engine.cmd_apply(host, _args())

    if listening:
        assert code == engine.EXIT_ROLLBACK_INCOMPLETE and "old_start" not in host.calls
    else:
        assert code == engine.EXIT_ROLLED_BACK and host.old["active"] is True


@pytest.mark.parametrize("mode", ["apply", "restore"])
@pytest.mark.parametrize("name", [".env", "config.yaml"])
def test_an_unfinished_run_is_not_recovered_while_a_file_is_a_symlink(home, tmp_path, capsys, mode, name):
    host = FakeHost(home)
    _interrupt(host)
    journal = engine.manifest_path(home).read_bytes()
    shared = tmp_path / f"shared-{name.strip('.')}"
    shared.write_bytes((home / name).read_bytes())
    (home / name).unlink()
    (home / name).symlink_to(shared)
    host.calls.clear()
    capsys.readouterr()

    code = (engine.cmd_apply(host, _args()) if mode == "apply"
            else engine.cmd_restore(host, _args(apply=False, restore=True)))

    assert code == engine.EXIT_BLOCKED and "symbolic link" in capsys.readouterr().out
    assert host.calls == [] and (home / name).is_symlink() and engine.manifest_path(home).read_bytes() == journal


# --------------------------------------------------------------------------- real-OS runs (macOS, Windows)


def test_a_python_m_backend_whose_working_directory_cannot_be_read_is_never_retired(home, capsys, monkeypatch):
    host = FakeHost(home, supervisor="manual")  # argv: /venv/bin/python -m hermes_cli.main dashboard ...
    monkeypatch.setattr(host, "process_cwd", lambda pid: None)
    before = _config_files(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_BLOCKED

    assert "working directory could not be read" in capsys.readouterr().out
    _nothing_applied(host, home, before)
    assert host.old["active"] is True


def test_a_hand_started_backend_is_restarted_in_its_recorded_working_directory(home):
    host = FakeHost(home, supervisor="manual")
    host.process_cwd = lambda pid: "/srv/hermes-agent"
    started = []
    host.old_start = lambda rec, env, argv=None: started.append(rec.get("cwd")) or host.old.update(active=True)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    assert engine.restore(host, yes=True) == engine.EXIT_OK

    assert started == ["/srv/hermes-agent"]


def test_an_old_backend_restarted_by_hand_is_no_proof_the_gateway_is_off(home):
    """One listener on the port proves nothing about the gateway: the run stays at exit 5 and says what to do."""
    host = FakeHost(home, supervisor="manual")
    host.listener_unknown = True  # no readable gateway identity (macOS after `gateway uninstall`)
    host.gateway_lies = True
    host.fail_at["old_start"] = RuntimeError("could not start")
    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE
    host.fail_at.clear()
    host.old["active"] = True  # the last printed step: the user started the old backend again by hand
    host.calls.clear()

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    assert "old_start" not in host.calls
    assert "stop it, then run this command again" in engine.status_payload(host)["message"]


def test_an_unknown_listener_says_what_to_do_next(home):
    host = FakeHost(home)
    host.gateway_lies = True
    host.listener_unknown = True
    host.stray_listener = True

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    assert "stop it, then run this command again" in engine.status_payload(host)["message"]


def test_detach_hands_an_unfinished_run_to_the_background_even_where_the_plan_is_manual_only(home):
    host = FakeHost(home)
    _interrupt(host)
    host.supervisor, host.scope = "systemd", "system"  # the backend found now could only be moved by hand
    host.old.update(active=True, enabled=True)  # (listed by the scan: a fake host lists only a live one)
    host.calls.clear()

    assert engine.cmd_apply(host, _args(detach=True)) == engine.EXIT_OK

    assert host.spawned and engine.read_manifest(home)["state"] == "running"  # recovery: the background run's


def test_detach_checks_a_stale_token_before_it_prints_the_manual_steps(home, capsys):
    host = FakeHost(home)
    token = _token(host, capsys)  # confirmed for the user unit
    host.supervisor, host.scope = "systemd", "system"  # replaced by a unit only steps can move

    assert engine.cmd_apply(host, _args(detach=True, expect=token, json=True)) == engine.EXIT_PLAN_CHANGED

    assert json.loads(capsys.readouterr().out)["plan_changed"] is True
    assert host.spawned == [] and host.calls == [] and engine.status_payload(host)["state"] == "none"


def test_an_acknowledgement_no_caller_waits_for_is_removed_by_the_next_run(home, capsys):
    from hermes_cli import gateway_migrate_backends_lock as run_lock
    folder = home / "backend_migration"
    folder.mkdir()
    abandoned, fresh = folder / "detach-ack-0123456789abcdef.json", folder / "detach-ack-fedcba9876543210.json"
    for path in (abandoned, fresh):
        path.write_text('{"locked": true, "pid": 3428}', encoding="utf-8")
    old = time.time() - run_lock.ACK_WAIT_SECONDS - 5  # written after its caller stopped waiting (exit 7)
    os.utime(abandoned, (old, old))

    with pytest.raises(SystemExit):
        engine.cmd_backends(_args(apply=False, restore=True))

    assert not abandoned.exists() and fresh.exists()  # a caller may still be waiting for the fresh one


def _old_comes_back_without_health(host, on: bool = True, *, already_stopped: bool = False) -> None:
    """Once the old backend has been stopped, what answers on its address again gives 401/403/404 on
    /api/health (``{"health": "missing"}``): something listens, but nothing proves it is the old backend."""
    real, stopped = host.http_status, [already_stopped]
    real_stop = host.old_stop

    def stop(rec):
        stopped[0] = True
        return real_stop(rec)

    def probe(h, port):
        status = real(h, port)
        return {"health": "missing"} if on and stopped[0] and status is not None and host.old["active"] else status

    host.old_stop, host.http_status = stop, probe


@pytest.mark.parametrize("health_missing", [True, False])
def test_a_restore_is_restored_only_after_a_real_health_result(home, health_missing):
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    _old_comes_back_without_health(host, health_missing, already_stopped=True)  # the apply stopped it

    code = engine.restore(host, yes=True)

    state = engine.status_payload(host)["state"]
    if health_missing:
        assert code == engine.EXIT_ROLLBACK_INCOMPLETE and state != "restored"
        assert "/api/health answered without a health result" in engine.status_payload(host)["message"]
    else:
        assert code == engine.EXIT_OK and state == "restored"


@pytest.mark.parametrize("health_missing", [True, False])
def test_a_failed_apply_is_rolled_back_only_after_a_real_health_result(home, health_missing):
    host = FakeHost(home)
    host.gateway_lies = True  # verify fails after the old backend stopped
    _old_comes_back_without_health(host, health_missing)

    code = engine.cmd_apply(host, _args())

    state = engine.status_payload(host)["state"]
    if health_missing:
        assert code == engine.EXIT_ROLLBACK_INCOMPLETE and state != "rolled_back"
    else:
        assert code == engine.EXIT_ROLLED_BACK and state == "rolled_back"


# --------------------------------------------------------------------------- --restore never loses later edits


def _restore_refused_and_nothing_changed(host, home, capsys, *expected):
    gw, old, files = dict(host.gw), dict(host.old), _config_files(home)
    capsys.readouterr()

    assert engine.restore(host, yes=True) == engine.EXIT_BLOCKED

    out = capsys.readouterr().out
    assert host.gw == gw and host.old == old and _config_files(home) == files  # the gateway keeps serving
    assert engine.read_manifest(home)["state"] == "done"
    assert "Nothing changed" in out and "To restore by hand, keeping your edits" in out
    for text in expected:
        assert text in out
    return out


@pytest.mark.parametrize("name", ["config.yaml", ".env"])
def test_restore_refuses_when_a_file_was_edited_after_the_migration(home, capsys, name):
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    path = home / name
    path.write_text(path.read_text(encoding="utf-8") + "# edited after the move\n", encoding="utf-8")

    _restore_refused_and_nothing_changed(host, home, capsys, f"{path} changed since the migration")


def test_restore_refuses_when_a_recorded_backup_cannot_be_read(home, capsys):
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    backup = Path(engine.read_manifest(home)["backups"]["config"])
    backup.unlink()

    _restore_refused_and_nothing_changed(host, home, capsys, f"the backup of config.yaml ({backup}) cannot be read")


def test_restore_refuses_a_manifest_without_the_after_apply_record(home, capsys):
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    m = engine.read_manifest(home)
    del m["files_after_apply"]
    engine.manifest_path(home).write_text(json.dumps(m), encoding="utf-8")

    _restore_refused_and_nothing_changed(host, home, capsys, "did not record config.yaml and .env")


def test_restore_of_unedited_files_still_restores(home):
    host = FakeHost(home)
    before = _config_files(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    assert engine.restore(host, yes=True) == engine.EXIT_OK
    assert _config_files(home) == before and engine.read_manifest(home)["state"] == "restored"


# --------------------------------------------------------------------------- unknown running turns are not 0


@pytest.mark.parametrize("force", [False, True])
def test_unknown_running_turns_block_like_a_running_turn(home, monkeypatch, force):
    """A plan whose running_turns is None (activity unreadable) never counts as 0: without
    --force-interrupt the apply refuses before any change; with it the move goes ahead."""
    real = engine.build_plan

    def unknown_turns(*a, **kw):
        plan = real(*a, **kw)
        plan.running_turns = None
        return plan

    monkeypatch.setattr(engine, "build_plan", unknown_turns)
    host = FakeHost(home)
    host.turns = 0
    before = _config_files(home)

    code = engine.cmd_apply(host, _args(force_interrupt=force))

    if force:
        assert code == engine.EXIT_OK
    else:
        assert code == engine.EXIT_BLOCKED
        _nothing_applied(host, home, before)



# --------------------------------------------------------------------------- another home's same-named gateway


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_gateway_unit_of_another_home_is_refused_before_any_change(home, capsys, dry_run):
    """``~/.hermes/profiles/work`` and ``/srv/team/profiles/work`` both map to ``hermes-gateway-work``."""
    host = FakeHost(home)
    host.foreign_home = "/srv/team/profiles/work"
    before = _config_files(home)

    assert engine.cmd_apply(host, _args(dry_run=dry_run)) == engine.EXIT_BLOCKED

    assert "belongs to another Hermes home (it runs HERMES_HOME=/srv/team/profiles/work)" in capsys.readouterr().out
    _nothing_applied(host, home, before)


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_legacy_hermes_service_is_refused_before_the_gateway_install_could_delete_it(home, capsys, dry_run):
    host = FakeHost(home)
    host.legacy_units = True
    before = _config_files(home)

    assert engine.cmd_apply(host, _args(dry_run=dry_run)) == engine.EXIT_BLOCKED

    out = capsys.readouterr().out
    assert "legacy Hermes gateway unit (hermes.service)" in out and "gateway migrate-legacy` first" in out
    _nothing_applied(host, home, before)


# --------------------------------------------------------------------------- edits are never written over


@pytest.mark.parametrize("name", ["config.yaml", ".env"])
def test_edit_during_verification_is_detected_by_restore(home, capsys, name):
    """Fingerprints are recorded at write_config, not after verification, so edits made during verify
    are never overwritten by a later --restore."""
    host = FakeHost(home)
    edited = []
    real_status = host.gateway_status

    def edit_during_verify(dial, port):
        if not edited:
            p = home / name
            p.write_text(p.read_text(encoding="utf-8") + "# edited during verify\n", encoding="utf-8")
            edited.append(True)
        return real_status(dial, port)

    host.gateway_status = edit_during_verify
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    _restore_refused_and_nothing_changed(host, home, capsys, f"{home / name} changed since the migration")


@pytest.mark.parametrize("name", ["config.yaml", ".env"])
def test_edit_during_verification_is_kept_when_rollback_runs(home, capsys, name):
    """If config.yaml or .env is edited during a failing verify, rollback keeps the user's edits,
    leaves the old backend down, and exits 5 with manual steps."""
    host = FakeHost(home)
    host.gateway_lies = True
    edited = []
    marker = "# operator edit during verify\n"
    real_status = host.gateway_status

    def edit_during_verify(dial, port):
        if not edited:
            p = home / name
            p.write_text(p.read_text(encoding="utf-8") + marker, encoding="utf-8")
            edited.append(True)
        return real_status(dial, port)

    host.gateway_status = edit_during_verify
    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    out = capsys.readouterr().out
    assert marker in (home / name).read_text(encoding="utf-8")
    assert host.old["active"] is False
    assert str(home / name) in out and "edits were kept" in out and "Do this by hand:" in out


@pytest.mark.parametrize("mutation", ["edit_config", "delete_backup"])
def test_change_during_restore_confirm_refuses_without_stopping_gateway(home, capsys, monkeypatch, mutation):
    """Files or backups that change while the confirmation prompt is open: refused, the gateway keeps
    serving."""
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    gw_before, old_before = dict(host.gw), dict(host.old)
    backup = Path(engine.read_manifest(home)["backups"]["config"])

    def mutate_at_prompt(question, yes):
        if mutation == "edit_config":
            (home / "config.yaml").write_text((home / "config.yaml").read_text(encoding="utf-8")
                                              + "# prompt edit\n", encoding="utf-8")
        else:
            backup.unlink()
        return True

    monkeypatch.setattr(engine, "_confirm", mutate_at_prompt)
    capsys.readouterr()

    assert engine.restore(host, yes=False) == engine.EXIT_BLOCKED

    out = capsys.readouterr().out
    assert host.gw == gw_before and host.old == old_before
    assert engine.read_manifest(home)["state"] == "done"
    if mutation == "edit_config":
        assert "# prompt edit\n" in (home / "config.yaml").read_text(encoding="utf-8")
        assert str(home / "config.yaml") in out and "edits were kept" in out
    else:
        assert str(backup) in out


@pytest.mark.parametrize("name", ["config.yaml", ".env"])
def test_interrupted_recovery_keeps_edits_made_before_recovery(home, capsys, name):
    """Recovering a killed run never overwrites operator edits made to config.yaml or .env before
    recovery runs: it keeps the edits, leaves the old backend down, and exits 5 with manual steps."""
    host = FakeHost(home)
    host.fail_at["gateway_start"] = Killed()
    with pytest.raises(Killed):
        engine.cmd_apply(host, _args())
    host.fail_at.clear()
    marker = "# edit before recovery\n"
    (home / name).write_text((home / name).read_text(encoding="utf-8") + marker, encoding="utf-8")
    capsys.readouterr()

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    out = capsys.readouterr().out
    assert marker in (home / name).read_text(encoding="utf-8")
    assert host.old["active"] is False
    assert str(home / name) in out and "edits were kept" in out and "Do this by hand:" in out


# --------------------------------------------------------------------------- hand-started backend rollback


def test_a_rollback_killed_as_it_records_the_end_keeps_the_launch_snapshot_for_the_next_run(home):
    """The copies of a hand-started backend's command line and environment go only after the final state
    is saved: a run killed in between is recovered with the old backend found running (exit 4)."""
    host = FakeHost(home, supervisor="manual")
    host.gateway_lies = True  # verify fails after the old backend was stopped: rollback
    real_finish = engine.Run.finish

    def killed_at_the_end(run, state, message):
        if state == "rolled_back" and not getattr(killed_at_the_end, "done", False):
            killed_at_the_end.done = True
            raise Killed()
        real_finish(run, state, message)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(engine.Run, "finish", killed_at_the_end)
        with pytest.raises(Killed):
            engine.cmd_apply(host, _args())
    assert host.old["active"] is True

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLED_BACK
    m = engine.read_manifest(home)
    assert m["state"] == "rolled_back" and m["backups"]["launcher_argv"] is None
    assert not list((home / "backend_migration").glob("launcher-*.json"))



_DETACHED_UNIT = "hermes-backend-migration-1700000000.service"


def test_a_background_rollback_never_starts_a_hand_started_backend_inside_its_own_unit(home, capsys):
    """Started from the background run's transient unit the old backend would be stopped with it: it stays
    down, the run ends with the command that starts it again from a terminal (exit 5)."""
    host = FakeHost(home, supervisor="manual")
    host.caller_unit = _DETACHED_UNIT
    host.gateway_lies = True

    assert engine.cmd_apply(host, _args()) == engine.EXIT_ROLLBACK_INCOMPLETE

    out = capsys.readouterr().out
    assert "old_start" not in host.calls and host.old["active"] is False
    assert _DETACHED_UNIT in out and "gateway migrate --backends --restore" in out
    assert engine.read_manifest(home)["state"] == "failed"
    assert list((home / "backend_migration").glob("launcher-*.json"))  # the terminal run needs them

    host.caller_unit = None  # from a terminal
    capsys.readouterr()
    assert engine.cmd_restore(host, _args(apply=False, restore=True)) == engine.EXIT_ROLLED_BACK
    assert host.old["active"] is True


def test_a_background_restore_of_a_hand_started_backend_changes_nothing(home, capsys):
    host = FakeHost(home, supervisor="manual")
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    host.caller_unit = _DETACHED_UNIT
    host.calls.clear()
    capsys.readouterr()

    assert engine.restore(host, yes=True, detached=True) == engine.EXIT_BLOCKED

    out = capsys.readouterr().out
    assert host.calls == [] and host.gw["active"] is True and engine.read_manifest(home)["state"] == "done"
    assert "Nothing changed" in out and _DETACHED_UNIT in out and "without --detach" in out


# --------------------------------------------------------------------------- unreadable turn markers, real apply path


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("force", [False, True])
def test_an_unreadable_turn_marker_blocks_the_real_apply_unless_forced(home, capsys, monkeypatch, force):
    """The real marker reader on a marker file that cannot be read: activity unknown, never idle. Without
    --force-interrupt the apply changes nothing and the old backend keeps running; with it the move goes
    ahead and says the activity could not be read."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root reads a mode-000 file")
    from tui_gateway import turn_marker
    marker = turn_marker._marker_path(home)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    marker.chmod(0o000)
    host = FakeHost(home)
    monkeypatch.setattr(host, "running_turns", SystemHost.running_turns.__get__(host, SystemHost))
    before = _config_files(home)
    try:
        code = engine.cmd_apply(host, _args(force_interrupt=force))
    finally:
        marker.chmod(0o600)

    out = capsys.readouterr().out
    assert "None" not in out
    if force:
        assert code == engine.EXIT_OK and host.old["active"] is False
        assert "turn activity could not be read" in out and "interrupted" in out
    else:
        assert code == engine.EXIT_BLOCKED and host.old["active"] is True
        _nothing_applied(host, home, before)
        assert "--force-interrupt" in out

