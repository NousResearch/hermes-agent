"""The backend migration and a profile a plugin runs elsewhere.

``hermes gateway migrate --backends --apply`` moves a home from the old ``hermes dashboard`` /
``hermes serve`` backend to the gateway: it writes ``.env`` and ``config.yaml``, installs and starts the
gateway service and stops the old backend. For a profile a plugin runs elsewhere
(``hermes_cli.profiles_run_elsewhere``) the gateway it would start exits 78, and the home's files are
written by the backend the plugin runs. So the move is blocked before anything changes, and the plan a
person reads first says why.

On Windows the backend the plugin runs is a process of this host (``hermes serve --isolated`` under the
profile's home), so the plan finds it as an old backend of that home, and the steps of a move by hand
end it. For a marked home the plan prints no step.

The command itself refuses before the plan: every form of ``--apply`` and ``--restore`` ends at the
entry they all pass through, before the home's record is read or its run lock is made. A home can
hold the record of a move made before a plugin took the profile over; while the profile is marked
that record is neither undone nor rolled back, and no run is started in the background.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli import profiles_run_elsewhere as pre
from hermes_cli.gateway_migrate_backends import BackendItem
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost, Killed, _args, _files


def test_the_fixture_home_is_moved_while_no_plugin_runs_it(home):
    host = FakeHost(home)

    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK

    assert "install_gateway" in host.calls and "old_stop" in host.calls
    assert engine.manifest_path(home).exists()


def test_a_home_a_plugin_runs_elsewhere_is_not_moved(home, capsys):
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
    refusal = pre.run_elsewhere_refusal(home)
    assert refusal  # the fixture's home is the marked one
    host = FakeHost(home)
    before = _files(home)

    code = engine.cmd_apply(host, _args())

    assert host.calls == [], f"the migration changed the host for a marked profile: {host.calls}"
    assert _files(home) == before and not engine.manifest_path(home).exists()
    assert code == engine.EXIT_BLOCKED
    assert refusal in " ".join(capsys.readouterr().out.split())


def test_the_dry_run_plan_says_a_plugin_runs_the_profile_elsewhere(home, capsys):
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
    host = FakeHost(home)
    before = _files(home)

    code = engine.cmd_apply(host, _args(dry_run=True))

    assert code == engine.EXIT_BLOCKED and host.calls == [] and _files(home) == before
    assert pre.run_elsewhere_refusal(home) in " ".join(capsys.readouterr().out.split())


class ByHandHost(FakeHost):
    """A Windows host whose one backend of this home is a hand-started ``serve``: the plan cannot drive
    it, prints the steps of a move by hand and exits 6. A sandbox's backend looks the same from here."""

    def os_family(self):
        return "win32"

    def backends(self):
        return [BackendItem(supervisor="manual", running=True, pid=4747, hermes_home=str(self.home), this_home=True,
                            argv=[r"C:\hermes\python.exe", "-m", "hermes_cli.main", "serve", "--isolated", "--host",
                                  "127.0.0.1", "--port", "47999"])]


def test_a_move_by_hand_lists_its_steps_while_no_plugin_runs_the_home(home, capsys):
    code = engine.cmd_apply(ByHandHost(home), _args(dry_run=True))

    assert code == engine.EXIT_UNSUPPORTED
    out = capsys.readouterr().out
    assert "taskkill /PID 4747 /T /F" in out and "config set webapp.enabled true" in out


def test_no_step_of_a_move_by_hand_is_printed_for_a_home_a_plugin_runs_elsewhere(home, capsys):
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
    host = ByHandHost(home)
    before = _files(home)

    code = engine.cmd_apply(host, _args(dry_run=True))

    out = capsys.readouterr().out
    assert code == engine.EXIT_BLOCKED and host.calls == [] and _files(home) == before
    assert "taskkill" not in out and "config set" not in out, out  # no step: none is listed under "Steps"
    assert pre.run_elsewhere_refusal(home) in " ".join(out.split())


# ── every form of the command, through the entry they all pass ───────────────────────


_BACKGROUND_RUN = {"detached_run": True, "detached_ack": "token-1"}  # what `--detach` starts
_FORMS = {
    "apply": {},
    "apply, forced": {"force_interrupt": True},
    "apply, dry run": {"dry_run": True},
    "apply in the background": {"detach": True},
    "the background run of an apply": _BACKGROUND_RUN,
    "restore": {"apply": False, "restore": True},
    "restore in the background": {"apply": False, "restore": True, "detach": True},
    "the background run of a restore": {"apply": False, "restore": True, **_BACKGROUND_RUN},
}
_RECORDS = ("no record", "a finished move", "an unfinished move")


def _record(home, kind: str) -> None:
    """Leaves in *home* what a move left that was made while the profile still ran here: nothing, a
    finished move, or one that died after it had stopped the old backend."""
    host = FakeHost(home)
    if kind == "a finished move":
        assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    elif kind == "an unfinished move":
        host.fail_at["gateway_start"] = Killed()
        with pytest.raises(Killed):
            engine.cmd_apply(host, _args())


def _command(monkeypatch, home, **flags):
    """``hermes gateway migrate --backends`` with *flags*, through ``cmd_backends`` on a fresh fake host of
    *home*. Returns the exit code and that host."""
    from hermes_cli import gateway_migrate_backends_lock as run_lock
    host = FakeHost(home)
    monkeypatch.setattr("hermes_cli.gateway_migrate_backends_host.SystemHost", lambda _home: host)
    try:
        with pytest.raises(SystemExit) as exited:
            engine.cmd_backends(_args(**flags))
    finally:
        run_lock.release()
    return exited.value.code, host


def test_the_command_moves_the_fixture_home_while_no_plugin_runs_it(home, monkeypatch):
    code, host = _command(monkeypatch, home)

    assert code == engine.EXIT_OK and "install_gateway" in host.calls and "old_stop" in host.calls


@pytest.mark.parametrize("record", _RECORDS)
@pytest.mark.parametrize("form", _FORMS)
def test_no_form_of_the_command_touches_a_home_a_plugin_runs_elsewhere(home, monkeypatch, capsys, form, record):
    """Each form, without and with the record of an earlier move: exit 3 with the run-elsewhere message, no
    call to the host, no run started, and no file made or changed in the home (the run lock included).
    The answer names no command that would undo or roll back the earlier move."""
    _record(home, record)
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
    refusal, before = pre.run_elsewhere_refusal(home), _files(home)
    capsys.readouterr()

    code, host = _command(monkeypatch, home, **_FORMS[form])

    out = " ".join(capsys.readouterr().out.split())
    assert host.calls == [] and host.spawned == [], (host.calls, host.spawned)
    assert _files(home) == before
    assert code == engine.EXIT_BLOCKED and refusal in out
    assert "--restore" not in out and "rolls it back" not in out


def test_a_caller_that_asked_for_json_gets_the_refusal_as_one_json_line(home, monkeypatch, capsys):
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)

    code, host = _command(monkeypatch, home, detach=True, json=True)

    answer = json.loads(capsys.readouterr().out)
    assert code == engine.EXIT_BLOCKED and host.spawned == []
    assert {k: answer[k] for k in ("applied", "detached", "mode", "outcome", "error")} == {
        "applied": False, "detached": False, "mode": "apply", "outcome": "refused",
        "error": "profile_runs_elsewhere"}
    assert pre.run_elsewhere_refusal(home) in answer["message"]


def test_status_still_reads_the_record_of_a_home_a_plugin_runs_elsewhere(home, monkeypatch, capsys):
    _record(home, "a finished move")
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
    before = _files(home)
    capsys.readouterr()

    code, host = _command(monkeypatch, home, apply=False, status=True)

    assert code == engine.EXIT_OK and "Backend migration: done" in capsys.readouterr().out
    assert host.calls == [] and _files(home) == before
