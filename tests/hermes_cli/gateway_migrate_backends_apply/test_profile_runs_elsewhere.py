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
"""

from __future__ import annotations

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli import profiles_run_elsewhere as pre
from hermes_cli.gateway_migrate_backends import BackendItem
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost, _args, _files


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
