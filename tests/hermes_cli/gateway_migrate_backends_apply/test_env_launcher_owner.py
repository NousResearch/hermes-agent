"""A stopped systemd unit whose ExecStart starts the backend through ``env`` (``/usr/bin/env HERMES_HOME=/srv/other
hermes dashboard``): ``env``'s pairs come after systemd's ``Environment=``, so that unit may serve another home.
``systemctl show`` joins the arguments with spaces and systemd expands ``$`` in them when it runs the command, so
the planner's ownership check (shared with ``hermes update``'s old-backend notice) does not try to read the
values on that line. A home variable set or unset there, or an option that changes the environment as a whole,
makes the owner unknown: the dry run refuses rather than claim the unit. A line that sets only other variables
leaves the home to the unit's environment, as on main. Follow-up to #157403 and #157484.
"""

from __future__ import annotations

import shlex
import subprocess
import types

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli.gateway_migrate_backends import BackendItem
from tests.hermes_cli.gateway_migrate_backends_apply.test_launcher_templates import (
    _LAUNCHER_ONLY,
    TemplateHost,
    _dotenv,
    _template_blockers,
)
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost, _plan

_HERMES = "/venv/bin/python -m hermes_cli.main dashboard --no-open --host 127.0.0.1 --port 9119"
_TEMPLATE = "HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=${SOMEVAR}\n"


def _exec_start(argv: list[str]) -> str:
    """``systemctl show -p ExecStart``'s value for a unit whose ExecStart has the arguments *argv*. systemd prints
    them joined by spaces and unquoted (measured on systemd 255): ``"A=1 b"`` shows as ``A=1 b``."""
    return f"{{ path=/usr/bin/env ; argv[]={' '.join(argv)} ; ignore_errors=no ; start_time=[n/a] ; pid=0 ; code=(null) }}"


class StoppedUserUnitHost(FakeHost):
    """A stopped user unit the report found by name: its environment is *environment*; its ExecStart is *argv*,
    a command line as written in the unit file (split like systemd does, for these quotes)."""

    def __init__(self, home, argv, *, environment=None):
        super().__init__(home)
        self.old["active"] = False
        self.exact = shlex.split(argv)
        self.environment = dict(environment or {})

    def backends(self):
        return [BackendItem(supervisor="systemd", running=False, unit="hermes-dashboard.service", scope="user",
                            state="enabled", hermes_home=self.environment.get("HERMES_HOME"))]

    def unit_environment(self, unit, scope):
        return dict(self.environment), None

    def unit_launcher_env(self, unit, scope):
        return dict(self.environment)

    def unit_props(self, unit, scope):
        return {**super().unit_props(unit, scope), "ExecStart": _exec_start(self.exact)}

    def default_home(self, env=None):
        return "HOME" not in (env or {})  # a unit that keeps the account's HOME runs this (default) home


def _owner_blocker(out):
    (blocker,) = [b for b in out["plan"]["blockers"] if b.startswith("Cannot tell which Hermes home")]
    return blocker


@pytest.mark.parametrize(("argv", "why"), [
    (f"/usr/bin/env HERMES_HOME=/srv/other {_HERMES}", "sets HERMES_HOME"),  # the original report
    (f"/usr/bin/env HERMES_HOME={{home}} {_HERMES}", "sets HERMES_HOME"),  # this home: still not read (conservative)
    (f"/usr/bin/env 'HERMES_HOME={{home}} Archive' {_HERMES}", "sets HERMES_HOME"),  # a space inside the argument
    (f"/usr/bin/env HERMES_HOME=/srv/other${{SUFFIX}}/.hermes {_HERMES}", "sets HERMES_HOME"),  # expanded by systemd
    (f"/usr/bin/env -- A=1 HERMES_HOME=/srv/other {_HERMES}", "sets HERMES_HOME"),
    (f"/usr/bin/env A=1 /usr/bin/env HERMES_HOME=/srv/other {_HERMES}", "sets HERMES_HOME"),
    (f"/usr/bin/env HOME=/home/other {_HERMES}", "sets HOME"),
    (f"/usr/bin/env -u HERMES_HOME {_HERMES}", "unsets HERMES_HOME"),
    (f"/usr/bin/env -uHERMES_HOME {_HERMES}", "unsets HERMES_HOME"),
    (f"/usr/bin/env --unset=HOME {_HERMES}", "unsets HOME"),
    (f"/usr/bin/env -i PATH=/usr/bin {_HERMES}", "with -i"),
    (f"/usr/bin/env - PATH=/usr/bin {_HERMES}", "with -,"),
    (f"/usr/bin/env --ignore-environment {_HERMES}", "with --ignore-environment"),
    (f"/usr/bin/env -S 'A=1 {_HERMES}'", "with -S"),
    (f"/usr/bin/env '--split-string=A=1' {_HERMES}", "with --split-string=A=1"),
    (f"/usr/bin/env -vi {_HERMES}", "with -vi"),
    (f"/usr/bin/env --argv0=x {_HERMES}", "with --argv0=x"),
    (f"/usr/bin/env $PAIRS {_HERMES}", "$ substitution"),  # systemd may split $PAIRS into assignments
    # a $ in an option or its operand: `Environment=DROP=HERMES_HOME` makes `-u ${DROP}` unset the home
    (f"/bin/env -u ${{DROP}} {_HERMES}", "$ substitution"),
    (f"/bin/env -u $DROP {_HERMES}", "$ substitution"),
    (f"/bin/env -u${{DROP}} {_HERMES}", "$ substitution"),
    (f"/bin/env --unset=${{DROP}} {_HERMES}", "$ substitution"),
    (f"/bin/env -C ${{DIR}} {_HERMES}", "$ substitution"),
    (f"/usr/bin/env ${{N}}=x {_HERMES}", "$ substitution"),
])
def test_an_env_line_that_may_set_the_home_makes_the_owner_unknown(home, capsys, argv, why):
    host = StoppedUserUnitHost(home, argv.replace("{home}", str(home)), environment={"HERMES_HOME": str(home)})

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED and out["steps"] == []
    blocker = _owner_blocker(out)
    assert why in blocker and "hermes-dashboard.service" in blocker
    assert "Environment= rather than on the env line" in blocker
    assert "systemctl --user disable --now hermes-dashboard.service" in blocker


@pytest.mark.parametrize("argv", [
    f"/usr/bin/env TRACE_ID=${{INVOCATION_ID}} {_HERMES}",  # sets another variable from systemd's own
    f"/usr/bin/env PYTHONUNBUFFERED=1 A=b {_HERMES}",
    f"/usr/bin/env -u PYTHONPATH -C /srv LANG=C.UTF-8 {_HERMES}",
    f"/bin/env -u PYTHONWARNINGS A=${{B}}x {_HERMES}",  # a literal -u name; $ only inside another value
    f"/usr/bin/env {_HERMES}",  # no assignment at all
    f"/usr/bin/env -v -- {_HERMES}",
    f"env -0 /usr/bin/env {_HERMES}",
    _HERMES,  # no launcher
])
def test_an_env_line_that_sets_only_other_variables_is_judged_by_the_units_environment(home, capsys, argv):
    ours = StoppedUserUnitHost(home, argv, environment={"HERMES_HOME": str(home)})
    other = StoppedUserUnitHost(home, argv, environment={"HERMES_HOME": "/srv/other"})
    default = StoppedUserUnitHost(home, argv)  # no HERMES_HOME: the account's default home, this one here

    assert _plan(ours, capsys)[0] == engine.EXIT_OK and _plan_steps(ours, capsys)
    code, out = _plan(other, capsys)
    assert code == engine.EXIT_OK and out["plan"]["nothing_to_do"] and out["steps"] == []
    assert _plan_steps(default, capsys)


def _plan_steps(host, capsys):
    code, out = _plan(host, capsys)
    assert code == engine.EXIT_OK and not out["plan"]["blockers"]
    return out["steps"]


# --- a .env sign-in template that reads a variable the env line sets: its value there is not read


def _env_line_host(home, argv, *, gateway_env):
    host = StoppedUserUnitHost(home, argv, environment={"HERMES_HOME": str(home), "COMMON": "sample",
                                                         "SOMEVAR": "sample"})
    host.gateway_environment = lambda: (dict(gateway_env), None)
    return host


def test_a_template_reading_a_variable_the_env_line_sets_is_refused(home, capsys):
    """`ExecStart=/usr/bin/env SOMEVAR=${COMMON} ...`: systemd runs it with SOMEVAR=sample, the gateway's value
    too, but the planner does not read values on that line, so it refuses (a known conservative refusal)."""
    _dotenv(home, _TEMPLATE)
    host = _env_line_host(home, f"/usr/bin/env SOMEVAR=${{COMMON}} {_HERMES}", gateway_env={"SOMEVAR": "sample"})

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED and out["steps"] == []
    (blocker,) = _template_blockers(out)
    assert "SOMEVAR is set on the env line of the old backend's ExecStart" in blocker


@pytest.mark.parametrize(("dotenv", "argv"), [
    (_TEMPLATE, f"/usr/bin/env TRACE_ID=${{INVOCATION_ID}} {_HERMES}"),  # the line sets another variable
    (_TEMPLATE, _HERMES),  # no launcher: compared as before
    ("HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=plain-pw\n",
     f"/usr/bin/env SOMEVAR=${{COMMON}} {_HERMES}"),  # a literal .env reads nothing
])
def test_a_template_not_reading_the_env_lines_variables_is_compared_as_before(home, capsys, dotenv, argv):
    _dotenv(home, dotenv)
    host = _env_line_host(home, argv, gateway_env={"SOMEVAR": "sample"})

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_OK and out["steps"] and not out["plan"]["blockers"]


@pytest.mark.parametrize("argv", [
    # systemd's joined line hides argument boundaries: a -u or -C operand may be other than it looks
    f"/bin/env -C '/srv/build -u AUX' {_HERMES}",  # a directory named "build -u AUX": AUX stays set
    f"/bin/env -u 'AUX ' {_HERMES}",  # unsets "AUX ", not AUX
    f"/bin/env -u AUX {_HERMES}",  # really unsets AUX: still refused (conservative)
    f"/bin/env --unset=AUX {_HERMES}",
    f"/bin/env -uAUX {_HERMES}",
    f"/bin/env -C /srv {_HERMES}",
    f"/bin/env --chdir=/srv {_HERMES}",
])
def test_a_template_with_an_env_line_that_unsets_or_changes_directory_is_refused(home, capsys, argv):
    """`Environment=AUX=unit`, `.env` `OIDC_CLIENT_ID=${AUX:-fallback}`: whether the old backend saw `unit` or
    `fallback` depends on argument boundaries `systemctl show` does not keep, so the value is unknown."""
    _dotenv(home, "HERMES_DASHBOARD_OIDC_CLIENT_ID=${AUX:-fallback}\n")
    host = StoppedUserUnitHost(home, argv, environment={"HERMES_HOME": str(home), "AUX": "unit"})
    host.gateway_environment = lambda: ({"AUX": "unit"}, None)  # the same value, were the line readable

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED and out["steps"] == []
    (blocker,) = _template_blockers(out)
    assert "changes the environment in a way this check cannot read" in blocker
    assert "Put the literal value in" in blocker


@pytest.mark.parametrize(("argv", "dotenv"), [
    (f"/bin/env -u AUX {_HERMES}", "HERMES_DASHBOARD_OIDC_CLIENT_ID=literal-id\n"),  # a literal .env reads nothing
    (f"/bin/env TRACE_ID=${{INVOCATION_ID}} {_HERMES}", "HERMES_DASHBOARD_OIDC_CLIENT_ID=${AUX:-fallback}\n"),
    (_HERMES, "HERMES_DASHBOARD_OIDC_CLIENT_ID=${AUX:-fallback}\n"),  # no launcher
])
def test_a_template_without_an_unreadable_env_line_is_compared_as_before(home, capsys, argv, dotenv):
    _dotenv(home, dotenv)
    host = StoppedUserUnitHost(home, argv, environment={"HERMES_HOME": str(home), "AUX": "unit"})
    host.gateway_environment = lambda: ({"AUX": "unit"}, None)

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_OK and out["steps"] and not out["plan"]["blockers"]


@pytest.mark.parametrize(("line", "code"), [("SOMEVAR=other", engine.EXIT_BLOCKED), ("", engine.EXIT_UNSUPPORTED),
                                            ("-u SOMEVAR", engine.EXIT_BLOCKED)])
def test_a_by_hand_move_refuses_a_template_reading_the_env_lines_variable(home, capsys, line, code):
    """A system unit (moved by hand): the gateway has the unit's SOMEVAR, so only the env line can change it
    (`-u SOMEVAR`: the old backend resolved ${SOMEVAR} blank, the gateway would not)."""
    _dotenv(home, _TEMPLATE)
    host = TemplateHost(home, supervisor="systemd", scope="system", gateway_env={"SOMEVAR": _LAUNCHER_ONLY})
    props = host.unit_props
    host.unit_props = lambda unit, scope: {**props(unit, scope),
                                           "ExecStart": _exec_start(shlex.split(f"/usr/bin/env {line} {_HERMES}"))}

    got, out = _plan(host, capsys)

    assert got == code and bool(_template_blockers(out)) is (code == engine.EXIT_BLOCKED)


# --- the update notice shares the check


def _notice_setup(tmp_path, monkeypatch, units: dict[str, tuple[str, str]]):
    """*units*: name -> (Environment= line, ExecStart). Returns (root, items by name)."""
    import pwd  # POSIX only, like these tests

    import hermes_cli.gateway_migrate_backends_host as backends_host
    from tests.hermes_cli.test_update_old_backend_notice import _standard_root

    root = _standard_root(tmp_path, monkeypatch)
    # The account's home is tmp_path, so a unit that names no home of its own would run this one.
    monkeypatch.setattr(pwd, "getpwnam", lambda _name: types.SimpleNamespace(pw_dir=str(tmp_path)))

    def fake_systemctl(scope, args, **kwargs):
        if args[:1] == ["show-environment"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[:1] == ["show"] and args[1] in units:
            environment, argv = units[args[1]]
            shown = (f"Environment={environment.replace('{root}', str(root))}\nEnvironmentFiles=\n"
                     f"ExecStart={_exec_start(shlex.split(argv.replace('{root}', str(root))))}\n")
            return subprocess.CompletedProcess(args, 0, stdout=shown, stderr="")
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="not found")

    monkeypatch.setattr(backends_host, "_systemctl", fake_systemctl)
    return root, {name: BackendItem("systemd", False, unit=name, scope="user", state="enabled") for name in units}


@pytest.mark.platforms("posix")  # the fake systemd reads POSIX unit paths
def test_the_update_notice_names_neither_a_foreign_nor_an_unknown_env_line_unit(tmp_path, monkeypatch):
    import hermes_cli.gateway_migrate_backends as backends
    from hermes_cli.update_backend_notice import old_backend_notice

    root, items = _notice_setup(tmp_path, monkeypatch, {
        "hermes-dashboard-other.service": ("", f"/usr/bin/env HERMES_HOME=/srv/other {_HERMES}"),
        "hermes-dashboard-trace.service": ("HERMES_HOME={root}", f"/usr/bin/env TRACE_ID=${{INVOCATION_ID}} {_HERMES}"),
        "hermes-dashboard-plain.service": ("HERMES_HOME={root}", f"/usr/bin/env {_HERMES}"),
        "hermes-dashboard-home.service": ("", f"/usr/bin/env HERMES_HOME={{root}} {_HERMES}"),
    })

    def notice_for(name):
        monkeypatch.setattr(backends, "collect_backends", lambda: [items[name]])
        return old_backend_notice(root)

    assert notice_for("hermes-dashboard-other.service") is None
    assert notice_for("hermes-dashboard-home.service") is None  # unknown: never claimed
    assert "hermes-dashboard-trace.service" in notice_for("hermes-dashboard-trace.service")
    assert "hermes-dashboard-plain.service" in notice_for("hermes-dashboard-plain.service")
