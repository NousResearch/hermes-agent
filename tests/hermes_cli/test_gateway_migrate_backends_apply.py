"""``hermes gateway migrate --backends --apply --dry-run``: the plan and preflight of the backend migration.

A fake host stands in for the service manager, the process table and the network (the gateway's user
service, the old backend's unit, ``/api/health``); config.yaml and .env are the real files in a temp
``HERMES_HOME``, and the bind check is the gateway's own ``webapp.host`` validation. A dry run changes
nothing: no file in the home is written or added. (The real command on a scratch machine:
``test_gateway_migrate_backends_dry_run_readonly.py``.)
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli.gateway_migrate_backends import BackendItem
from hermes_cli.gateway_migrate_backends_host import SystemHost
from hermes_cli.gateway_migrate_backends_plan import WEBAPP_STATUS_WAIT


@pytest.fixture(autouse=True)
def _standard_root(monkeypatch):
    """The homes here stand for the user's standard Hermes root, so steps say `hermes -p NAME`; the root
    pin for any other root is tested in test_gateway_migrate_backends_root_pin.py."""
    monkeypatch.setattr("hermes_cli.gateway_migrate_backends_plan._is_standard_root", lambda root: True)

_PASSWORD = "pw-never-printed-7d1c"
_SECRET = "secret-never-printed-91aa"
_BACKEND_PID = 4242
_GATEWAY_PID = 5151


class FakeHost(SystemHost):
    """The machine as the plan reads it: one old backend (a systemd unit by default), the gateway's user
    service and its ``GET /api/health`` read. Reads only, like :class:`SystemHost`; config.yaml and .env are real."""

    def __init__(self, home, *, host="0.0.0.0", supervisor="systemd", scope="user", gateway_installed=False,
                 gateway_active=False):
        super().__init__(home)
        self.bind = host
        self.probes: list[tuple[str, int]] = []
        self.argv = ["/venv/bin/python", "-m", "hermes_cli.main", "dashboard", "--no-open", "--host", host,
                     "--port", "9119"]
        self.supervisor, self.scope = supervisor, scope
        self.old = {"enabled": True, "active": True}
        self.gw = {"installed": gateway_installed, "enabled": gateway_installed, "active": gateway_active}

    # --- discovery
    def backends(self):
        return [BackendItem(supervisor=self.supervisor, running=self.old["active"],
                            pid=_BACKEND_PID if self.old["active"] else None, argv=list(self.argv),
                            unit="hermes-dashboard.service", scope=self.scope,
                            unit_path="/home/u/.config/systemd/user/hermes-dashboard.service",
                            state="enabled" if self.old["enabled"] else "disabled", hermes_home=str(self.home),
                            this_home=True)]

    def unit_launcher_env(self, unit, scope):
        return self.process_env(_BACKEND_PID)

    def unit_environment(self, unit, scope):
        return self.unit_launcher_env(unit, scope), None

    def os_family(self):
        return "linux"

    def username(self):
        return "u"

    def unit_props(self, unit, scope):
        return {"UnitFileState": "enabled" if self.old["enabled"] else "disabled",
                "ActiveState": "active" if self.old["active"] else "inactive", "ControlGroup": f"/fake/{unit}"}

    def process_env(self, pid):
        return {"PATH": "/usr/bin", "HERMES_DASHBOARD_BASIC_AUTH_USERNAME": "spike",
                "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": _PASSWORD, "HERMES_DASHBOARD_BASIC_AUTH_SECRET": _SECRET}

    def process_cwd(self, pid):
        return "/"

    def start_time(self, pid):
        return 1000 + pid

    turns = 0

    def running_turns(self, pid, start_time):
        return self.turns

    def bundle_missing(self):
        return None

    # --- gateway
    def gateway_kind(self):
        return "systemd"

    def gateway_state(self):
        return {"kind": "systemd", "unit": "hermes-gateway", **self.gw,
                "running_pid": _GATEWAY_PID if self.gw["active"] else None}

    def gateway_preflight(self):
        return None

    def linger(self):
        return False

    # --- network
    def http_status(self, host, port):
        """The old backend answers where it is bound (a wildcard bind answers every address)."""
        self.probes.append((host, port))
        if self.old["active"] and (self.bind in ("0.0.0.0", "::") or host == self.bind):
            return {"gateway_state": None, "auth_required": True, "auth_providers": ["basic"]}
        return None


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes-home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    (h / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    (h / ".env").write_text("OPENAI_API_KEY=sk-fake-local\n", encoding="utf-8")
    os.chmod(h / ".env", 0o600)
    return h


def _args(**kw):
    base = dict(backends=True, apply=True, dry_run=False, yes=False, json=False, address=None,
                keep_webapp_address=False, force_interrupt=False, multiplex=False)
    return SimpleNamespace(**{**base, **kw})


def _files(home):
    """Every path under *home* with its bytes (None for a directory): a dry run adds, edits or removes none."""
    return {str(p.relative_to(home)): (p.read_bytes() if p.is_file() else None) for p in sorted(home.rglob("*"))}


def _nothing_changed(home, before):
    assert _files(home) == before and not (home / "backend_migration.json").exists()


def _plan(host, capsys, **kw):
    code = engine.cmd_apply(host, _args(dry_run=True, json=True, **kw))
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(("bind", "expected"), [("0.0.0.0", "0.0.0.0:9119"), ("127.0.0.1", "127.0.0.1:9119"),
                                                ("192.168.7.20", "192.168.7.20:9119"), ("example.lan", None)])
def test_the_plan_never_widens_exposure(home, capsys, bind, expected):
    host = FakeHost(home, host=bind)
    before = _files(home)

    code, out = _plan(host, capsys)

    _nothing_changed(home, before)
    if expected is None:  # a bind the gateway cannot take: refused
        assert code == engine.EXIT_BLOCKED and out["plan"]["blockers"]
    else:
        assert code == engine.EXIT_OK and out["plan"]["new_address"] == expected


def test_a_specific_ip_backend_is_probed_where_it_listens_and_planned_on_ip_and_loopback(home, capsys):
    host = FakeHost(home, host="192.168.7.20")  # the old backend answers on that IP only

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_OK and ("192.168.7.20", 9119) in host.probes
    verify = next(s["title"] for s in out["steps"] if s["name"] == "verify")
    assert "192.168.7.20:9119" in verify and "127.0.0.1:9119" in verify


def test_the_plan_json_lists_the_steps_by_name(home, capsys):
    from hermes_cli.gateway_migrate_backends_plan import APPLY_STEPS

    code, out = _plan(FakeHost(home), capsys)

    assert code == engine.EXIT_OK and out["dry_run"] is True and out["applied"] is False
    assert out["steps"] == out["plan"]["steps"]
    assert [s["name"] for s in out["steps"]] == list(APPLY_STEPS) and all(s["title"] for s in out["steps"])


@pytest.mark.parametrize("as_json", [False, True])
def test_the_plan_never_prints_a_secret_and_names_the_keys_it_would_copy(home, capsys, as_json):
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args(dry_run=True, json=as_json)) == engine.EXIT_OK
    out = capsys.readouterr().out
    assert _PASSWORD not in out and _SECRET not in out
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in out


def test_a_running_turn_blocks_unless_force_interrupt(home, capsys):
    host = FakeHost(home)
    host.turns = 2
    before = _files(home)

    code, out = _plan(host, capsys)
    assert code == engine.EXIT_BLOCKED and any("--force-interrupt" in b for b in out["plan"]["blockers"])
    code, out = _plan(host, capsys, force_interrupt=True)
    assert code == engine.EXIT_OK and out["plan"]["running_turns"] == 2
    _nothing_changed(home, before)


def test_an_address_that_matches_no_backend_blocks(home, capsys):
    code, out = _plan(FakeHost(home), capsys, address="127.0.0.1:9999")
    assert code == engine.EXIT_BLOCKED and out["plan"]["blockers"]


def test_an_explicit_address_needs_a_backend_even_if_the_inventory_is_empty(home, capsys, monkeypatch):
    host = FakeHost(home)
    monkeypatch.setattr(host, "backends", list)
    assert _plan(host, capsys)[0] == engine.EXIT_OK
    code, out = _plan(host, capsys, address="127.0.0.1:9999")
    assert code == engine.EXIT_BLOCKED and out["plan"]["blockers"]


def test_mixed_supervisors_require_a_unique_selection(home, capsys, monkeypatch):
    host = FakeHost(home, host="127.0.0.1")
    # 0.0.0.0, not another loopback address: the gateway refuses 127.0.0.2 as webapp.host.
    items = host.backends() + FakeHost(home, host="0.0.0.0", supervisor="container").backends()
    monkeypatch.setattr(host, "backends", lambda: items)
    code, out = _plan(host, capsys)
    assert code == engine.EXIT_BLOCKED and out["plan"]["unsupported"] == []
    assert _plan(host, capsys, address="127.0.0.1:9119")[0] == engine.EXIT_OK
    assert _plan(host, capsys, address="0.0.0.0:9119")[0] == engine.EXIT_UNSUPPORTED


@pytest.mark.parametrize(("supervisor", "scope"), [("nix-systemd", "system"), ("nix-launchd", "gui/501"), ("s6", None),
                                                   ("container", None), ("systemd", "system")])
def test_unsupported_supervisors_exit_6_and_change_nothing(home, supervisor, scope, capsys):
    host = FakeHost(home, supervisor=supervisor, scope=scope)
    before = _files(home)

    code = engine.cmd_apply(host, _args(dry_run=True))

    assert code == engine.EXIT_UNSUPPORTED
    _nothing_changed(home, before)
    out = capsys.readouterr().out
    assert "webapp.enabled" in out and "webapp.host" in out and "webapp.port" in out
    declared = supervisor in ("nix-systemd", "nix-launchd", "s6", "container")
    # Nix refuses `hermes config set`, and a container image ships the Webapp built: no host commands there.
    assert (" config set webapp" in out) is not declared and (" gateway install" in out) is not declared


def test_a_system_unit_is_exit_6_with_user_steps_and_one_sudo_line(home, capsys):
    host = FakeHost(home, supervisor="systemd", scope="system")

    code = engine.cmd_apply(host, _args(dry_run=True))

    assert code == engine.EXIT_UNSUPPORTED
    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    steps = lines[lines.index("Steps, in this order (one command per line):") + 1:]
    sudo = [line for line in lines if line.startswith("sudo ")]
    assert sudo == ["sudo systemctl disable --now hermes-dashboard.service"]
    # The old unit is stopped and shown inactive before anything can start the gateway on its port.
    after = steps[steps.index(sudo[0]) + 1:]
    assert after[0].startswith("systemctl is-active hermes-dashboard.service")
    h = "hermes -p default"  # never a bare `hermes`: it would follow a sticky `hermes profile use` choice
    assert after[1:] == [f"{h} gateway status",
                         f"# then by what it prints: 'Gateway is not running' (no service installed) -> {h} gateway "
                         f"install; 'service is stopped', 'service is not loaded' or 'registered with launchd' "
                         f"-> {h} gateway start; 'service is running', 'supervised by launchd' or 'Running "
                         f"manually' -> {h} gateway restart",
                         WEBAPP_STATUS_WAIT, f"{h} webapp --status"]
    before_stop = steps[:steps.index(sudo[0])]
    assert not any(" gateway " in s for s in before_stop)
    assert all(s.startswith(("hermes -p default ", "sudo ", "systemctl ", "#")) for s in steps)
    assert not any("sudo hermes" in line or "sudo -E" in line or "systemctl --user" in line for line in lines)
    assert any("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in s and _PASSWORD not in s for s in steps)


def test_a_run_from_inside_the_old_unit_is_blocked_without_pointing_at_a_flag_this_version_lacks(home, capsys,
                                                                                             monkeypatch):
    monkeypatch.setattr("hermes_cli.update_cmd_posix_pause._pid_cgroup",
                        lambda pid="self": "/fake/hermes-dashboard.service/shell")

    code, out = _plan(FakeHost(home), capsys)

    (blocker,) = out["plan"]["blockers"]
    assert code == engine.EXIT_BLOCKED and "runs inside hermes-dashboard.service" in blocker
    assert "--detach" not in blocker


@pytest.mark.parametrize(("flags", "message"), [({}, "--apply needs --dry-run"),
                                                ({"dry_run": True, "address": "nope"}, "--address must be HOST:PORT")])
def test_the_cli_refuses_what_this_version_cannot_do(flags, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        engine.cmd_backends(_args(**flags))
    assert exit_info.value.code == 2 and message in capsys.readouterr().err


def _everything(host, capsys, **kw):
    """The plan's text and JSON output of one dry run each."""
    engine.cmd_apply(host, _args(dry_run=True, **kw))
    text = capsys.readouterr().out
    engine.cmd_apply(host, _args(dry_run=True, json=True, **kw))
    return text + capsys.readouterr().out


@pytest.mark.parametrize("where", ["config", "env"])
def test_a_public_url_the_gateway_refuses_blocks_without_printing_it(home, capsys, monkeypatch, where):
    bad = "https://admin:pw-in-url-55@dash.example/"
    host = FakeHost(home)
    if where == "config":
        (home / "config.yaml").write_text(f"dashboard:\n  public_url: {bad}\n", encoding="utf-8")
    else:
        env = host.process_env(_BACKEND_PID)
        monkeypatch.setattr(host, "process_env", lambda pid: {**env, "HERMES_DASHBOARD_PUBLIC_URL": bad})

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED and out["plan"]["public_url"] is None
    assert any("not one the gateway accepts as webapp.public_url" in b for b in out["plan"]["blockers"])
    assert "pw-in-url-55" not in _everything(host, capsys)


def test_an_accepted_public_url_is_copied(home, capsys):
    (home / "config.yaml").write_text("dashboard:\n  public_url: https://dash.example/\n", encoding="utf-8")

    code, out = _plan(FakeHost(home), capsys)

    assert code == engine.EXIT_OK and out["plan"]["public_url"] == "https://dash.example"


def test_no_secret_reaches_the_plan_from_the_argv_the_env_or_a_url(home, capsys):
    host = FakeHost(home)
    host.argv = ["/usr/bin/env", "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=argv-pw-77", "/venv/bin/python", "-m",
                 "hermes_cli.main", "dashboard", "--no-open", "--port", "9119", "--token", "argv-token-88",
                 "--api-key=argv-key-99", "--webhook", "https://u:url-pw-66@hook.example/x"]

    out = _everything(host, capsys)

    for secret in (_PASSWORD, _SECRET, "argv-pw-77", "argv-token-88", "argv-key-99", "url-pw-66"):
        assert secret not in out
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=<redacted>" in out and '"--port"' in out


def test_query_string_credentials_and_inline_c_source_are_redacted_in_json_and_text(home, capsys):
    host = FakeHost(home)
    host.argv = ["/venv/bin/python", "-m", "hermes_cli.main", "dashboard", "--no-open", "--port", "9119",
                 "--webhook", "https://hook.example/x?api_key=query-secret-44&mode=1",
                 "-c", "import os; SECRET='inline-secret-55'"]

    assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_OK
    text = capsys.readouterr().out
    assert "query-secret-44" not in text and "inline-secret-55" not in text

    code, out = _plan(host, capsys)
    assert code == engine.EXIT_OK
    rendered = json.dumps(out)
    assert "query-secret-44" not in rendered and "inline-secret-55" not in rendered
    argv = out["plan"]["old_backend"]["argv"]
    assert "<inline launcher source>" in argv
    assert any("https://hook.example/x?api_key=" in tok and "mode=1" in tok for tok in argv)


@pytest.mark.parametrize("joined", [True, False])
def test_flag_url_query_credential_is_redacted_without_clobbering_next_flag(home, joined):
    from hermes_cli.gateway_migrate_backends_plan import build_plan, format_plan
    host = FakeHost(home)
    url = "https://example.invalid/x?api_key=sample44"
    webhook = [f"--webhook={url}"] if joined else ["--webhook", url]
    host.argv = ["/venv/bin/python", "-m", "hermes_cli.main", "dashboard", "--no-open", *webhook, "--port", "9119"]

    plan = build_plan(host)
    data = plan.to_dict()
    text = "\n".join(format_plan(plan, dry_run=True))

    assert "sample44" not in json.dumps(data) and "sample44" not in text
    argv = data["old_backend"]["argv"]
    port_idx = argv.index("--port")
    assert argv[port_idx + 1] == "9119"


def test_ordinary_arguments_are_printed_as_they_are():
    from hermes_cli.gateway_migrate_backends_plan import redacted_argv
    argv = ["/venv/bin/python", "-m", "hermes_cli.main", "dashboard", "--host", "0.0.0.0", "--port", "9119",
            "--config=/srv/a=b/c?d.yaml", "/opt/x=y/run?z", "--root", "C:\\a=b\\q?", "--insecure"]
    assert redacted_argv(argv) == argv


@pytest.mark.parametrize("exe", ["/venv/bin/python", "C:\\venv\\Scripts\\pythonw.exe"])
def test_combined_python_short_c_flag_redacts_inline_source(home, exe):
    from hermes_cli.gateway_migrate_backends_plan import build_plan, format_plan
    host = FakeHost(home)
    host.argv = [exe, "-Ic", "import os; SECRET='inline-ic-66'", "hermes_cli.main",
                 "dashboard", "--no-open", "--port", "9119"]

    plan = build_plan(host)
    data = plan.to_dict()
    text = "\n".join(format_plan(plan, dry_run=True))

    assert "inline-ic-66" not in json.dumps(data) and "inline-ic-66" not in text
    argv = data["old_backend"]["argv"]
    ic_idx = argv.index("-Ic")
    assert argv[ic_idx + 1] == "<inline launcher source>"


def test_without_a_service_manager_the_steps_run_the_gateway_under_the_operators_supervisor(home, capsys,
                                                                                           monkeypatch):
    host = FakeHost(home, host="127.0.0.1")
    monkeypatch.setattr(host, "gateway_state", lambda: {"kind": None, "installed": False, "enabled": False,
                                                        "active": False, "running_pid": None, "unit": None})

    code, out = _plan(host, capsys)

    (entry,) = out["plan"]["unsupported"]
    assert code == engine.EXIT_UNSUPPORTED and entry["steps"][-3:] == ["hermes -p default gateway run",
                                                                       WEBAPP_STATUS_WAIT,
                                                                       "hermes -p default webapp --status"]
    assert not any("gateway install" in s or "gateway start" in s for s in entry["steps"])


def test_a_profiles_steps_name_the_profile(tmp_path, monkeypatch, capsys):
    profile = tmp_path / ".hermes" / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    # A standalone profile has a gateway of its own; one served by the default gateway is blocked (below).
    (profile / "config.yaml").write_text("model:\n  default: fake-model\ngateway:\n  standalone: true\n",
                                         encoding="utf-8")

    engine.cmd_apply(FakeHost(profile, supervisor="systemd", scope="system"), _args(dry_run=True))

    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    steps = lines[lines.index("Steps, in this order (one command per line):") + 1:]
    assert [s for s in steps if s.startswith("hermes -p work ")]
    assert not any(s.startswith("hermes ") and not s.startswith("hermes -p work ") for s in steps)
    assert all("hermes -p work gateway" in s for s in steps if s.startswith("# then"))


class StoppedUnitHost(FakeHost):
    """A stopped system unit the report found by name: no pid, no argv, no HERMES_HOME of its own."""

    def __init__(self, home, *, supervisor="systemd", user="", exec_start=None, environment=None, env_problem=None,
                 this_home=None):
        super().__init__(home, supervisor=supervisor, scope="system")
        self.old["active"] = False
        self.unit_user, self.exec_start, self.environment = user, exec_start, environment or {}
        self.env_problem, self.report_this_home = env_problem, this_home

    def unit_environment(self, unit, scope):
        return dict(self.environment), self.env_problem

    def backends(self):
        return [BackendItem(supervisor=self.supervisor, running=False, unit="hermes-dashboard.service",
                            scope="system", state="enabled", this_home=self.report_this_home)]

    def unit_props(self, unit, scope):
        return {**super().unit_props(unit, scope), "User": self.unit_user,
                **({"ExecStart": self.exec_start} if self.exec_start else {})}

    def unit_launcher_env(self, unit, scope):
        return dict(self.environment)

    def default_home(self, env=None):
        return True


@pytest.mark.parametrize(("user", "mine"), [("", False), ("someone-else", False), ("u", True)])
def test_a_stopped_unpinned_system_unit_belongs_to_this_home_only_when_it_runs_as_this_user(home, capsys,
                                                                                            user, mine):
    code, out = _plan(StoppedUnitHost(home, user=user), capsys)

    assert code == (engine.EXIT_UNSUPPORTED if mine else engine.EXIT_OK)
    assert out["plan"]["nothing_to_do"] is not mine
    assert any(f"runs as {user or 'root'}" in n for n in out["plan"]["notices"]) is not mine


def test_a_stopped_unit_pinned_to_another_home_by_an_environment_file_is_not_this_homes(home, capsys):
    code, out = _plan(StoppedUnitHost(home, user="u", environment={"HERMES_HOME": "/srv/other-home"}), capsys)

    assert code == engine.EXIT_OK and out["plan"]["nothing_to_do"]


def test_a_stopped_nix_unit_is_planned_on_the_address_in_its_exec_start(home, capsys):
    host = StoppedUnitHost(home, supervisor="nix-systemd", user="u", exec_start=(
        "{ path=/nix/store/x-hermes/bin/hermes ; argv[]=/nix/store/x-hermes/bin/hermes dashboard --no-open "
        "--host 0.0.0.0 --port 9200 ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) }"))

    code, out = _plan(host, capsys)

    (entry,) = out["plan"]["unsupported"]
    assert code == engine.EXIT_UNSUPPORTED and "webapp.host: 0.0.0.0, webapp.port: 9200" in entry["steps"][0]


def test_a_backend_whose_command_line_cannot_be_read_gets_no_invented_address(home, capsys):
    code, out = _plan(StoppedUnitHost(home, user="u"), capsys)

    (entry,) = out["plan"]["unsupported"]
    assert code == engine.EXIT_UNSUPPORTED and "could not be read" in entry["instructions"]
    assert not any("9119" in s or "127.0.0.1" in s for s in entry["steps"])


# --- round 3: the gateway's effective Webapp settings, systemd's environment precedence, unknown addresses


def _webapp_conflict(home, capsys, monkeypatch, config_webapp, *, env=None, managed=None, keep=False):
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    from hermes_cli import managed_scope
    monkeypatch.setattr(managed_scope, "load_managed_config", lambda **_: dict(managed or {}))
    (home / "config.yaml").write_text("model:\n  default: fake-model\n" + config_webapp, encoding="utf-8")
    return _plan(FakeHost(home, host="127.0.0.1"), capsys, keep_webapp_address=keep)


def test_the_webapp_settings_are_the_gateways_after_var_expansion(home, capsys, monkeypatch):
    webapp = "webapp:\n  enabled: true\n  host: ${WEB_BIND}\n  port: 9200\n"
    env = {"WEB_BIND": "127.0.0.1"}

    code, out = _webapp_conflict(home, capsys, monkeypatch, webapp, env=env)
    assert code == engine.EXIT_BLOCKED
    assert any("already enabled on 127.0.0.1:9200" in b for b in out["plan"]["blockers"])

    code, out = _webapp_conflict(home, capsys, monkeypatch, webapp, env=env, keep=True)
    assert code == engine.EXIT_OK and out["plan"]["new_address"] == "127.0.0.1:9200"


def test_the_webapp_settings_include_the_managed_overlay(home, capsys, monkeypatch):
    managed = {"webapp": {"enabled": True, "host": "127.0.0.1", "port": 9300}}

    code, out = _webapp_conflict(home, capsys, monkeypatch, "", managed=managed)

    assert code == engine.EXIT_BLOCKED
    assert any("already enabled on 127.0.0.1:9300" in b for b in out["plan"]["blockers"])


def test_unit_environment_follows_systemd_precedence(tmp_path, monkeypatch):
    """systemd.exec(5): EnvironmentFile= settings override Environment=, and a later file wins."""
    from hermes_cli import gateway_migrate_backends_host as host_mod
    first, second = tmp_path / "first.env", tmp_path / "second.env"
    first.write_text("HERMES_HOME=/srv/first\nA=1\n", encoding="utf-8")
    second.write_text("HERMES_HOME=/srv/second\n", encoding="utf-8")
    shown = (f"Environment=HERMES_HOME=/srv/inline B=2\n"
             f"EnvironmentFiles={first} (ignore_errors=no)\nEnvironmentFiles={second} (ignore_errors=yes)\n"
             f"EnvironmentFiles={tmp_path}/absent.env (ignore_errors=yes)\n")
    monkeypatch.setattr(host_mod, "_systemctl",
                        lambda scope, args: SimpleNamespace(returncode=0, stdout="" if args == ["show-environment"] else shown,
                                                           stderr=""))

    env, problem = SystemHost(tmp_path).unit_environment("x.service", "system")

    assert problem is None and env == {"HERMES_HOME": "/srv/second", "A": "1", "B": "2"}


@pytest.mark.parametrize("make", ["missing", "unreadable"])
def test_unit_environment_is_unknown_when_a_required_file_cannot_be_read(tmp_path, monkeypatch, make):
    from hermes_cli import gateway_migrate_backends_host as host_mod
    path = tmp_path / "unit.env"
    if make == "unreadable":
        path.write_text("HERMES_HOME=/srv/x\n", encoding="utf-8")
        monkeypatch.setattr(host_mod.Path, "read_bytes",
                            lambda self: (_ for _ in ()).throw(PermissionError(13, "denied")))
    shown = f"Environment=\nEnvironmentFiles={path} (ignore_errors=no)\n"
    monkeypatch.setattr(host_mod, "_systemctl",
                        lambda scope, args: SimpleNamespace(returncode=0, stdout="" if args == ["show-environment"] else shown,
                                                           stderr=""))

    _, problem = SystemHost(tmp_path).unit_environment("x.service", "system")

    assert problem and str(path) in problem


def test_an_environment_file_overrides_the_reports_inline_home_for_a_stopped_unit(home, capsys):
    """The report read only Environment= (this home); the unit's EnvironmentFile= pins another one."""
    host = StoppedUnitHost(home, user="u", environment={"HERMES_HOME": "/srv/other-home"}, this_home=True)

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_OK and out["plan"]["nothing_to_do"]


def test_a_stopped_unit_whose_environment_cannot_be_read_blocks(home, capsys):
    host = StoppedUnitHost(home, user="u", env_problem="its EnvironmentFile=/etc/hermes.env cannot be read by this user",
                           this_home=True)

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED
    assert any("Cannot tell which Hermes home hermes-dashboard.service serves" in b for b in out["plan"]["blockers"])


class UnreadableArgvHost(FakeHost):
    """A running backend this home owns whose command line cannot be read."""

    def backends(self):
        (item,) = super().backends()
        item.argv = []
        return [item]


def test_an_unknown_address_is_absent_not_a_default(home, capsys):
    host = UnreadableArgvHost(home)

    code, out = _plan(host, capsys)
    entry = out["plan"]
    assert code == engine.EXIT_BLOCKED
    assert entry["old_address"] is None and entry["new_address"] is None and entry["steps"] == []
    assert "9119" not in json.dumps(out) and "127.0.0.1" not in json.dumps(out)

    engine.cmd_apply(host, _args(dry_run=True))
    text = capsys.readouterr().out
    assert "9119" not in text and "exposure unchanged" not in text and "Steps:" not in text


def test_default_home_steps_name_the_default_profile(home, capsys, monkeypatch):
    """A sticky `hermes profile use work` would send a bare `hermes` to the work profile."""
    import hermes_constants
    from hermes_cli import gateway_migrate_backends_plan as plan_mod
    monkeypatch.setattr(hermes_constants, "profile_name_for_home", lambda h: "default")

    assert plan_mod.hermes_command(str(home)) == "hermes -p default"


# --- round 4: what cannot be read blocks (exit 3), nothing is guessed


def test_a_glob_environment_file_makes_the_units_home_unknown(tmp_path, monkeypatch):
    """systemd expands an EnvironmentFile= pattern; an optional `-*.env` may override HERMES_HOME."""
    from hermes_cli import gateway_migrate_backends_host as host_mod
    (tmp_path / "override.env").write_text("HERMES_HOME=/srv/other-profile\n", encoding="utf-8")
    shown = f"Environment=HERMES_HOME={tmp_path}\nEnvironmentFiles={tmp_path}/*.env (ignore_errors=yes)\n"
    monkeypatch.setattr(host_mod, "_systemctl",
                        lambda scope, args: SimpleNamespace(returncode=0, stdout="" if args == ["show-environment"] else shown,
                                                           stderr=""))

    _, problem = SystemHost(tmp_path).unit_environment("x.service", "system")

    assert problem and "is a pattern" in problem


def test_a_unit_whose_home_is_unknown_blocks_with_identify_the_owner_guidance(home, capsys):
    host = StoppedUnitHost(home, user="u", this_home=True,
                           env_problem="its EnvironmentFile=/etc/hermes/*.env is a pattern, so the files that "
                                       "set its HERMES_HOME are unknown")

    code, out = _plan(host, capsys)

    (blocker,) = out["plan"]["blockers"]
    assert code == engine.EXIT_BLOCKED and "Identify the owner (systemctl cat hermes-dashboard.service)" in blocker
    assert out["plan"]["steps"] == [] and out["plan"]["old_address"] is None


def test_a_config_yaml_that_cannot_be_parsed_blocks_the_dry_run(home, capsys):
    (home / "config.yaml").write_text("webapp:\n  enabled: true\n  host: [unclosed\napi_key: sk-never-printed\n",
                                      encoding="utf-8")
    before = _files(home)

    code, out = _plan(FakeHost(home), capsys)
    (blocker,) = out["plan"]["blockers"]
    assert code == engine.EXIT_BLOCKED and blocker.startswith(f"{home / 'config.yaml'} cannot be read or parsed (")
    assert blocker.endswith(": fix it and re-run the dry run. Nothing changed.")
    assert "sk-never-printed" not in json.dumps(out)

    assert engine.cmd_apply(FakeHost(home), _args(dry_run=True)) == engine.EXIT_BLOCKED
    text = capsys.readouterr().out
    assert "cannot be read or parsed" in text and "fix it and re-run the dry run" in text
    assert "sk-never-printed" not in text
    _nothing_changed(home, before)


# --- review round 1: the probe reads /api/health (+ /api/auth/providers), never /api/status


class _Answer:
    def __init__(self, status, body):
        self.status, self._body = status, json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *a):
        return self._body


def test_the_probe_reads_health_and_providers_never_status(home, monkeypatch):
    """GET /api/status creates the home's install id and prunes a stale gateway.pid on its first call."""
    import urllib.request
    seen, proxies = [], []
    answers = {"/api/health": {"ok": True, "auth_required": True},
               "/api/auth/providers": {"providers": [{"name": "basic"}]}}

    class Opener:
        def open(self, request, timeout=None):
            path = request.full_url.split(":9119", 1)[1]
            seen.append(path)
            return _Answer(200, answers[path])

    real = urllib.request.build_opener
    monkeypatch.setattr(urllib.request, "build_opener",
                        lambda *handlers: (proxies.extend(handlers), Opener())[1])
    got = SystemHost(home).http_status("127.0.0.1", 9119)
    monkeypatch.setattr(urllib.request, "build_opener", real)

    assert seen == ["/api/health", "/api/auth/providers"] and got["auth_required"] is True
    # No proxy for the on-machine check: an HTTP_PROXY without a loopback exemption would divert it.
    (handler,) = proxies
    assert isinstance(handler, urllib.request.ProxyHandler) and handler.proxies == {}


def test_signin_choices_unknown_when_the_provider_list_cannot_be_read(home, capsys, monkeypatch):
    host = FakeHost(home)
    monkeypatch.setattr(host, "http_status", lambda h, p: {"ok": True, "auth_required": True, "auth_providers": None})

    engine.cmd_apply(host, _args(dry_run=True))

    assert "sign-in:       required (interactive providers: unknown)" in capsys.readouterr().out


def test_a_running_backend_whose_health_probe_fails_is_not_called_stopped(home, capsys, monkeypatch):
    host = FakeHost(home)
    monkeypatch.setattr(host, "http_status", lambda h, p: None)

    assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_BLOCKED

    out = capsys.readouterr().out
    assert "unknown (health probe failed)" in out and "backend not running" not in out


def test_running_turns_count_markers_the_backend_wrote_in_any_profile(tmp_path, monkeypatch):
    """`prompt_turn` writes the marker in the session's own home: a backend launched from the default
    home that works in profiles/work has its turn marker there."""
    from tui_gateway import turn_marker
    root = tmp_path / "root"
    work = root / "profiles" / "work"
    work.mkdir(parents=True)
    (work / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    turn_marker.record_turn_start(work, "session-1", "hello")

    for home in (root, work):
        assert SystemHost(home).running_turns(os.getpid(), None) == 1


@pytest.mark.parametrize("case", ["unreadable", "corrupt_json", "non_dict", "unknown_writer"])
def test_running_turns_returns_none_when_marker_activity_is_unknown(tmp_path, monkeypatch, case):
    """An unreadable or invalid marker file, or a writer whose liveness is unknown, is unknown (None), not 0."""
    from tui_gateway import turn_marker
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = turn_marker._marker_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()

    if case == "unreadable":
        path.write_text("{}", encoding="utf-8")
        path.chmod(0o000)
    elif case == "corrupt_json":
        path.write_text("{not-json", encoding="utf-8")
    elif case == "non_dict":
        path.write_text("[]", encoding="utf-8")
    else:
        turn_marker.record_turn_start(home, "session-1", "hello")
        monkeypatch.setattr("hermes_cli.active_sessions._pid_liveness", lambda p, s: None)

    try:
        assert SystemHost(home).running_turns(pid, None) is None
    finally:
        if case == "unreadable":
            path.chmod(0o600)


def test_unknown_turn_activity_blocks_and_reports_unknown_in_text_and_null_in_json(home, capsys, monkeypatch):
    """A real unreadable marker file keeps running_turns unknown (text 'unknown', JSON null) and blocks unless --force-interrupt."""
    from tui_gateway import turn_marker
    path = turn_marker._marker_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o000)

    host = FakeHost(home)
    monkeypatch.setattr(host, "running_turns", SystemHost.running_turns.__get__(host, SystemHost))
    try:
        assert engine.cmd_apply(host, _args(dry_run=True)) == engine.EXIT_BLOCKED
        text = capsys.readouterr().out
        assert "running turns: unknown" in text and "--force-interrupt" in text

        code, out = _plan(host, capsys)
        assert code == engine.EXIT_BLOCKED and out["plan"]["running_turns"] is None
        assert any("--force-interrupt" in b for b in out["plan"]["blockers"])

        code, out = _plan(host, capsys, force_interrupt=True)
        assert code == engine.EXIT_OK and out["plan"]["running_turns"] is None
    finally:
        path.chmod(0o600)


# --- review round 1: a by-hand move passes the same target checks as an automatic one


class ManualHost(FakeHost):
    """An s6/container backend: the plan prints by-hand steps (exit 6) instead of moving it."""

    def __init__(self, home, **kw):
        super().__init__(home, supervisor="container", **kw)


def _webapp(home, block):
    (home / "config.yaml").write_text("model:\n  default: fake-model\nwebapp:\n" + block, encoding="utf-8")


def test_a_by_hand_move_blocks_when_the_webapp_is_already_enabled_elsewhere(home, capsys):
    _webapp(home, "  enabled: true\n  host: 127.0.0.1\n  port: 9200\n")

    code, out = _plan(ManualHost(home, host="127.0.0.1"), capsys)

    assert code == engine.EXIT_BLOCKED and any("already enabled on 127.0.0.1:9200" in b for b in out["plan"]["blockers"])
    (entry,) = out["plan"]["unsupported"]
    assert entry["steps"] == []  # nothing that would set webapp.port 9119 over the Webapp on 9200


@pytest.mark.parametrize("turns", [2, None])
def test_a_by_hand_move_blocks_on_active_or_unknown_turns_unless_force_interrupt(home, capsys, turns):
    host = ManualHost(home, host="127.0.0.1")
    host.turns = turns

    code, out = _plan(host, capsys)
    assert code == engine.EXIT_BLOCKED and out["plan"]["running_turns"] == turns
    assert any("--force-interrupt" in b for b in out["plan"]["blockers"])
    assert out["plan"]["unsupported"][0]["steps"] == []

    code, out = _plan(host, capsys, force_interrupt=True)
    assert code == engine.EXIT_UNSUPPORTED and out["plan"]["running_turns"] == turns
    assert out["plan"]["unsupported"][0]["steps"]


def test_a_forced_by_hand_move_with_unreadable_turns_warns_above_its_stop_steps(home, capsys):
    host = ManualHost(home, host="127.0.0.1")
    host.turns = None

    assert engine.cmd_apply(host, _args(dry_run=True, force_interrupt=True)) == engine.EXIT_UNSUPPORTED
    out = capsys.readouterr().out
    warning = "Its turn activity could not be read, so turns may be running on it now: stopping it interrupts them."
    assert warning in out
    assert out.index(warning) < out.index("Steps, in this order")


def test_an_already_blocked_by_hand_plan_skips_activity_read_and_adds_no_turn_blocker(home, capsys, monkeypatch):
    (home / "config.yaml").write_text(
        "model:\n  default: fake-model\nsecrets:\n  vault:\n    enabled: true\n", encoding="utf-8",
    )
    host = ManualHost(home, host="127.0.0.1")
    reads = []

    def fail_if_called(pid, start_time):
        reads.append((pid, start_time))
        return 2

    monkeypatch.setattr(host, "running_turns", fail_if_called)

    code, out = _plan(host, capsys)
    assert code == engine.EXIT_BLOCKED
    assert reads == []
    assert len(out["plan"]["blockers"]) == 1 and "external secret source is enabled" in out["plan"]["blockers"][0]
    assert out["plan"]["unsupported"][0]["steps"] == []


def test_a_by_hand_move_never_plans_keep_webapp_address(home, capsys):
    _webapp(home, "  enabled: true\n  host: 127.0.0.1\n  port: 9200\n")

    code, out = _plan(ManualHost(home, host="127.0.0.1"), capsys, keep_webapp_address=True)

    assert code == engine.EXIT_BLOCKED and out["plan"]["unsupported"][0]["steps"] == []
    assert any("--keep-webapp-address is not planned for a by-hand move" in b for b in out["plan"]["blockers"])


def test_a_by_hand_move_blocks_on_a_bind_the_gateway_refuses(home, capsys):
    code, out = _plan(ManualHost(home, host="example.lan"), capsys)

    assert code == engine.EXIT_BLOCKED and out["plan"]["unsupported"][0]["steps"] == []
    assert any("widening the bind is never done" in b for b in out["plan"]["blockers"])


@pytest.mark.parametrize("manual", [False, True])
def test_an_existing_refused_webapp_public_url_blocks_the_move(home, capsys, manual):
    """The plan sets enabled/host/port but leaves webapp.public_url: the gateway would still refuse it."""
    _webapp(home, "  enabled: false\n  public_url: https://user:pw@dash.example\n")
    host = ManualHost(home, host="127.0.0.1") if manual else FakeHost(home, host="127.0.0.1")

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED
    assert any("After the move the gateway would still refuse the Webapp: webapp.public_url" in b
               for b in out["plan"]["blockers"])
    assert "user:pw" not in json.dumps(out)


# --- review round 1: the public URL the old backend really serves (env first, resolved config)


def _url_plan(home, capsys, monkeypatch, config, env=None, managed=None, shell=None):
    from hermes_cli import managed_scope
    monkeypatch.setattr(managed_scope, "load_managed_config", lambda **_: dict(managed or {}))
    for k, v in (shell or {}).items():
        monkeypatch.setenv(k, v)
    (home / "config.yaml").write_text("model:\n  default: fake-model\n" + config, encoding="utf-8")
    host = FakeHost(home, host="127.0.0.1")
    base = host.process_env(_BACKEND_PID)
    host.process_env = lambda pid: {**base, **(env or {})}
    return _plan(host, capsys)


def test_the_backends_environment_url_wins_over_config_like_the_backend(home, capsys, monkeypatch):
    code, out = _url_plan(home, capsys, monkeypatch, "dashboard:\n  public_url: https://old.example\n",
                          env={"HERMES_DASHBOARD_PUBLIC_URL": "https://live.example"})

    assert code == engine.EXIT_OK and out["plan"]["public_url"] == "https://live.example"


def test_a_blank_environment_url_falls_back_to_config_like_the_backend(home, capsys, monkeypatch):
    code, out = _url_plan(home, capsys, monkeypatch, "dashboard:\n  public_url: https://cfg.example\n",
                          env={"HERMES_DASHBOARD_PUBLIC_URL": "  "})

    assert code == engine.EXIT_OK and out["plan"]["public_url"] == "https://cfg.example"


def test_the_config_url_is_read_after_var_expansion(home, capsys, monkeypatch):
    code, out = _url_plan(home, capsys, monkeypatch, "dashboard:\n  public_url: ${DASH_URL}\n",
                          shell={"DASH_URL": "https://expanded.example"})

    assert code == engine.EXIT_OK and out["plan"]["public_url"] == "https://expanded.example"


def test_the_config_url_includes_the_managed_overlay(home, capsys, monkeypatch):
    code, out = _url_plan(home, capsys, monkeypatch, "",
                          managed={"dashboard": {"public_url": "https://managed.example"}})

    assert code == engine.EXIT_OK and out["plan"]["public_url"] == "https://managed.example"


# --- review round 1: a stopped unit's account and startup environment, not this shell's


def test_a_pinned_system_unit_of_another_account_is_skipped(home, capsys):
    """Returning on HERMES_HOME skipped the User= check: a unit of root pinned to this path is root's."""
    code, out = _plan(StoppedUnitHost(home, user="root", environment={"HERMES_HOME": str(home)}), capsys)

    assert code == engine.EXIT_OK and out["plan"]["nothing_to_do"]
    assert any("runs as root, not as u" in n for n in out["plan"]["notices"])


def test_a_unit_that_picks_a_profile_on_its_command_line_blocks(home, capsys):
    host = StoppedUnitHost(home, user="u", environment={"HERMES_HOME": str(home)}, exec_start=(
        "{ path=/venv/bin/hermes ; argv[]=/venv/bin/hermes -p work dashboard --no-open --port 9119 ; "
        "ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) }"))

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED
    assert any("picks a profile" in b and "Identify the owner" in b for b in out["plan"]["blockers"])


def test_the_default_home_is_the_accounts_not_the_shells_root(tmp_path, monkeypatch):
    """A shell pinned to a custom root does not make that root the service account's default home."""
    import getpass
    import pwd
    account = Path(pwd.getpwnam(getpass.getuser()).pw_dir) / ".hermes"
    custom = tmp_path / "custom-root"
    custom.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(custom))

    assert SystemHost(custom).default_home() is False
    assert SystemHost(account).default_home() is True


def test_the_managers_environment_sets_a_units_home_unless_the_unit_overrides_it(tmp_path, monkeypatch):
    from hermes_cli import gateway_migrate_backends_host as host_mod
    answers = {"show-environment": "HERMES_HOME=/srv/from-manager\nLANG=C\n"}

    def systemctl(scope, args):
        unit = "Environment=" + answers.get("unit", "")
        return SimpleNamespace(returncode=0, stdout=answers["show-environment"] if args == ["show-environment"]
                               else unit + "\n", stderr="")

    monkeypatch.setattr(host_mod, "_systemctl", systemctl)
    assert SystemHost(tmp_path).unit_environment("x.service", "user")[0]["HERMES_HOME"] == "/srv/from-manager"
    answers["unit"] = "HERMES_HOME=/srv/from-unit"
    assert SystemHost(tmp_path).unit_environment("x.service", "user")[0]["HERMES_HOME"] == "/srv/from-unit"


@pytest.mark.parametrize("line", ["HERMES_HOME=/srv/team #1", 'HERMES_HOME="/srv/team"', "HERMES_HOME=/srv/te\\am",
                                  "HERMES_HOME=$HOME/hermes", "OTHER=a \\\nHERMES_HOME=/srv/team"])
def test_an_environment_file_systemd_reads_differently_makes_the_home_unknown(tmp_path, monkeypatch, line):
    """`HERMES_HOME=/srv/team #1` is the whole path to systemd, but a .env reader strips ` #1`."""
    from hermes_cli import gateway_migrate_backends_host as host_mod
    unit_env = tmp_path / "unit.env"
    unit_env.write_text(line + "\n", encoding="utf-8")
    shown = f"Environment=\nEnvironmentFiles={unit_env} (ignore_errors=no)\n"
    monkeypatch.setattr(host_mod, "_systemctl", lambda scope, args: SimpleNamespace(
        returncode=0, stdout="" if args == ["show-environment"] else shown, stderr=""))

    _, problem = SystemHost(tmp_path).unit_environment("x.service", "system")

    assert problem and str(unit_env) in problem


def test_a_plain_environment_file_home_is_read(tmp_path, monkeypatch):
    from hermes_cli import gateway_migrate_backends_host as host_mod
    unit_env = tmp_path / "unit.env"
    unit_env.write_text("# a comment line\nHERMES_HOME=/srv/team\nOTHER=x\n", encoding="utf-8")
    shown = f"Environment=\nEnvironmentFiles={unit_env} (ignore_errors=no)\n"
    monkeypatch.setattr(host_mod, "_systemctl", lambda scope, args: SimpleNamespace(
        returncode=0, stdout="" if args == ["show-environment"] else shown, stderr=""))

    assert SystemHost(tmp_path).unit_environment("x.service", "system") == (
        {"HERMES_HOME": "/srv/team", "OTHER": "x"}, None)


def test_a_backend_that_asked_for_a_free_port_has_no_address(home, capsys):
    """`--port 0` records the request for a free port, not the port it serves: never probe or plan port 0."""
    host = FakeHost(home, host="127.0.0.1")
    host.argv = host.argv[:-1] + ["0"]

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED and host.probes == []
    assert out["plan"]["old_address"] is None and out["plan"]["steps"] == []
    assert any("--port 0" in b for b in out["plan"]["blockers"])


def test_a_by_hand_move_of_a_free_port_backend_prints_no_port(home, capsys):
    host = ManualHost(home, host="127.0.0.1")
    host.argv = host.argv[:-1] + ["0"]

    code, out = _plan(host, capsys)

    (entry,) = out["plan"]["unsupported"]
    assert code == engine.EXIT_UNSUPPORTED and not any("webapp.port: 0" in s or "port 0" in s for s in entry["steps"])


def _work_profile(tmp_path, monkeypatch, *, installed=False, other=None, hosts=False):
    from hermes_cli import gateway as gw
    profile = tmp_path / ".hermes" / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    (profile / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    monkeypatch.setattr(gw, "_is_service_installed", lambda: installed)
    monkeypatch.setattr(gw, "_served_by_another_host_gateway", lambda name=None: other)
    monkeypatch.setattr(gw, "host_multiplexer_serving", lambda name=None: other or (object() if hosts else None))
    return profile


@pytest.mark.parametrize("supervisor", ["systemd", "container"])
def test_a_profile_another_homes_gateway_serves_is_refused_naming_that_gateway(tmp_path, monkeypatch, capsys,
                                                                                supervisor):
    owner = SimpleNamespace(describe=lambda: "PID 42 (launched by profile 'default'; serves: default, work)")
    profile = _work_profile(tmp_path, monkeypatch, other=owner)

    code, out = _plan(FakeHost(profile, supervisor=supervisor), capsys)

    assert code == engine.EXIT_BLOCKED and out["plan"]["steps"] == []
    assert any("served by another home's gateway (PID 42 (launched by profile 'default'" in b
               for b in out["plan"]["blockers"])
    assert all(entry["steps"] == [] for entry in out["plan"]["unsupported"])


@pytest.mark.parametrize("own", ["installed-service", "hosts-the-multiplexer"])
def test_a_profile_with_its_own_gateway_is_planned_without_the_standalone_flag(tmp_path, monkeypatch, capsys, own):
    profile = _work_profile(tmp_path, monkeypatch, installed=own == "installed-service",
                            hosts=own == "hosts-the-multiplexer")

    code, out = _plan(FakeHost(profile, host="127.0.0.1"), capsys)

    assert code == engine.EXIT_OK and not out["plan"]["blockers"]


def test_a_profile_whose_gateway_is_unknown_blocks_without_guessing_an_owner(tmp_path, monkeypatch, capsys):
    profile = _work_profile(tmp_path, monkeypatch)

    code, out = _plan(FakeHost(profile, host="127.0.0.1"), capsys)

    (blocker,) = out["plan"]["blockers"]
    assert code == engine.EXIT_BLOCKED and "Identify which gateway serves this profile" in blocker
    assert "default" not in blocker


# --- review round 2: no-service steps come from the selected target, and never beside a blocker


def _no_service_host(home, monkeypatch, **kw):
    host = FakeHost(home, host="127.0.0.1", **kw)
    monkeypatch.setattr(host, "gateway_state", lambda: {"kind": None, "installed": False, "enabled": False,
                                                        "active": False, "running_pid": None, "unit": None})
    return host


def test_no_service_steps_keep_the_kept_webapp_address(home, capsys, monkeypatch):
    """--keep-webapp-address keeps the Webapp on 9200: no step may set webapp.port 9119 (the old address)."""
    (home / "config.yaml").write_text("model:\n  default: fake-model\nwebapp:\n  enabled: true\n"
                                      "  host: 127.0.0.1\n  port: 9200\n", encoding="utf-8")

    code, out = _plan(_no_service_host(home, monkeypatch), capsys, keep_webapp_address=True)

    (entry,) = out["plan"]["unsupported"]
    assert code == engine.EXIT_UNSUPPORTED and entry["steps"]
    assert not any("webapp.port" in s or "webapp.host" in s for s in entry["steps"]), entry["steps"]


def test_no_service_prints_no_steps_while_a_blocker_remains(home, capsys, monkeypatch):
    host = _no_service_host(home, monkeypatch)
    host.turns = 1

    code, out = _plan(host, capsys)
    assert code == engine.EXIT_BLOCKED and all(e["steps"] == [] for e in out["plan"]["unsupported"])

    engine.cmd_apply(host, _args(dry_run=True))
    text = capsys.readouterr().out
    assert "running" in text and "gateway run" not in text and "webapp.port" not in text


def test_a_blocked_automatic_plan_lists_no_steps(home, capsys):
    host = FakeHost(home, host="127.0.0.1")
    host.turns = 1

    code, out = _plan(host, capsys)
    assert code == engine.EXIT_BLOCKED and out["plan"]["steps"] == [] and out["steps"] == []

    engine.cmd_apply(host, _args(dry_run=True))
    assert "Steps:" not in capsys.readouterr().out


# --- review round 2: a unit's home from its own HOME / suffix, and `export` lines refused


@pytest.mark.parametrize("line", ["export HERMES_HOME=/srv/team", "export HOME=/srv/svc",
                                  "  export HERMES_DATA_DIR_SUFFIX=-dev", 'HOME="/srv/svc"'])
def test_environment_file_lines_systemd_reads_differently_for_the_home_variables(tmp_path, monkeypatch, line):
    """systemd does not strip `export`: `export HERMES_HOME=...` does not set HERMES_HOME for the unit."""
    from hermes_cli import gateway_migrate_backends_host as host_mod
    unit_env = tmp_path / "unit.env"
    unit_env.write_text(line + "\n", encoding="utf-8")
    shown = f"Environment=\nEnvironmentFiles={unit_env} (ignore_errors=no)\n"
    monkeypatch.setattr(host_mod, "_systemctl", lambda scope, args: SimpleNamespace(
        returncode=0, stdout="" if args == ["show-environment"] else shown, stderr=""))

    _, problem = SystemHost(tmp_path).unit_environment("x.service", "system")

    assert problem and str(unit_env) in problem


def test_a_units_default_home_follows_its_own_home_and_suffix():
    import getpass
    import pwd
    account = Path(pwd.getpwnam(getpass.getuser()).pw_dir)

    assert SystemHost(account / ".hermes").default_home({"HOME": "/srv/svc"}) is False
    assert SystemHost(Path("/srv/svc/.hermes")).default_home({"HOME": "/srv/svc"}) is True
    assert SystemHost(account / ".hermes-dev").default_home({"HERMES_DATA_DIR_SUFFIX": "-dev"}) is True
    assert SystemHost(account / ".hermes").default_home({"HERMES_DATA_DIR_SUFFIX": "-dev"}) is False
    assert SystemHost(account / ".hermes").default_home({"HOME": "relative/dir"}) is None


class UnresolvedHomeUnitHost(StoppedUnitHost):
    def default_home(self, env=None):
        return None


def test_a_unit_whose_default_home_cannot_be_told_blocks(home, capsys):
    code, out = _plan(UnresolvedHomeUnitHost(home, user="u", environment={"HOME": "relative/dir"}), capsys)

    assert code == engine.EXIT_BLOCKED
    assert any("sets HOME to something other than a plain absolute path" in b for b in out["plan"]["blockers"])


# --- review round 2: the public URL as the old backend's process has it after load_hermes_dotenv


def _dotenv_url(home, value):
    with (home / ".env").open("a", encoding="utf-8") as f:
        f.write(f"HERMES_DASHBOARD_PUBLIC_URL={value}\n")


def test_the_homes_env_file_url_beats_the_launcher_env_and_config(home, capsys, monkeypatch):
    """hermes_cli/main.py loads the home's .env over the launch environment (override), before YAML."""
    _dotenv_url(home, "https://dotenv.example")

    code, out = _url_plan(home, capsys, monkeypatch, "dashboard:\n  public_url: https://yaml.example\n",
                          env={"HERMES_DASHBOARD_PUBLIC_URL": "https://launcher.example"})

    assert code == engine.EXIT_OK and out["plan"]["public_url"] == "https://dotenv.example"


def test_a_blank_env_file_url_overrides_the_launcher_and_falls_back_to_config(home, capsys, monkeypatch):
    _dotenv_url(home, "")

    code, out = _url_plan(home, capsys, monkeypatch, "dashboard:\n  public_url: https://yaml.example\n",
                          env={"HERMES_DASHBOARD_PUBLIC_URL": "https://launcher.example"})

    assert code == engine.EXIT_OK and out["plan"]["public_url"] == "https://yaml.example"


def test_the_managed_env_url_wins(home, capsys, monkeypatch):
    from hermes_cli import managed_scope
    monkeypatch.setattr(managed_scope, "load_managed_env", lambda: {"HERMES_DASHBOARD_PUBLIC_URL": "https://managed.example"})
    _dotenv_url(home, "https://dotenv.example")

    code, out = _url_plan(home, capsys, monkeypatch, "dashboard:\n  public_url: https://yaml.example\n")

    assert code == engine.EXIT_OK and out["plan"]["public_url"] == "https://managed.example"


@pytest.mark.parametrize("webapp_url", [False, True])
def test_an_enabled_secret_source_blocks_unless_webapp_public_url_is_set(home, capsys, monkeypatch, webapp_url):
    config = "secrets:\n  vault:\n    enabled: true\ndashboard:\n  public_url: https://yaml.example\n"
    if webapp_url:
        config += "webapp:\n  public_url: https://chosen.example\n"

    code, out = _url_plan(home, capsys, monkeypatch, config)

    blocked = any("external secret source is enabled" in b for b in out["plan"]["blockers"])
    assert blocked is not webapp_url and (code == engine.EXIT_BLOCKED) is not webapp_url


# --- review round 2: a backend older than v2026.7.30 has no /api/health


@pytest.mark.parametrize("code", [401, 403, 404])  # v2026.7.1-v2026.7.20 answer 401 before routing
def test_a_backend_without_api_health_is_reported_as_missing(home, monkeypatch, code):
    import io
    import urllib.error
    import urllib.request

    class Opener:
        def open(self, request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, code, "refused", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())

    assert SystemHost(home).http_status("127.0.0.1", 9119) == {"health": "missing"}


def test_current_releases_serve_api_health_without_sign_in():
    """401/403 can only mean an old release: /api/health is on the allowlist both auth gates read."""
    from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS

    assert "/api/health" in PUBLIC_API_PATHS


@pytest.mark.parametrize(("supervisor", "hint"), [("systemd", "systemctl --user restart hermes-dashboard.service"),
                                                   ("manual", "stop it and start it again the way it was started")])
def test_an_old_backend_without_api_health_says_to_restart_it_with_the_updated_hermes(home, capsys, monkeypatch,
                                                                                         supervisor, hint):
    host = FakeHost(home, host="127.0.0.1", supervisor=supervisor)
    monkeypatch.setattr(host, "http_status", lambda h, p: {"health": "missing"})

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED and out["plan"]["steps"] == []
    (blocker,) = [b for b in out["plan"]["blockers"] if "v2026.7.30" in b]
    assert "restart the old backend with the updated hermes" in blocker and hint in blocker
    assert "re-run the dry run" in blocker


# --- Bear round 2: a backend without /api/health may still be writing a reply


@pytest.mark.parametrize(("turns", "expect"), [(2, "2 turn(s) are running on it now: let them finish"),
                                                (0, "confirm no reply is being written in it"),
                                                (OSError, "confirm no reply is being written in it")])
def test_an_old_backend_without_api_health_is_not_restarted_mid_reply(home, capsys, monkeypatch, turns, expect):
    host = FakeHost(home, host="127.0.0.1")
    monkeypatch.setattr(host, "http_status", lambda h, p: {"health": "missing"})
    if turns is OSError:
        def unreadable(*a):
            raise OSError("marker unreadable")
        monkeypatch.setattr(host, "running_turns", unreadable)
    else:
        host.turns = turns

    code, out = _plan(host, capsys)

    (blocker,) = [b for b in out["plan"]["blockers"] if "v2026.7.30" in b]
    assert code == engine.EXIT_BLOCKED and expect in blocker
    assert "Restart the old backend" not in blocker and "restart the old backend with the updated hermes" in blocker
    assert out["plan"]["running_turns"] == (turns if turns == 2 else None)


# --- review round 5: a by-hand backend older than v2026.7.30 writes no turn markers


@contextlib.contextmanager
def no_health_server():
    """A real HTTP server on loopback that answers 404 on every path, as a Hermes before v2026.7.30 does
    for GET /api/health (it has no such route and writes no turn markers)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen: list[str] = []

    class NoHealth(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), NoHealth)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(port=server.server_address[1], seen=seen)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture
def no_health_backend():
    with no_health_server() as stub:
        yield stub


def no_health_host(host_cls, home, stub, **kw):
    """*host_cls* for a by-hand backend on the stub's port, probed with the real ``SystemHost.http_status``
    and the real marker reader (no marker files in the home, as an old backend leaves it)."""
    host = host_cls(home, host="127.0.0.1", **kw)
    host.argv = [a if a != "9119" else str(stub.port) for a in host.argv]
    host.http_status = SystemHost.http_status.__get__(host, SystemHost)
    host.running_turns = SystemHost.running_turns.__get__(host, SystemHost)
    return host


@pytest.mark.parametrize("supervisor", ["container", "s6", "systemd-system"])
def test_a_by_hand_backend_without_api_health_is_unknown_activity(home, capsys, no_health_backend, supervisor):
    kw = {"supervisor": "systemd", "scope": "system"} if supervisor == "systemd-system" else {"supervisor": supervisor}
    host = no_health_host(FakeHost, home, no_health_backend, **kw)

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED and out["plan"]["running_turns"] is None
    assert "/api/health" in no_health_backend.seen
    assert any("--force-interrupt" in b for b in out["plan"]["blockers"])
    assert out["plan"]["unsupported"][0]["steps"] == []

    code, out = _plan(host, capsys, force_interrupt=True)
    assert code == engine.EXIT_UNSUPPORTED and out["plan"]["running_turns"] is None
    assert out["plan"]["unsupported"][0]["steps"]


# --- review round 5: forced past known running turns, every by-hand plan still says so


_RUNNING = "2 turn(s) are running on it now: stopping it interrupts them"


@pytest.mark.parametrize("supervisor", ["container", "s6", "systemd-system"])
def test_a_forced_by_hand_plan_with_running_turns_warns_above_its_steps(home, capsys, supervisor):
    """The forced plan (what the dashboard shows after "Show the steps anyway") names the count in its
    instructions on Linux and macOS too, not only in the Windows recipe."""
    kw = {"supervisor": "systemd", "scope": "system"} if supervisor == "systemd-system" else {"supervisor": supervisor}
    host = FakeHost(home, host="127.0.0.1", **kw)
    host.turns = 2

    code, out = _plan(host, capsys, force_interrupt=True)
    (entry,) = out["plan"]["unsupported"]
    assert code == engine.EXIT_UNSUPPORTED and entry["steps"] and _RUNNING in entry["instructions"]

    assert engine.cmd_apply(host, _args(dry_run=True, force_interrupt=True)) == engine.EXIT_UNSUPPORTED
    text = capsys.readouterr().out
    assert _RUNNING in text and text.index(_RUNNING) < text.index("Steps, in this order")
