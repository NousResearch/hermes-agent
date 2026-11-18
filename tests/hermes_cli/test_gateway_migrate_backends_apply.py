"""``hermes gateway migrate --backends --apply --dry-run``: the plan and preflight of the backend migration.

A fake host stands in for the service manager, the process table and the network (the gateway's user
service, the old backend's unit, ``/api/health``); config.yaml and .env are the real files in a temp
``HERMES_HOME``, and the bind check is the gateway's own ``webapp.host`` validation. A dry run changes
nothing: no file in the home is written or added. (The real command on a scratch machine:
``test_gateway_migrate_backends_dry_run_readonly.py``.)
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli.gateway_migrate_backends import BackendItem
from hermes_cli.gateway_migrate_backends_host import SystemHost
from hermes_cli.gateway_migrate_backends_plan import WEBAPP_STATUS_WAIT

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
                         f"install; 'service is stopped' or 'service is not loaded' -> {h} gateway start; "
                         f"'service is running' or 'supervised by launchd' -> {h} gateway restart",
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
    (profile / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")

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

    def default_home(self):
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
                        lambda scope, args: SimpleNamespace(returncode=0, stdout=shown, stderr=""))

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
                        lambda scope, args: SimpleNamespace(returncode=0, stdout=shown, stderr=""))

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
                        lambda scope, args: SimpleNamespace(returncode=0, stdout=shown, stderr=""))

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
