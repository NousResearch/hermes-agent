"""Multiplex-only reporting: one host gateway serving N profiles must be reported as such.

Invariants (all fail on origin/main):
* ``hermes -p <served> cron status`` names the HOST gateway as the ticker and never tells the
  user to start a second process.
* ``hermes -p <served> doctor`` reports the single host gateway + its roster, not
  "Per-profile gateways: up/total", and still checks the HOST systemd unit's linger.
* ``hermes claw``'s destructive-action warning FIRES for a served profile.
* the state.db holder line names the shared host process and the profiles it serves.
"""

import json
import os

import pytest


@pytest.fixture
def served_host(tmp_path, monkeypatch):
    """A host gateway (this PID) serving ``default`` + ``served``; the active profile is ``served``,
    which owns no gateway.pid / gateway_state.json of its own (#97120)."""
    from gateway import host_rendezvous as hr

    root = tmp_path / "hermes"
    home = root / "profiles" / "served"
    home.mkdir(parents=True)
    locks = tmp_path / "locks"
    locks.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "served")
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(locks))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
    monkeypatch.setattr("gateway.status.is_gateway_runtime_lock_active", lambda lock_path=None: False)
    hr.publish_record(hr.ROLE_GATEWAY, profiles=("default", "served"))
    return root


def test_cron_status_names_the_host_gateway_for_a_served_profile(served_host, capsys, monkeypatch):
    from cron import jobs
    from hermes_cli import cron

    monkeypatch.setattr(jobs, "CRON_DIR", served_host / "profiles/served/cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", served_host / "profiles/served/cron/jobs.json")
    cron.cron_status()
    out = capsys.readouterr().out

    assert f"the host gateway (PID {os.getpid()}) serving profiles default, served" in out
    assert "not running" not in out.lower()
    # Never route the user to a SECOND host process: per-profile install is legacy-only remediation.
    assert "hermes gateway install" not in out
    assert "sudo hermes gateway install --system" not in out


def test_claw_warning_fires_for_a_served_profile(served_host, capsys, monkeypatch):
    from hermes_cli import claw

    (served_host / "gateway_state.json").write_text(json.dumps(
        {"served_profiles": ["default", "served"],
         "platforms": {"served:telegram": {"state": "connected"}, "telegram": {"state": "connected"}}}))
    fired = []
    monkeypatch.setattr(claw, "_warn_running", lambda *a, **k: fired.append(a) or True)

    claw._warn_if_gateway_running(auto_yes=True)

    assert fired, "destructive-action warning was silently skipped for a served profile"
    assert "telegram" in fired[0][1]


def test_doctor_reports_the_single_host_gateway_not_per_profile_slots(served_host, capsys, monkeypatch):
    from hermes_cli import doctor_platform

    class _Mgr:
        def is_running(self, name):
            return name in ("main-hermes", "gateway-legacy")

        def list_profile_gateways(self):
            return ["default", "served", "legacy"]

    monkeypatch.setattr("hermes_cli.service_manager.detect_service_manager", lambda: "s6")
    monkeypatch.setattr("hermes_cli.service_manager.S6ServiceManager", _Mgr)
    issues: list[str] = []
    doctor_platform._check_s6_supervision(issues)
    out = capsys.readouterr().out

    assert "Per-profile gateways:" not in out
    assert f"Host gateway: the host gateway (PID {os.getpid()}) serving profiles default, served" in out
    assert "LEGACY per-profile gateway slots still supervised: legacy" in out
    assert any("migrate --multiplex" in i for i in issues)


def test_doctor_raises_an_issue_when_nothing_owns_the_gateway_role(served_host, capsys, monkeypatch):
    """The WORST state must produce remediation: slots exist, nothing serves them. The strictly
    less severe LEGACY case already appended an issue, so `doctor` exited 0 with `issues: []`
    on the outage and non-zero on the mere leftover."""
    from hermes_cli import doctor_platform

    class _Mgr:
        def is_running(self, name):
            return name == "main-hermes"

        def list_profile_gateways(self):
            return ["default", "served", "legacy"]

    monkeypatch.setattr("gateway.host_topology.host_gateway_topology", lambda: None)
    monkeypatch.setattr("hermes_cli.service_manager.detect_service_manager", lambda: "s6")
    monkeypatch.setattr("hermes_cli.service_manager.S6ServiceManager", _Mgr)
    issues: list[str] = []
    doctor_platform._check_s6_supervision(issues)

    assert "No host gateway owns the gateway role" in capsys.readouterr().out
    assert any("gateway start" in i for i in issues), "the outage state produced no remediation"


@pytest.mark.parametrize("xdg", [False, True])
def test_doctor_checks_host_unit_linger_under_a_served_profile(served_host, tmp_path, capsys, monkeypatch, xdg):
    from hermes_cli import doctor_platform

    fake_home = tmp_path / "fakehome"
    config_home = (tmp_path / "xdgconfig") if xdg else (fake_home / ".config")
    unit_dir = config_home / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "hermes-gateway.service").write_text("[Unit]\n")  # the HOST unit; no per-profile unit
    monkeypatch.setenv("HOME", str(fake_home))
    # A host that moves XDG_CONFIG_HOME keeps its units there; probing ~/.config false-negatives.
    if xdg:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    else:
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr("hermes_cli.gateway.is_linux", lambda: True)
    monkeypatch.setattr("hermes_cli.service_manager.detect_service_manager", lambda: "systemd")
    monkeypatch.setattr("hermes_cli.gateway.get_systemd_linger_status", lambda *a, **k: (False, "off"))

    issues: list[str] = []
    doctor_platform._check_gateway_service_linger(issues)

    assert "Systemd linger disabled" in capsys.readouterr().out
    assert any("enable-linger" in i for i in issues)


def test_state_db_holder_line_names_the_shared_host_gateway(served_host):
    from hermes_cli import doctor_state

    rows = doctor_state._render_state_db_stats(
        {"messages": 5}, holders=3, host_note=doctor_state.host_gateway_note())
    text = " ".join(t for _kind, t, _detail in rows)

    assert f"3 process(es) holding the DB open (the host gateway (PID {os.getpid()}) " \
           "serving profiles default, served)" in text
