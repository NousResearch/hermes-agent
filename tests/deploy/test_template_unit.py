"""litco-agent@.service: the per-matter slot unit on a machine host (FIRM_AGENT_HOST section 3.2)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.deploy._systemd_unit import parse_unit, space_list
from tests.host._load import HOST

UNIT_PATH = HOST / "litco-agent@.service"
UNIT = parse_unit(UNIT_PATH.read_text())
SERVICE = UNIT["Service"]


def env_of(unit) -> dict:
    return dict(item.split("=", 1) for value in unit["Service"].get("Environment", []) for item in value.split())


# Every hardening line section 3.2 names, with the value it must carry.
HARDENING = {
    "User": "m_%i",
    "ProtectProc": "invisible",
    "ProcSubset": "pid",
    "PrivateTmp": "yes",
    "PrivateIPC": "yes",
    "TemporaryFileSystem": "/srv/litco/m",
    "BindPaths": "/srv/litco/m/%i",
    "IPAddressDeny": "169.254.169.254",
    "NoNewPrivileges": "yes",
    "MemoryMax": "2G",
    "TasksMax": "2048",
    "CPUWeight": "100",
    "EnvironmentFile": "/run/litco-agent/%i.env",
}


@pytest.mark.parametrize("key, value", sorted(HARDENING.items()))
def test_every_section_3_2_hardening_directive_is_set_once(key, value):
    assert SERVICE.get(key) == [value]


def test_the_slot_runs_in_its_own_home_and_cannot_see_the_secrets_store():
    assert SERVICE["Group"] == ["m_%i"]
    assert SERVICE["WorkingDirectory"] == ["/srv/litco/m/%i"]
    env = env_of(UNIT)
    assert env["HOME"] == "/srv/litco/m/%i"
    assert env["HERMES_HOME"].startswith("/srv/litco/m/%i/")
    assert env["LITCO_MATTER_HOME"].startswith("/srv/litco/m/%i/")
    assert env["LITCO_SLOT_ID"] == "%i"
    hidden = {p.lstrip("-") for p in space_list(UNIT, "Service", "InaccessiblePaths")}
    assert {"/run/litco-agent", "/var/lib/litco-supervisor", "/etc/litco-supervisor.env"} <= hidden


def test_no_start_without_the_supervisors_env_file_and_never_enabled():
    assert UNIT["Unit"]["ConditionPathExists"] == ["/run/litco-agent/%i.env"]
    # /run is empty after a reboot; the supervisor starts slots on demand.
    assert "Install" not in UNIT


def test_stop_drains_first_and_never_times_out():
    assert SERVICE["ExecStop"][0] == "/opt/litco-agent/current/deploy/host/litco-agent-drain"
    assert SERVICE["TimeoutStopSec"] == ["infinity"]
    assert SERVICE["KillMode"] == ["mixed"]


def test_start_resolves_the_current_release_once():
    assert SERVICE["ExecStartPre"] == ["/opt/litco-agent/current/deploy/host/litco-agent-init"]
    (start,) = SERVICE["ExecStart"]
    assert start.startswith("/bin/sh -c ")
    assert "readlink -f /opt/litco-agent/current" in start
    assert '"$$app/.venv/bin/python" -m hermes_cli.main gateway run' in start
    assert "exec " in start, "the gateway must be the main process, so KillMode=mixed signals it"


def test_no_secret_value_or_secret_name_in_the_unit():
    text = UNIT_PATH.read_text()
    for name in ("LITCO_HOST_SECRET", "LITCO_AGENT_TOKEN", "_API_KEY", "lkm_"):
        assert name not in text


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze not available on this host")
def test_systemd_analyze_verify(tmp_path):
    # verify refuses a bare template; a file named as an instance loads with %i=smoke.
    copy = tmp_path / "litco-agent@smoke.service"
    copy.write_text(UNIT_PATH.read_text())
    out = subprocess.run(["systemd-analyze", "verify", f"{copy}"], capture_output=True, text=True)
    assert "Unknown key" not in out.stderr and "Invalid" not in out.stderr, out.stderr
