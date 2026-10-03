"""Regression tests for #126076: post-update gateway verification must vouch for OUR
relaunched gateways — never any fleet process, and never nothing when ours are alive.

Two failure modes shared one symptom (``RuntimeError: ... not verified alive``, exit 1):

1. False failure — the scope-filtered fleet poll (``_owned_gateway_pids``) drops gateways
   whose home is unreadable (service-logon processes hide their environment), so a live
   relaunched gateway read as dead. A live per-profile ``gateway.pid`` must vouch.
2. False pass (fixed by attribution, pinned here) — a fleet hit from a foreign gateway
   vouched for OUR dead profile. A PID file naming a dead process is positive proof of
   death; the fleet poll must not override it.

Also pinned: ``sc start`` success alone never proves the gateway child survived, so a
restarted SCM service requires SCM ``running`` AND gateway liveness — including on the
service-only resume path, which previously skipped verification entirely.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_mod
import hermes_cli.main as hm
import hermes_cli.profiles as profiles_mod
import hermes_cli.update_cmd_windows as uw
from hermes_cli import gateway_windows


@pytest.fixture
def verify_stubs(monkeypatch, tmp_path):
    """Stub the fleet poll + attestation; route profile homes into tmp_path."""
    state = {"ready": [], "unfiltered": [], "attested": []}
    monkeypatch.setattr(
        gateway_windows, "_wait_for_gateway_ready", lambda **kw: list(state["ready"])
    )
    monkeypatch.setattr(
        gateway_windows, "_write_start_attestation",
        lambda pids, via, **kw: state["attested"].append((list(pids), via)),
    )
    monkeypatch.setattr(
        gateway_mod, "find_gateway_pids", lambda **kw: list(state["unfiltered"])
    )
    homes = tmp_path / "homes"
    homes.mkdir()
    monkeypatch.setattr(
        profiles_mod, "get_profile_dir", lambda name: homes / str(name)
    )
    return state


def _write_pid_file(home: Path, pid: int) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "gateway.pid"
    path.write_text(json.dumps({"pid": int(pid)}), encoding="utf-8")
    return path


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    assert proc.pid != os.getpid()
    return int(proc.pid)


def _profile_token(**overrides):
    token = {
        "resume_needed": True,
        "profiles": {"default": 999999},
        "unmapped": [],
        "services": [],
        "restarted_services": [],
        "service_profiles": {},
    }
    token.update(overrides)
    return token


class TestDirectProfileVerification:
    def test_live_pid_file_vouches_when_fleet_poll_is_empty(self, verify_stubs, tmp_path):
        """False failure of #126076: the scope filter hid our gateway, the PID file proves it."""
        home = tmp_path / "homes" / "default"
        _write_pid_file(home, os.getpid())
        token = _profile_token()

        uw._verify_relaunched_gateways_alive(token, token["profiles"], token["unmapped"])

        assert verify_stubs["attested"] == [([os.getpid()], "post-update relaunch")]

    def test_stale_pid_file_is_not_vouched_by_a_foreign_fleet_hit(self, verify_stubs, tmp_path):
        """False pass of #126076: OUR pid file names a dead process; a fleet hit is someone else's."""
        home = tmp_path / "homes" / "default"
        _write_pid_file(home, _dead_pid())
        verify_stubs["ready"] = [os.getpid()]
        token = _profile_token()

        with pytest.raises(RuntimeError, match="not verified alive"):
            uw._verify_relaunched_gateways_alive(token, token["profiles"], token["unmapped"])

        assert token["profiles"] == {"default": 999999}
        assert token["unmapped"] == []
        assert verify_stubs["attested"] == []

    def test_missing_pid_file_accepts_fleet_poll(self, verify_stubs):
        """Slow PID-file write (or stubbed probe): the fleet poll still vouches."""
        verify_stubs["ready"] = [4242]
        token = _profile_token()

        uw._verify_relaunched_gateways_alive(token, token["profiles"], token["unmapped"])

        assert verify_stubs["attested"] == [([4242], "post-update relaunch")]

    def test_dead_relaunch_with_no_evidence_still_fails(self, verify_stubs):
        """The #48820 gate survives: nothing anywhere means the relaunch is not verified."""
        token = _profile_token()

        with pytest.raises(RuntimeError, match="not verified alive"):
            uw._verify_relaunched_gateways_alive(token, token["profiles"], token["unmapped"])

        assert token["profiles"] == {"default": 999999}

    def test_one_fleet_hit_does_not_vouch_for_two_missing_profiles(self, verify_stubs):
        """kvnloo review: one scope-filtered hit cannot vouch for two record-less profiles.

        A has no PID file and is alive (the lone fleet hit), B has no PID file and is
        dead — B must still fail instead of riding A's hit.
        """
        verify_stubs["ready"] = [4242]
        profiles = {"a": 1111, "b": 2222}
        token = _profile_token(profiles=dict(profiles))
        # Neither home has a gateway.pid (slow/missing write for A, dead gateway for B).

        with pytest.raises(RuntimeError, match="not verified alive"):
            uw._verify_relaunched_gateways_alive(token, profiles, [])

        missing = uw._missing_relaunched_gateways(token, profiles, [], {4242}, {})
        assert any("'b'" in entry for entry in missing)
        assert verify_stubs["attested"] == []

    def test_two_fleet_hits_vouch_for_two_missing_profiles(self, verify_stubs):
        """Enough distinct fleet processes cover every record-less profile (slow boot)."""
        verify_stubs["ready"] = [4242, 4243]
        profiles = {"a": 1111, "b": 2222}
        token = _profile_token(profiles=dict(profiles))

        uw._verify_relaunched_gateways_alive(token, profiles, [])

        assert verify_stubs["attested"] == [([4242, 4243], "post-update relaunch")]


class TestServiceVerification:
    @staticmethod
    def _stub_service(monkeypatch, status: str):
        class _Service:
            def __init__(self, service_status: str):
                self._status = service_status

            def status(self):
                return self._status

        monkeypatch.setattr(uw, "_win_service", lambda name: (None, _Service(status)))

    def test_restarted_service_requires_scm_running(self, monkeypatch, verify_stubs):
        """``sc start`` success alone never proves the gateway child survived (#126076)."""
        self._stub_service(monkeypatch, "stopped")
        verify_stubs["ready"] = [4242]
        token = _profile_token(
            profiles={}, restarted_services=["svc"], service_profiles={"svc": "default"}
        )

        with pytest.raises(RuntimeError, match="not verified alive"):
            uw._verify_relaunched_gateways_alive(token, {}, [])

    def test_restarted_service_verified_via_pid_file(self, monkeypatch, verify_stubs, tmp_path):
        self._stub_service(monkeypatch, "running")
        _write_pid_file(tmp_path / "homes" / "default", os.getpid())
        token = _profile_token(
            profiles={}, restarted_services=["svc"], service_profiles={"svc": "default"}
        )

        uw._verify_relaunched_gateways_alive(token, {}, [])

        assert verify_stubs["attested"] == [([os.getpid()], "post-update relaunch")]

    def test_service_only_resume_verifies_restarted_services(self, monkeypatch, verify_stubs, tmp_path):
        """The resume path with no direct relaunches must still gate on service liveness."""
        self._stub_service(monkeypatch, "running")
        monkeypatch.setattr(hm, "_is_windows", lambda: True)
        monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda: None)
        monkeypatch.setattr(uw, "_resume_windows_services", lambda token: None)
        monkeypatch.setattr(uw, "_cold_start_attested_profiles", lambda token: None)
        token = _profile_token(
            profiles={}, unmapped=[],
            restarted_services=["svc"], service_profiles={"svc": "default"},
        )

        with pytest.raises(RuntimeError, match="not verified alive"):
            uw._resume_windows_gateways_after_update(token)

        assert token["resume_needed"] is True


class TestUnmappedVerification:
    def test_unmapped_vouched_by_unfiltered_scan(self, verify_stubs):
        """A task-launched gateway with an unreadable home still verifies via the raw scan."""
        verify_stubs["unfiltered"] = [7777]
        token = _profile_token(
            profiles={}, unmapped=[{"pid": 1234, "argv": ["python", "-m", "gateway"]}]
        )

        uw._verify_relaunched_gateways_alive(token, {}, token["unmapped"])

        assert verify_stubs["attested"] == [([7777], "post-update relaunch")]

    def test_unmapped_with_no_process_anywhere_fails(self, verify_stubs):
        token = _profile_token(
            profiles={}, unmapped=[{"pid": 1234, "argv": ["python", "-m", "gateway"]}]
        )

        with pytest.raises(RuntimeError, match="not verified alive"):
            uw._verify_relaunched_gateways_alive(token, {}, token["unmapped"])
