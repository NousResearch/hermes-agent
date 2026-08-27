"""Tests for hermes_cli.update_receipt — Phase 1 of the fleet-update plan (#91277).

Covers:
- receipt lifecycle (begin → record → finalize → read back)
- skip recording with reasons
- gateway restart phase recording (success + phase-error shapes)
- receipt pruning
- fleet version classification (current / stale / unknown)
- gateway_state.json code-identity stamping (gateway/status.py side)
"""

import json
import os
import sys
from pathlib import Path

import pytest

import hermes_cli.update_receipt as ur


@pytest.fixture()
def receipt_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME for receipt writes."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home", lambda: home, raising=False
    )
    # ensure no receipt bleeds between tests
    ur._current = None
    yield home
    ur._current = None


def _finalize(outcome="success", fleet=None):
    return ur.finalize_update_receipt(outcome, fleet=fleet)


class TestReceiptLifecycle:
    def test_begin_record_finalize_roundtrip(self, receipt_home):
        ur.begin_update_receipt()
        ur.record_step("pre_update_backup", True, "snapshot=abc123")
        ur.record_skip("gateway_restart", "no gateways running")
        ur.record_gateway_restart(
            restarted_services=["hermes-gateway"],
            relaunched_profiles=["work"],
            killed_pids=[123],
            failed_units=[],
            incomplete=False,
        )
        path = _finalize("success", fleet=[{"profile": "default", "state": "current"}])
        assert path is not None and path.is_file()

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == 1
        assert payload["outcome"] == "success"
        assert payload["finished_at"] is not None
        assert payload["steps"][0]["name"] == "pre_update_backup"
        assert payload["steps"][0]["ok"] is True
        assert payload["skips"][0]["reason"] == "no gateways running"
        gr = payload["gateway_restart"]
        assert gr["restarted_services"] == ["hermes-gateway"]
        assert gr["relaunched_profiles"] == ["work"]
        assert gr["killed_pids"] == [123]
        assert gr["incomplete"] is False
        assert payload["fleet"][0]["profile"] == "default"

    def test_fresh_recovery_result_reaches_persisted_receipt(self, receipt_home):
        recovery = {
            "requested": ["coder", "default", "ops"],
            "verified": ["default"],
            "relaunch_attempted": ["ops"],
            "failed": ["coder"],
            "skipped": [
                {
                    "profile": "desk",
                    "kind": "serve",
                    "supervisor": "desktop",
                    "reason": "desktop app owns and respawns this serve backend",
                }
            ],
        }

        ur.begin_update_receipt()
        ur.record_gateway_restart(
            restarted_services=[],
            incomplete=True,
            phase_error="boom: module vanished mid-pull",
            fresh_recovery=recovery,
        )
        path = _finalize("partial")

        payload = json.loads(path.read_text(encoding="utf-8"))
        persisted = payload["gateway_restart"]["fresh_recovery"]
        assert persisted == recovery
        # The conservative vocabulary is the persisted contract: no bucket may
        # rebrand an unverified relaunch as supervisor-backed success.
        assert "succeeded" not in persisted

    def test_latest_pointer_written_and_readable(self, receipt_home):
        ur.begin_update_receipt()
        ur.record_step("git_pull", True)
        _finalize("partial")
        latest = ur.read_latest_receipt()
        assert latest is not None
        assert latest["outcome"] == "partial"

    def test_phase_error_shape(self, receipt_home):
        ur.begin_update_receipt()
        ur.record_gateway_restart(
            restarted_services=[],
            incomplete=True,
            phase_error="boom: module vanished mid-pull",
        )
        path = _finalize("partial")
        payload = json.loads(path.read_text(encoding="utf-8"))
        gr = payload["gateway_restart"]
        assert gr["incomplete"] is True
        assert "boom" in gr["phase_error"]

    def test_record_without_begin_is_noop(self, receipt_home):
        # No begin — nothing should raise, nothing should be written.
        ur.record_step("orphan", True)
        ur.record_skip("orphan", "no receipt")
        ur.record_gateway_restart(restarted_services=[])
        assert _finalize("success") is None

    def test_finalize_clears_current(self, receipt_home):
        ur.begin_update_receipt()
        assert ur._current is not None
        _finalize("success")
        assert ur._current is None

    def test_pruning_keeps_recent(self, receipt_home, monkeypatch):
        monkeypatch.setattr(ur, "_RECEIPT_KEEP", 3)
        directory = receipt_home / "logs" / "update_receipts"
        directory.mkdir(parents=True)
        for i in range(6):
            p = directory / f"update_2026010{i}_000000_1.json"
            p.write_text("{}", encoding="utf-8")
            os.utime(p, (1000 + i, 1000 + i))
        ur._prune_old_receipts(directory)
        remaining = sorted(p.name for p in directory.glob("update_*.json"))
        assert len(remaining) == 3
        # newest three survive
        assert remaining == [
            "update_20260103_000000_1.json",
            "update_20260104_000000_1.json",
            "update_20260105_000000_1.json",
        ]

    def test_read_latest_receipt_missing(self, receipt_home):
        assert ur.read_latest_receipt() is None


class TestCommandBoundaryFinalization:
    """Receipt lifetime is owned by the update-command boundary (#91283 review).

    Early sys.exit paths (concurrent-instance preflight exit-2, venv-holder
    refusal, fetch failure) predate the inner finalize sites; the boundary
    safety net must persist the receipt exactly once with the stop reason,
    while inner-finalized runs are untouched.
    """

    def test_pending_receipt_persisted_on_exit_2_refusal(self, receipt_home):
        ur.begin_update_receipt()
        ur.record_step("windows_preflight", False, "another hermes.exe running")
        path = ur.finalize_pending_update_receipt(2, "sys.exit(2)")
        assert path is not None and path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["outcome"] == "refused"
        assert payload["exit_code"] == 2
        assert payload["stop_reason"] == "sys.exit(2)"
        assert payload["finished_at"] is not None
        assert ur._current is None

    def test_pending_receipt_persisted_on_exit_1_failure(self, receipt_home):
        ur.begin_update_receipt()
        path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["outcome"] == "failed"
        assert payload["exit_code"] == 1

    def test_noop_when_inner_path_already_finalized(self, receipt_home):
        """Exactly-once: boundary call after an inner finalize writes nothing."""
        ur.begin_update_receipt()
        first = ur.finalize_update_receipt("success")
        assert first is not None
        second = ur.finalize_pending_update_receipt(0, "boundary")
        assert second is None
        directory = receipt_home / "logs" / "update_receipts"
        assert len(list(directory.glob("update_*.json"))) == 1

    def test_noop_when_never_begun(self, receipt_home):
        assert ur.finalize_pending_update_receipt(2, "sys.exit(2)") is None
        assert ur.read_latest_receipt() is None

    def test_cmd_update_boundary_finalizes_on_early_exit(
        self, receipt_home, monkeypatch
    ):
        """End-to-end through the real cmd_update wrapper: an impl that begins
        a receipt then sys.exit(2)s (the concurrent-instance shape) must leave
        a finalized 'refused' receipt, preserve the exit code, and clear the
        singleton."""
        from types import SimpleNamespace

        from hermes_cli import main as hermes_main

        def _fake_impl(args, gateway_mode):
            ur.begin_update_receipt()
            ur.record_step("windows_preflight", False, "hermes.exe holds venv")
            sys.exit(2)

        monkeypatch.setattr(hermes_main, "_cmd_update_impl", _fake_impl)
        monkeypatch.setattr(
            hermes_main, "detect_install_method", lambda *a, **k: "git", raising=False
        )
        monkeypatch.setattr(
            hermes_main,
            "_install_hangup_protection",
            lambda gateway_mode: None,
            raising=False,
        )
        monkeypatch.setattr(
            hermes_main, "_finalize_update_output", lambda state: None, raising=False
        )

        class _FakeLock:
            holder = None

            def acquire(self):
                return True

            def release(self):
                pass

        import hermes_cli.update_lock as update_lock_mod

        monkeypatch.setattr(update_lock_mod, "UpdateLock", _FakeLock)

        args = SimpleNamespace(
            check=False, gateway=False, branch=None, yes=False,
            force=False, force_venv=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            hermes_main.cmd_update(args)

        assert exc_info.value.code == 2  # exit code preserved
        latest = ur.read_latest_receipt()
        assert latest is not None
        assert latest["outcome"] == "refused"
        assert latest["exit_code"] == 2
        assert latest["stop_reason"] == "sys.exit(2)"
        assert latest["steps"][0]["name"] == "windows_preflight"
        assert ur._current is None
        # exactly-once: exactly one receipt file
        directory = receipt_home / "logs" / "update_receipts"
        assert len(list(directory.glob("update_*.json"))) == 1


class TestFleetClassification:
    def _fleet_with(self, monkeypatch, tmp_path, record, expected_sha="a" * 40):
        """Run collect_fleet_versions against one fake default profile."""
        home = tmp_path / "fleet_home"
        home.mkdir()
        (home / "gateway_state.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        monkeypatch.setattr(
            "hermes_cli.build_info.get_code_identity",
            lambda refresh=False: {"sha": expected_sha, "short_sha": expected_sha[:8],
                                   "version": "1.0", "source": "git"},
        )
        monkeypatch.setattr(
            "hermes_cli.profiles._get_default_hermes_home", lambda: home
        )
        monkeypatch.setattr(
            "hermes_cli.profiles._get_profiles_root",
            lambda: tmp_path / "nonexistent_profiles_root",
        )
        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
        return ur.collect_fleet_versions()

    def test_current_gateway(self, monkeypatch, tmp_path):
        sha = "a" * 40
        fleet = self._fleet_with(
            monkeypatch, tmp_path,
            {"pid": 4242, "code_sha": sha, "code_version": "1.0"},
            expected_sha=sha,
        )
        assert len(fleet) == 1
        assert fleet[0]["state"] == "current"
        assert fleet[0]["pid"] == 4242

    def test_stale_gateway(self, monkeypatch, tmp_path):
        fleet = self._fleet_with(
            monkeypatch, tmp_path,
            {"pid": 4242, "code_sha": "b" * 40, "code_version": "0.9"},
            expected_sha="a" * 40,
        )
        assert fleet[0]["state"] == "stale"

    def test_unstamped_gateway_is_unknown(self, monkeypatch, tmp_path):
        # Pre-feature gateway: no code_sha in its runtime status.
        fleet = self._fleet_with(
            monkeypatch, tmp_path, {"pid": 4242}, expected_sha="a" * 40
        )
        assert fleet[0]["state"] == "unknown"

    def test_dead_pid_excluded(self, monkeypatch, tmp_path):
        home = tmp_path / "fleet_home2"
        home.mkdir()
        (home / "gateway_state.json").write_text(
            json.dumps({"pid": 999999, "code_sha": "a" * 40}), encoding="utf-8"
        )
        monkeypatch.setattr(
            "hermes_cli.profiles._get_default_hermes_home", lambda: home
        )
        monkeypatch.setattr(
            "hermes_cli.profiles._get_profiles_root",
            lambda: tmp_path / "nope",
        )
        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
        assert ur.collect_fleet_versions() == []

    def test_matrix_returns_true_only_on_stale(self, capsys):
        assert ur.print_fleet_version_matrix([]) is False
        ok = ur.print_fleet_version_matrix(
            [{"profile": "default", "pid": 1, "code_sha": "a" * 40, "state": "current"}]
        )
        assert ok is False
        stale = ur.print_fleet_version_matrix(
            [
                {"profile": "default", "pid": 1, "code_sha": "a" * 40, "state": "current"},
                {"profile": "work", "pid": 2, "code_sha": "b" * 40, "state": "stale"},
            ]
        )
        assert stale is True
        out = capsys.readouterr().out
        assert "STALE" in out
        assert "hermes -p <profile> gateway restart" in out

    def test_unknown_does_not_fail_update(self, capsys):
        ok = ur.print_fleet_version_matrix(
            [{"profile": "default", "pid": 1, "code_sha": None, "state": "unknown"}]
        )
        assert ok is False
        assert "version unknown" in capsys.readouterr().out


class TestGatewayStatusStamping:
    def test_runtime_status_record_carries_code_identity(self, monkeypatch):
        import gateway.status as gs

        monkeypatch.setattr(
            "hermes_cli.build_info.get_code_identity",
            lambda refresh=False: {"sha": "c" * 40, "short_sha": "c" * 8,
                                   "version": "2.0", "source": "git"},
        )
        record = gs._build_runtime_status_record()
        assert record["code_sha"] == "c" * 40
        assert record["code_version"] == "2.0"

    def test_code_identity_failure_degrades_to_absent(self, monkeypatch):
        import gateway.status as gs

        def _boom(refresh=False):
            raise RuntimeError("no build info")

        monkeypatch.setattr("hermes_cli.build_info.get_code_identity", _boom)
        record = gs._build_runtime_status_record()
        # Must not raise, and must not stamp bogus values.
        assert "code_sha" not in record
        assert record["gateway_state"] == "starting"


class TestCodeIdentity:
    def test_get_code_identity_shape(self):
        from hermes_cli.build_info import get_code_identity

        identity = get_code_identity(refresh=True)
        assert set(identity) == {"sha", "short_sha", "version", "source"}
        # Running from a git checkout in CI/dev: sha resolves via git.
        if identity["sha"]:
            assert identity["short_sha"] == identity["sha"][:8]
            assert identity["source"] in ("git", "build-file")

    def test_get_code_identity_cached(self):
        from hermes_cli.build_info import get_code_identity

        first = get_code_identity(refresh=True)
        second = get_code_identity()
        assert first == second
        # returned dicts are copies, not the shared cache
        second["sha"] = "mutated"
        assert get_code_identity()["sha"] == first["sha"]
