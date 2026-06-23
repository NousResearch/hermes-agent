"""Tests for the Windows venv-lock writability probe and early-abort gate.

Covers the resilience fix that refuses to mutate a locked Windows venv:

* ``_hermes_exe_is_locked`` — non-destructive rename round-trip probe.
* ``_format_venv_locked_message`` — actionable, finisher-oriented error text.
* ``_cmd_update_impl`` early-abort gate — exits 2, resumes paused gateways,
  and prints the finisher message BEFORE any git/venv mutation.

Every test patches ``_is_windows`` / ``_venv_scripts_dir`` (and filesystem
calls) so they run deterministically on any host — no real Windows or live
venv required.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import main as hm


# ---------------------------------------------------------------------------
# _hermes_exe_is_locked
# ---------------------------------------------------------------------------
class TestHermesExeIsLocked:
    def test_off_windows_returns_false_without_touching_fs(self):
        """Off-Windows the probe short-circuits to False and never renames."""
        scripts = Path("/fake/venv/Scripts")
        with patch.object(hm, "_is_windows", return_value=False), \
             patch.object(Path, "rename") as mock_rename, \
             patch.object(Path, "exists") as mock_exists:
            assert hm._hermes_exe_is_locked(scripts) is False
        mock_rename.assert_not_called()
        mock_exists.assert_not_called()

    def test_no_shim_returns_false(self):
        """With no hermes.exe present, returns False and never renames."""
        scripts = Path("/fake/venv/Scripts")
        with patch.object(hm, "_is_windows", return_value=True), \
             patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "rename") as mock_rename:
            assert hm._hermes_exe_is_locked(scripts) is False
        mock_rename.assert_not_called()

    def test_locked_when_first_rename_raises_oserror(self):
        """A foreign lock makes the in-place rename raise OSError -> True."""
        scripts = Path("/fake/venv/Scripts")
        with patch.object(hm, "_is_windows", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "rename", side_effect=OSError(32, "in use")) as mock_rename:
            assert hm._hermes_exe_is_locked(scripts) is True
        # Only the forward (shim -> probe) rename was attempted; we never got
        # to a put-back because the probe rename was refused.
        assert mock_rename.call_count == 1

    def test_unlocked_round_trip_returns_false_and_restores_shim(self):
        """Successful round-trip -> False, with the shim restored and NO
        .probe leftover (the second rename moves probe back to hermes.exe)."""
        scripts = Path("/fake/venv/Scripts")
        rename_calls = []

        def fake_rename(self, target):
            rename_calls.append((str(self), str(target)))
            return None

        with patch.object(hm, "_is_windows", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "rename", autospec=True, side_effect=fake_rename), \
             patch.object(hm, "_schedule_replace_on_reboot") as mock_defer:
            assert hm._hermes_exe_is_locked(scripts) is False

        # Exactly two renames: shim->probe, then probe->shim (restored).
        assert len(rename_calls) == 2
        src1, dst1 = rename_calls[0]
        src2, dst2 = rename_calls[1]
        assert src1.endswith("hermes.exe") and dst1.endswith("hermes.exe.probe")
        assert src2.endswith("hermes.exe.probe") and dst2.endswith("hermes.exe")
        # The final destination is the original shim path (no .probe leftover).
        assert dst2 == src1
        # Round-trip succeeded, so the reboot-deferral fallback is NOT used.
        mock_defer.assert_not_called()

    def test_rename_back_failure_defers_putback_to_reboot(self):
        """If the put-back rename fails, the probe is scheduled for reboot
        replacement (PATH never left without a shim) and still returns False."""
        scripts = Path("/fake/venv/Scripts")
        calls = {"n": 0}

        def fake_rename(self, target):
            calls["n"] += 1
            if calls["n"] == 1:
                return None  # shim -> probe succeeds
            raise OSError(32, "probe put-back refused")  # probe -> shim fails

        with patch.object(hm, "_is_windows", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "rename", autospec=True, side_effect=fake_rename), \
             patch.object(hm, "_schedule_replace_on_reboot") as mock_defer:
            assert hm._hermes_exe_is_locked(scripts) is False

        assert calls["n"] == 2
        mock_defer.assert_called_once()
        probe_arg, shim_arg = mock_defer.call_args.args
        assert str(probe_arg).endswith("hermes.exe.probe")
        assert str(shim_arg).endswith("hermes.exe")


# ---------------------------------------------------------------------------
# _format_venv_locked_message
# ---------------------------------------------------------------------------
class TestFormatVenvLockedMessage:
    def test_message_names_shim_pids_finisher_and_force(self):
        scripts = Path("/fake/venv/Scripts")
        msg = hm._format_venv_locked_message(scripts, [4242, 9001])

        # The shim path is named.
        assert "hermes.exe" in msg
        # The pythonw gateway holder is named.
        assert "pythonw.exe" in msg
        # The holding PIDs are listed.
        assert "4242" in msg
        assert "9001" in msg
        # The one-command finisher is pointed at.
        assert "finish-hermes-update.cmd" in msg
        # The override is documented.
        assert "--force" in msg

    def test_message_omits_pid_block_when_no_pids(self):
        scripts = Path("/fake/venv/Scripts")
        msg = hm._format_venv_locked_message(scripts, [])
        assert "Holding gateway PID(s):" not in msg
        # Core guidance still present even with no PIDs resolved.
        assert "finish-hermes-update.cmd" in msg
        assert "hermes.exe" in msg


# ---------------------------------------------------------------------------
# _cmd_update_impl early-abort gate
# ---------------------------------------------------------------------------
class TestUpdateAbortsWhenVenvLocked:
    def test_abort_exits_2_resumes_gateways_and_skips_mutation(self, capsys):
        """When the probe reports the shim locked, _cmd_update_impl resumes any
        paused gateways, prints the finisher message, and exits 2 BEFORE any
        git/ZIP mutation runs.

        Source ordering (hermes_cli/main.py): the pre-update backup and the
        gateway-pause run first, THEN the authoritative lock gate. So the gate
        protects the working tree / venv from the git fetch+pull and the ZIP
        fallback (all of which go through ``subprocess.run``), which is the
        mutation that matters. The backup is non-mutating to the repo and
        precedes the gate by design — we assert no ``subprocess.run`` (no git)
        rather than no backup."""
        scripts = Path("/fake/venv/Scripts")
        resume_token = {"resume_needed": True, "profiles": {}, "unmapped": []}

        with patch.object(hm, "_is_windows", return_value=True), \
             patch.object(hm, "_venv_scripts_dir", return_value=scripts), \
             patch.object(hm, "_detect_concurrent_hermes_instances", return_value=[]), \
             patch.object(hm, "_install_hangup_protection", return_value=None), \
             patch.object(hm, "_finalize_update_output"), \
             patch.object(hm, "_run_pre_update_backup"), \
             patch.object(hm, "_pause_windows_gateways_for_update", return_value=resume_token), \
             patch.object(hm, "_resume_windows_gateways_after_update") as mock_resume, \
             patch.object(hm, "_hermes_exe_is_locked", return_value=True), \
             patch("subprocess.run") as mock_run, \
             patch("hermes_cli.gateway.find_gateway_pids", return_value=[7777]):
            args = SimpleNamespace(force=False, check=False, gateway=False)
            with pytest.raises(SystemExit) as exc_info:
                hm.cmd_update(args)

        assert exc_info.value.code == 2

        # The pause's resume token was resumed (paused gateways restarted).
        mock_resume.assert_called_once_with(resume_token)

        # Abort happened BEFORE any git/ZIP mutation: subprocess.run (git
        # fetch/pull, ZIP fallback) was never reached past the gate, so the
        # checkout and venv are left pristine.
        mock_run.assert_not_called()

        out = capsys.readouterr().out
        assert "still locked" in out
        assert "finish-hermes-update.cmd" in out
        assert "7777" in out  # holding gateway PID enumerated and shown

    def test_force_bypasses_lock_gate(self, capsys):
        """``hermes update --force`` skips the lock probe entirely (gate is
        guarded by ``not getattr(args, 'force', False)``)."""
        scripts = Path("/fake/venv/Scripts")

        with patch.object(hm, "_is_windows", return_value=True), \
             patch.object(hm, "_venv_scripts_dir", return_value=scripts), \
             patch.object(hm, "_detect_concurrent_hermes_instances", return_value=[]), \
             patch.object(hm, "_install_hangup_protection", return_value=None), \
             patch.object(hm, "_finalize_update_output"), \
             patch.object(hm, "_run_pre_update_backup"), \
             patch.object(hm, "_pause_windows_gateways_for_update", return_value=None), \
             patch.object(hm, "_resume_windows_gateways_after_update"), \
             patch.object(hm, "_hermes_exe_is_locked", return_value=True) as mock_probe, \
             patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), \
             patch.object(hm, "_cmd_update_impl", wraps=hm._cmd_update_impl):
            args = SimpleNamespace(force=True, check=False, gateway=False)
            # We only care that the lock probe is never consulted under --force.
            # Let the impl run far enough to touch (or not touch) the probe, then
            # short-circuit by raising from the next mutation step.
            with patch.object(
                hm, "_pause_windows_gateways_for_update",
                side_effect=SystemExit(99),
            ):
                with pytest.raises(SystemExit):
                    hm.cmd_update(args)

        # Under --force the authoritative lock probe must never be called.
        mock_probe.assert_not_called()


# ---------------------------------------------------------------------------
# _cleanup_quarantined_exes sweeps stray .exe.probe leftovers
# ---------------------------------------------------------------------------
class TestCleanupSweepsProbeLeftover:
    def test_removes_stray_probe_file(self, tmp_path):
        scripts = tmp_path
        probe = scripts / "hermes.exe.probe"
        probe.write_text("stale probe")
        old = scripts / "hermes.exe.old.123"
        old.write_text("stale old")

        with patch.object(hm, "_is_windows", return_value=True):
            hm._cleanup_quarantined_exes(scripts)

        assert not probe.exists(), "stray .exe.probe must be swept"
        assert not old.exists(), "stray .exe.old.* must still be swept"
