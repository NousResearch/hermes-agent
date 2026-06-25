"""Tests for gateway code-skew detection (stale-checkout guard).

Companion to ``tests/test_stale_utils_module_import.py``: that test proves the
crash; these prove the guard that turns it into a clear "restart the gateway"
message before a model switch can hit it.
"""

import pytest

from gateway import code_skew


@pytest.fixture(autouse=True)
def _reset_boot_fingerprint(monkeypatch):
    """Each test starts with no recorded boot fingerprint."""
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)


class TestDetectCodeSkew:
    def test_no_boot_fingerprint_means_no_skew(self, monkeypatch):
        # Nothing recorded (e.g. non-git install) -> never a false positive.
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def456")
        assert code_skew.detect_code_skew() is None

    def test_unchanged_checkout_is_not_skew(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()
        assert code_skew.detect_code_skew() is None

    def test_drift_is_detected_with_short_revs(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()

        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:def4567890123")
        skew = code_skew.detect_code_skew()
        assert skew == ("abc1234567", "def4567890")

    def test_unreadable_current_rev_does_not_false_positive(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567890")
        code_skew.record_boot_fingerprint()

        monkeypatch.setattr(code_skew, "_fingerprint", lambda: None)
        assert code_skew.detect_code_skew() is None

    def test_record_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:first")
        code_skew.record_boot_fingerprint()
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:second")
        code_skew.record_boot_fingerprint()  # must not overwrite the boot snapshot
        assert code_skew._boot_fingerprint == "git:refs/heads/main:first"


class TestShort:
    def test_shortens_long_sha(self):
        assert code_skew._short("git:refs/heads/main:abcdef0123456789") == "abcdef0123"

    def test_keeps_unresolved_marker(self):
        assert code_skew._short("git:refs/heads/main:unresolved") == "unresolved"

    def test_passes_short_sha_through_untruncated(self):
        assert code_skew._short("git:HEAD:abc1234") == "abc1234"


class TestModelSwitchSkewGuard:
    def test_guard_returns_none_without_skew(self, monkeypatch):
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
        assert slash_commands._model_switch_skew_guard() is None

    def test_guard_message_names_revs_and_restart(self, monkeypatch):
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = slash_commands._model_switch_skew_guard()
        assert msg is not None
        assert "abc1234567" in msg
        assert "def4567890" in msg
        assert "hermes gateway restart" in msg

    def test_guard_triggers_graceful_restart_when_runner_provided(self, monkeypatch):
        """When a runner is passed, the guard triggers a graceful restart
        instead of only telling the user to restart manually."""
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))

        calls = []

        class FakeRunner:
            def request_restart(self, *, detached=False, via_service=False):
                calls.append((detached, via_service))
                return True

        msg = slash_commands._model_switch_skew_guard(runner=FakeRunner())
        assert msg is not None
        assert "Restarting to load the new code" in msg
        assert "re-run /model after restart" in msg
        assert len(calls) == 1  # request_restart was called exactly once

    def test_guard_falls_back_to_manual_when_restart_fails(self, monkeypatch):
        """If request_restart returns False (e.g. already restarting), the
        guard falls back to the manual restart message."""
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))

        class FakeRunner:
            def request_restart(self, *, detached=False, via_service=False):
                return False  # restart already in progress

        msg = slash_commands._model_switch_skew_guard(runner=FakeRunner())
        assert msg is not None
        assert "hermes gateway restart" in msg  # manual fallback

    def test_guard_falls_back_to_manual_without_runner(self, monkeypatch):
        """Without a runner, the guard returns the manual restart message
        (preserving backwards compatibility for any caller not yet passing one)."""
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
        msg = slash_commands._model_switch_skew_guard()
        assert msg is not None
        assert "hermes gateway restart" in msg

    def test_guard_passes_correct_restart_flags_for_env(self, monkeypatch):
        """The guard must mirror _handle_restart_command's environment
        detection: under launchd/systemd or in a container, use the service
        path (via_service=True); otherwise detached=True. The defaults
        (detached=False, via_service=False) match neither path and can
        leave the gateway stopped instead of restarted."""
        from gateway import slash_commands

        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))

        calls = []

        class FakeRunner:
            def request_restart(self, *, detached=False, via_service=False):
                calls.append((detached, via_service))
                return True

        # Simulate macOS launchd (XPC_SERVICE_NAME set, not "0")
        monkeypatch.setenv("XPC_SERVICE_NAME", "com.apple.xpc.launchd")
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        slash_commands._model_switch_skew_guard(runner=FakeRunner())
        assert calls == [(False, True)], f"service path expected, got {calls}"

        # Simulate plain shell (no service manager) → detached path
        calls.clear()
        monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        slash_commands._model_switch_skew_guard(runner=FakeRunner())
        assert calls == [(True, False)], f"detached path expected, got {calls}"


class TestPurgeStalePycache:
    """Tests for purge_stale_pycache — purging __pycache__ when the checkout
    advanced between the previous gateway boot and the current one."""

    def test_no_purge_when_previous_fingerprint_missing(self, monkeypatch, tmp_path):
        """No persisted fingerprint file → nothing to compare → no purge."""
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567")
        monkeypatch.setattr(code_skew, "_last_boot_fingerprint_file", tmp_path / "nonexistent")
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", tmp_path)
        assert code_skew.purge_stale_pycache() is False

    def test_no_purge_when_fingerprints_match(self, monkeypatch, tmp_path):
        """Same checkout as previous boot → no drift → no purge."""
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        fp_file.write_text("git:refs/heads/main:abc1234567")
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:abc1234567")
        monkeypatch.setattr(code_skew, "_last_boot_fingerprint_file", fp_file)
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", tmp_path)
        assert code_skew.purge_stale_pycache() is False

    def test_purges_pyc_files_on_drift(self, monkeypatch, tmp_path):
        """Drift detected → .pyc files inside __pycache__ are deleted."""
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        fp_file.write_text("git:refs/heads/main:oldhash123")
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:newhash456")
        monkeypatch.setattr(code_skew, "_last_boot_fingerprint_file", fp_file)
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", tmp_path)

        # Create a fake __pycache__ with .pyc files
        pycache = tmp_path / "gateway" / "__pycache__"
        pycache.mkdir(parents=True)
        (pycache / "slash_commands.cpython-311.pyc").write_bytes(b"stale bytecode")
        (pycache / "run.cpython-311.pyc").write_bytes(b"stale bytecode")

        assert code_skew.purge_stale_pycache() is True
        assert not (pycache / "slash_commands.cpython-311.pyc").exists()
        assert not (pycache / "run.cpython-311.pyc").exists()

    def test_skips_venv_pycache(self, monkeypatch, tmp_path):
        """.pyc inside .venv are not purged (they're managed by pip)."""
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        fp_file.write_text("git:refs/heads/main:oldhash123")
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: "git:refs/heads/main:newhash456")
        monkeypatch.setattr(code_skew, "_last_boot_fingerprint_file", fp_file)
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", tmp_path)

        venv_pycache = tmp_path / ".venv" / "lib" / "__pycache__"
        venv_pycache.mkdir(parents=True)
        (venv_pycache / "site.cpython-311.pyc").write_bytes(b"should not be touched")

        code_skew.purge_stale_pycache()
        assert (venv_pycache / "site.cpython-311.pyc").exists()

    def test_no_purge_when_fingerprint_unreadable(self, monkeypatch, tmp_path):
        """If _fingerprint returns None (non-git), no purge."""
        monkeypatch.setattr(code_skew, "_fingerprint", lambda: None)
        monkeypatch.setattr(code_skew, "_last_boot_fingerprint_file", tmp_path / "x")
        monkeypatch.setattr(code_skew, "_PROJECT_ROOT", tmp_path)
        assert code_skew.purge_stale_pycache() is False


class TestPersistBootFingerprint:
    def test_persists_fingerprint_to_file(self, monkeypatch, tmp_path):
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        monkeypatch.setattr(code_skew, "_last_boot_fingerprint_file", fp_file)
        code_skew._persist_boot_fingerprint("git:refs/heads/main:abcdef1234")
        assert fp_file.read_text() == "git:refs/heads/main:abcdef1234"

    def test_does_not_persist_none(self, monkeypatch, tmp_path):
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        monkeypatch.setattr(code_skew, "_last_boot_fingerprint_file", fp_file)
        code_skew._persist_boot_fingerprint(None)
        assert not fp_file.exists()
