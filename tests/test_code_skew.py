"""Tests for gateway code-skew detection (stale-checkout guard).

Companion to ``tests/test_stale_utils_module_import.py``: that test proves the
crash; these prove the guard that turns it into a clear "restart the gateway"
message before a model switch can hit it.
"""

import ast
import subprocess
import sys
from pathlib import Path

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


class TestBootstrapBytecodePurge:
    def test_purges_stale_gateway_bytecode_before_gateway_run_import(
        self, monkeypatch, tmp_path
    ):
        from hermes_cli import gateway_bootstrap

        project_root = tmp_path / "repo"
        pycache = project_root / "gateway" / "__pycache__"
        pycache.mkdir(parents=True)
        stale_pyc = pycache / "run.cpython-311.pyc"
        stale_pyc.write_bytes(b"stale")
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()
        (hermes_home / ".gateway_boot_fingerprint").write_text("git:HEAD:oldsha")

        monkeypatch.setattr(gateway_bootstrap, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(gateway_bootstrap, "get_hermes_home", lambda: hermes_home)
        monkeypatch.setattr(
            gateway_bootstrap.subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "newsha\n", ""),
        )

        assert gateway_bootstrap.purge_stale_gateway_pycache_before_import() is True
        assert not stale_pyc.exists()

    def test_every_supported_launcher_bootstraps_before_gateway_run_import(self):
        project_root = Path(__file__).parents[1]
        launchers = (
            project_root / "cli.py",
            project_root / "hermes_cli" / "gateway.py",
            project_root / "scripts" / "hermes-gateway",
        )

        for launcher in launchers:
            tree = ast.parse(launcher.read_text(encoding="utf-8"), filename=str(launcher))
            bootstrap_imports = [
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module == "hermes_cli.gateway_bootstrap"
            ]
            purge_calls = [
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "purge_stale_gateway_pycache_before_import"
            ]
            gateway_run_imports = [
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "gateway.run"
            ]

            assert bootstrap_imports, f"{launcher} must import the pre-import bootstrap"
            assert purge_calls, f"{launcher} must invoke the pre-import bootstrap"
            assert gateway_run_imports, f"{launcher} must import gateway.run"
            assert min(bootstrap_imports) < min(purge_calls) < min(gateway_run_imports)
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

        # Simulate an externally supervised gateway → service path.
        calls.clear()
        monkeypatch.setenv("HERMES_GATEWAY_EXTERNAL_SUPERVISOR", "true")
        slash_commands._model_switch_skew_guard(runner=FakeRunner())
        assert calls == [(False, True)], f"service path expected, got {calls}"


class TestPersistBootFingerprint:
    def test_persists_fingerprint_to_file(self, monkeypatch, tmp_path):
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        monkeypatch.setattr(code_skew, "_fingerprint_file", lambda: fp_file)
        code_skew._persist_boot_fingerprint("git:refs/heads/main:abcdef1234")
        assert fp_file.read_text() == "git:refs/heads/main:abcdef1234"

    def test_does_not_persist_none(self, monkeypatch, tmp_path):
        fp_file = tmp_path / ".gateway_boot_fingerprint"
        monkeypatch.setattr(code_skew, "_fingerprint_file", lambda: fp_file)
        code_skew._persist_boot_fingerprint(None)
        assert not fp_file.exists()

    def test_direct_module_execution_has_documented_cli_migration(self):
        result = subprocess.run(
            [sys.executable, "-m", "gateway.run"],
            capture_output=True,
            cwd=Path(__file__).parents[1],
            text=True,
            timeout=30,
        )

        assert result.returncode == 1
        assert "Direct module execution is unsupported" in result.stderr
        assert "hermes gateway run" in result.stderr
