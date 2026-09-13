"""CLI metadata dispatch keeps credential implementations lazy."""
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

class TestUpdateEntrypointImportHygiene:
    """Subprocess-verified: the CLI/update path never eagerly loads crypto.

    Empirical sys.modules assertions in a clean interpreter — the technique
    that pinned #86735's root cause.  If any future module-level import of
    cryptography (or another self-locking native module) creeps back into
    the startup path, these fail before a Windows user ever hits the loop.
    """

    def _run(self, code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=120,
        )

    def test_import_main_does_not_load_self_locking_modules(self):
        result = self._run(
            textwrap.dedent(
                """
                import sys
                import hermes_cli.main
                assert "cryptography.hazmat.bindings._rust" not in sys.modules
                print("OK")
                """
            )
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout

    def test_update_dispatch_does_not_load_cryptography(self):
        result = self._run(
            textwrap.dedent(
                """
                import sys
                from unittest.mock import patch
                sys.argv = ["hermes", "update", "--check"]
                import hermes_cli.main as m
                with patch("hermes_cli.update_cmd._cmd_update_check", lambda *a, **k: 0):
                    try:
                        m.main()
                    except SystemExit:
                        pass
                assert "cryptography.hazmat.bindings._rust" not in sys.modules, (
                    "update dispatch eagerly loaded cryptography._rust"
                )
                print("OK")
                """
            )
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout
