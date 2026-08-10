"""Test: _install_python_dependencies_with_optional_fallback mit stale VIRTUAL_ENV.

Simuliert den heutigen Crash: pip/System-Python-Install, wo PROJECT_ROOT =
site-packages ist und VIRTUAL_ENV=PROJECT_ROOT/venv nicht existiert.
"""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import hermes_cli.main as main_mod


class StaleVirtualEnvTest(unittest.TestCase):
    def _call(self, uv_bin, venv_path, fake_executable):
        """Führe die Funktion mit gemocktem uv/env aus und fange den subprocess-Aufruf."""
        captured = []

        real_run_quarantined = main_mod._run_quarantined_install

        def fake_quarantine(cmd, *, env=None, scripts_dir=None):
            captured.append((list(cmd), dict(env or {})))
            # Nicht wirklich installieren; Verhalten wie Erfolg simulieren.
            return None

        def fake_verify(prefix, *, env=None):
            return None

        with mock.patch.object(main_mod, "_run_quarantined_install", fake_quarantine), \
             mock.patch.object(main_mod, "_verify_console_scripts_installed", fake_verify), \
             mock.patch.object(main_mod, "_venv_scripts_dir", return_value=None), \
             mock.patch.object(main_mod, "_is_windows", return_value=False), \
             mock.patch.object(main_mod.sys, "executable", fake_executable), \
             mock.patch.object(main_mod, "PROJECT_ROOT", Path("/fake/project")):
            main_mod._install_python_dependencies_with_optional_fallback(
                [str(uv_bin), "pip"],
                env={"VIRTUAL_ENV": str(venv_path)},
                group="all",
            )
        return captured

    def test_stale_virtualenv_pins_python(self):
        """VIRTUAL_ENV zeigt auf nicht-existierendes venv -> --python sys.executable."""
        captured = self._call(
            uv_bin=Path("/fake/uv"),
            venv_path=Path("/fake/project/venv"),  # existiert nicht
            fake_executable="/fake/python311/python.exe",
        )
        self.assertTrue(captured, "kein subprocess-Aufruf erfasst")
        cmd, env = captured[0]
        # --python muss nach 'install' stehen: uv pip install --python <exe> ...
        self.assertIn("install", cmd)
        self.assertIn("--python", cmd)
        self.assertEqual(cmd[cmd.index("--python") + 1], "/fake/python311/python.exe")
        # VIRTUAL_ENV aus env entfernt
        self.assertNotIn("VIRTUAL_ENV", env)

    def test_existing_virtualenv_keeps_env(self):
        """VIRTUAL_ENV zeigt auf existierendes venv -> unverändert, kein --python."""
        real_venv = Path(sys.executable).resolve().parent.parent
        if not real_venv.is_dir():
            self.skipTest("kein echtes venv im Testlauf verfügbar")
        captured = self._call(
            uv_bin=Path("/fake/uv"),
            venv_path=real_venv,
            fake_executable=sys.executable,
        )
        cmd, env = captured[0]
        self.assertNotIn("--python", cmd)
        self.assertEqual(env.get("VIRTUAL_ENV"), str(real_venv))


if __name__ == "__main__":
    unittest.main(verbosity=2)
