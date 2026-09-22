"""Test: _install_python_dependencies_with_optional_fallback with stale VIRTUAL_ENV.

Simulates the real crash: a pip/system-Python install where PROJECT_ROOT is
site-packages and VIRTUAL_ENV=PROJECT_ROOT/venv does not exist.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import hermes_cli.main as main_mod
from hermes_cli import main_install_repair


class StaleVirtualEnvTest(unittest.TestCase):
    def _call(self, uv_cmd, venv_path, fake_executable, is_windows=False):
        """Run the function with a mocked uv/env and capture the subprocess call."""
        captured = []

        def fake_quarantine(cmd, *, env=None, scripts_dir=None, strict_quarantine=False):
            captured.append((list(cmd), dict(env or {}), scripts_dir))
            return None

        def fake_verify(prefix, *, env=None):
            return None

        with mock.patch.object(main_install_repair, "_run_quarantined_install", fake_quarantine), \
             mock.patch.object(main_install_repair, "_verify_console_scripts_installed", fake_verify), \
             mock.patch.object(main_install_repair, "_venv_scripts_dir", return_value=None), \
             mock.patch.object(main_install_repair, "_is_windows", return_value=is_windows), \
             mock.patch.object(main_install_repair.sys, "executable", fake_executable), \
             mock.patch.object(main_mod, "PROJECT_ROOT", Path("/fake/project")):
            main_install_repair._install_python_dependencies_with_optional_fallback(
                list(uv_cmd),
                env={"VIRTUAL_ENV": str(venv_path)},
                group="all",
            )
        return captured

    def test_stale_virtualenv_pins_python(self):
        """VIRTUAL_ENV points at a nonexistent venv -> --python sys.executable."""
        captured = self._call(
            uv_cmd=[Path("/fake/uv"), "pip"],
            venv_path=Path("/fake/project/venv"),  # does not exist
            fake_executable="/fake/python311/python.exe",
        )
        self.assertTrue(captured, "no subprocess call captured")
        cmd, env, _ = captured[0]
        # --python must come after 'install': uv pip install --python <exe> ...
        self.assertIn("install", cmd)
        self.assertIn("--python", cmd)
        self.assertEqual(cmd[cmd.index("--python") + 1], "/fake/python311/python.exe")
        # VIRTUAL_ENV removed from env
        self.assertNotIn("VIRTUAL_ENV", env)

    def test_existing_virtualenv_keeps_env(self):
        """VIRTUAL_ENV points at an existing venv -> unchanged, no --python."""
        real_venv = Path(sys.executable).resolve().parent.parent
        if not real_venv.is_dir():
            self.skipTest("no real venv available in this test run")
        captured = self._call(
            uv_cmd=[Path("/fake/uv"), "pip"],
            venv_path=real_venv,
            fake_executable=sys.executable,
        )
        cmd, env, _ = captured[0]
        self.assertNotIn("--python", cmd)
        self.assertEqual(env.get("VIRTUAL_ENV"), str(real_venv))

    def test_python_dash_m_uv_is_detected(self):
        """python -m uv must also trigger the pin (naive basename check misses it)."""
        captured = self._call(
            uv_cmd=[Path("/fake/python"), "-m", "uv", "pip"],
            venv_path=Path("/fake/project/venv"),  # does not exist
            fake_executable="/fake/python311/python.exe",
        )
        self.assertTrue(captured, "no subprocess call captured")
        cmd, _, _ = captured[0]
        self.assertIn("--python", cmd)
        self.assertEqual(cmd[cmd.index("--python") + 1], "/fake/python311/python.exe")

    def test_existing_python_flag_wins(self):
        """A caller-supplied --python is not duplicated by the pin."""
        captured = self._call(
            uv_cmd=[Path("/fake/uv"), "pip"],
            venv_path=Path("/fake/project/venv"),
            fake_executable="/fake/python311/python.exe",
        )
        # Force the caller path through a manual pin with a pre-existing flag.
        args = ["install", "--python", "/caller/choice/python.exe", "hermes"]
        pinned = main_install_repair._insert_python_pin(args)
        self.assertEqual(pinned, args, "existing --python must win")
        self.assertEqual(pinned.count("--python"), 1)

    def test_windows_pins_quarantine_to_interpreter_scripts_dir(self):
        """On Windows with a missing project venv, quarantine must target the
        interpreter's Scripts dir (where the shims actually live), not None."""
        fake_scripts = Path("/fake/python311/Scripts")
        with mock.patch.object(main_install_repair, "_interpreter_scripts_dir", return_value=fake_scripts
        ):
            captured = self._call(
                uv_cmd=[Path("/fake/uv"), "pip"],
                venv_path=Path("/fake/project/venv"),
                fake_executable="/fake/python311/python.exe",
                is_windows=True,
            )
        self.assertTrue(captured, "no subprocess call captured")
        _, _, scripts_dir = captured[0]
        self.assertEqual(scripts_dir, fake_scripts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
