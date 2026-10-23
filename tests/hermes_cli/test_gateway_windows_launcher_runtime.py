"""Gateway launchers must use the installation bootstrap, not a dependency snapshot.

These focused contract tests also run with stdlib unittest when a repair must not
install pytest or sync the live package-manager environment.
"""

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_cli import _launchers, gateway_windows


class GatewayLauncherRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "source checkout"
        self.project.mkdir()
        self.home = self.root / "home with spaces" / "profiles" / "work"
        self.home.mkdir(parents=True)
        self.python = str(Path(getattr(sys, "_base_executable", sys.executable)))
        self.gateway = types.ModuleType("hermes_cli.gateway")
        self.gateway.PROJECT_ROOT = self.project
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"hermes_cli.gateway": self.gateway}).start()
        patch.object(gateway_windows, "_hermes_home", return_value=self.home).start()
        self.store = patch.object(
            _launchers, "resolve_store_python", return_value=Path(self.python)
        ).start()

    def test_generated_launchers_use_the_same_bootstrap_as_the_cli(self):
        args = ["--profile", "work", "gateway", "run"]
        command = _launchers.runtime_command(
            self.project, args, python=self.python, home=str(self.home)
        )
        expected_vbs = gateway_windows._quote_vbs_string(subprocess.list2cmdline(command))
        expected_cmd = " ".join(gateway_windows._quote_cmd_script_arg(arg) for arg in command)
        for kind, renderer, expected in (
            ("vbs", gateway_windows._build_gateway_vbs_script, expected_vbs),
            ("cmd", gateway_windows._build_gateway_cmd_script, expected_cmd),
        ):
            with self.subTest(kind=kind):
                content = renderer(self.python, str(self.home), str(self.home), "--profile work")
                self.assertIn(expected, content)
                self.assertNotIn('set "PYTHONPATH=', content)
                self.assertNotIn('env.Item("PYTHONPATH") =', content)
                self.assertNotIn('set "VIRTUAL_ENV=', content)
                self.assertNotIn('env.Item("VIRTUAL_ENV") =', content)
                self.assertNotIn("--replace", content)
                self.assertIn(str(self.home), content)
                self.assertTrue(content.endswith("\r\n"))
                if kind == "vbs":
                    self.assertNotIn("cmd.exe", content.lower())
                    self.assertIn(", 0, False", content)

    def test_direct_start_uses_the_same_runtime_as_generated_launchers(self):
        settings = (self.python, str(self.home), str(self.home), "--profile work")
        with patch.object(gateway_windows, "_launcher_settings", return_value=settings), \
                patch.object(gateway_windows, "_assert_windows"):
            argv, cwd, env = gateway_windows._build_gateway_argv()
        expected = _launchers.runtime_command(
            self.project, ["--profile", "work", "gateway", "run"],
            python=self.python, home=str(self.home),
        )
        self.assertEqual(argv, expected)
        self.assertEqual(cwd, str(self.home))
        self.assertEqual(env, {"HERMES_HOME": str(self.home), **dict(gateway_windows._GATEWAY_ENV)})

    def test_managed_command_ignores_a_retired_dependency_interpreter(self):
        retired = self.root / "environments" / "retired" / "Scripts" / "python.exe"
        argv, env = gateway_windows._gateway_launch_spec(str(retired), str(self.home), "")
        self.assertEqual(argv[0], self.python)
        self.assertEqual(argv[1:3], ["-I", "-c"])
        self.assertEqual(argv[-2:], ["gateway", "run"])
        self.assertNotIn(str(retired), " ".join(argv))
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("VIRTUAL_ENV", env)
        self.store.assert_called_with(self.project, publication=True)

    def test_external_install_keeps_its_interpreter_and_import_environment(self):
        self.store.return_value = None
        external = self.root / "external venv" / "Scripts" / "python.exe"
        argv, env = gateway_windows._gateway_launch_spec(str(external), str(self.home), "--profile work")
        self.assertEqual(argv, [str(external), "-m", "hermes_cli.main", "--profile", "work", "gateway", "run"])
        self.assertEqual(env["VIRTUAL_ENV"], str(external.parent.parent))
        self.assertEqual(env["PYTHONPATH"], str(self.project))
        cmd = gateway_windows._build_gateway_cmd_script(str(external), str(self.home), str(self.home), "")
        vbs = gateway_windows._build_gateway_vbs_script(str(external), str(self.home), str(self.home), "")
        self.assertIn(f'set "PYTHONPATH={self.project};%PYTHONPATH%"', cmd)
        self.assertIn('existing_pp = env.Item("PYTHONPATH")', vbs)
        self.assertIn(" & existing_pp", vbs)
        self.assertNotIn("-I -c", cmd)
        self.assertNotIn("-I -c", vbs)

    def test_one_command_bootstraps_each_start_without_ambient_python_state(self):
        # A tiny checkout fixture proves the emitted bootstrap is actually executed.
        # It reads a selection at child start, not while the launcher is generated.
        (self.project / "hermes_bootstrap.py").write_text(
            "from pathlib import Path\n"
            "SELECTED = Path(__file__).with_name('selection').read_text()\n",
            encoding="utf-8",
        )
        package = self.project / "hermes_cli"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "main.py").write_text(
            "import json, os, sys, hermes_bootstrap\n"
            "print(json.dumps({'selected': hermes_bootstrap.SELECTED, "
            "'argv': sys.argv[1:], 'home': os.environ['HERMES_HOME'], "
            "'dirty': [k for k in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV') if k in os.environ]}))\n",
            encoding="utf-8",
        )
        argv, env = gateway_windows._gateway_launch_spec(self.python, str(self.home), "--profile 'work team'")
        child_env = {**os.environ, **env, "PYTHONHOME": str(self.root / "invalid-home"),
                     "PYTHONPATH": str(self.root / "wrong-source"), "VIRTUAL_ENV": str(self.root / "retired")}
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        for selection in ("first", "second"):
            (self.project / "selection").write_text(selection, encoding="utf-8")
            result = subprocess.run(argv, cwd=self.home, env=child_env, capture_output=True,
                                    text=True, encoding="utf-8", timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {
                "selected": selection, "argv": ["--profile", "work team", "gateway", "run"],
                "home": str(self.home), "dirty": [],
            })

    def test_write_task_script_renders_both_bootstrap_launchers_atomically(self):
        target = self.root / "service fixture" / "Hermes_Gateway_work.cmd"
        target.parent.mkdir()
        settings = (self.python, str(self.home), str(self.home), "--profile work")
        with patch.object(gateway_windows, "_assert_windows"), \
                patch.object(gateway_windows, "_launcher_settings", return_value=settings), \
                patch.object(gateway_windows, "get_task_script_path", return_value=target):
            self.assertEqual(gateway_windows._write_task_script(), target)
        cmd = target.read_bytes()
        vbs = target.with_suffix(".vbs").read_bytes()
        self.assertIn(b"-I -c", cmd)
        self.assertIn(b"-I -c", vbs)
        self.assertIn(b", 0, False\r\n", vbs)
        self.assertNotIn(b"cmd.exe", vbs)
        self.assertEqual({p.name for p in target.parent.iterdir()}, {target.name, target.with_suffix(".vbs").name})
        self.assertNotIn(b"\n", cmd.replace(b"\r\n", b""))
        self.assertNotIn(b"\n", vbs.replace(b"\r\n", b""))

    def test_bootstrap_failure_never_falls_back_to_raw_gateway(self):
        (self.project / "hermes_bootstrap.py").write_text("raise RuntimeError('broken bootstrap fixture')\n", encoding="utf-8")
        argv, env = gateway_windows._gateway_launch_spec(self.python, str(self.home), "")
        result = subprocess.run(argv, cwd=self.home, env={**os.environ, **env}, capture_output=True,
                                text=True, encoding="utf-8", timeout=30, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("broken bootstrap fixture", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
