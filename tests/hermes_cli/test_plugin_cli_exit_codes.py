"""Regression coverage for plugin CLI handler exit codes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _write_exit_code_plugin(hermes_home: Path) -> None:
    plugin_dir = hermes_home / "plugins" / "exit-code-fixture"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: exit-code-fixture\nprovides_cli:\n  - exit-fixture\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    def setup(parser):\n"
        "        parser.add_argument('--value', default='unused')\n"
        "        parser.add_argument('--boolean-result', action='store_true')\n"
        "    def handler(args):\n"
        "        return True if args.boolean_result else 7\n"
        "    ctx.register_cli_command(\n"
        "        name='exit-fixture',\n"
        "        help='exit-code fixture',\n"
        "        setup_fn=setup,\n"
        "        handler_fn=handler,\n"
        "    )\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - exit-code-fixture\n",
        encoding="utf-8",
    )


def test_python_module_propagates_plugin_cli_handler_exit_code(tmp_path: Path):
    _write_exit_code_plugin(tmp_path)
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "exit-fixture"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 7, result.stderr or result.stdout


def test_plugin_cli_boolean_result_preserves_success_exit(tmp_path: Path):
    _write_exit_code_plugin(tmp_path)
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "exit-fixture", "--boolean-result"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout
