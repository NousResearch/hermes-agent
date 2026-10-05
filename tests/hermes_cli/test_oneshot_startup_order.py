"""Regression for #86526: ``hermes -z`` must turn on its approval overrides
before startup discovers plugins and registers shell hooks.

run_oneshot() sets HERMES_YOLO_MODE and HERMES_ACCEPT_HOOKS, but every -z
dispatch path calls _prepare_agent_startup() first. By then a plugin may have
imported tools.approval, which freezes the yolo flag from the env at import,
and unseen hooks have already gone through the consent prompt.
"""

import os
from argparse import Namespace
from unittest.mock import patch

import pytest

from agent import shell_hooks
from hermes_cli import main as main_mod
from hermes_cli import plugins

_OVERRIDES = ("HERMES_YOLO_MODE", "HERMES_ACCEPT_HOOKS")


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in _OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    shell_hooks.reset_for_tests()
    yield home
    plugins._reset_plugin_managers_for_tests()
    shell_hooks.reset_for_tests()
    # Startup writes these straight to os.environ, so monkeypatch can't undo them.
    for name in _OVERRIDES:
        os.environ.pop(name, None)


def _oneshot_args():
    return Namespace(
        command=None,
        oneshot="hi",
        yolo=False,
        accept_hooks=False,
        toolsets=None,
        tui=False,
        cron_command=None,
        gateway_command=None,
        mcp_action=None,
    )


def test_plugins_load_with_yolo_already_on(hermes_home):
    # tools.approval freezes the yolo flag the first time it is imported, and a
    # plugin can be the thing that imports it. So record what a plugin sees.
    seen = hermes_home / "yolo_seen_by_plugin"
    plugin = hermes_home / "plugins" / "probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: probe\nversion: 0.0.1\n", encoding="utf-8"
    )
    (plugin / "__init__.py").write_text(
        "import os, pathlib\n"
        f"pathlib.Path({str(seen)!r}).write_text("
        "os.environ.get('HERMES_YOLO_MODE', ''), encoding='utf-8')\n"
        "def register(ctx):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - probe\n", encoding="utf-8"
    )

    main_mod._prepare_agent_startup(_oneshot_args())
    plugins.discover_plugins()  # waits for the background discovery thread

    assert seen.read_text(encoding="utf-8") == "1"


def test_unseen_hooks_are_accepted_without_a_prompt(hermes_home, tmp_path, monkeypatch):
    script = tmp_path / "hook.sh"
    script.write_text("#!/usr/bin/env bash\nprintf '{}\\n'\n", encoding="utf-8")
    script.chmod(0o755)
    (hermes_home / "config.yaml").write_text(
        f"hooks:\n  on_session_start:\n    - command: {script}\n", encoding="utf-8"
    )
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return "n"

    monkeypatch.setattr("builtins.input", fake_input)
    with patch("sys.stdin") as stdin:
        stdin.isatty.return_value = True
        main_mod._prepare_agent_startup(_oneshot_args())

    assert prompts == []
    assert shell_hooks.allowlist_entry_for("on_session_start", str(script)) is not None
