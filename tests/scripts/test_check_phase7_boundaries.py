"""The architecture checker rejects dependency forms used by runtime consumers."""
import pytest

from scripts.check_phase7_boundaries import audit, scan_source


@pytest.mark.parametrize("source,path", [
    ("from hermes_cli.commands import CommandDef", "gateway/consumer.py"),
    ("from hermes_cli import commands as c", "gateway/consumer.py"),
    ("from . import commands as c", "hermes_cli/consumer.py"),
    ("from ..commands import resolve_command", "hermes_cli/nested/consumer.py"),
    ("import hermes_cli as hc\nx = hc.commands.resolve_command", "gateway/consumer.py"),
    ("import importlib\nc = importlib.import_module('hermes_cli.commands')", "gateway/consumer.py"),
    ("from importlib import import_module as load\nc = load('.commands', package='hermes_cli')", "gateway/consumer.py"),
    ("c = __import__('hermes_cli.slash_exec')", "gateway/consumer.py"),
    ("c = _mod('hermes_cli.toolset_scope')", "gateway/consumer.py"),
    ("from hermes_cli.tools_config import _get_platform_tools", "gateway/consumer.py"),
    ("from .tools_config import _save_platform_tools", "hermes_cli/consumer.py"),
    ("from hermes_cli.tools_config import *", "gateway/consumer.py"),
    ("from hermes_cli import tools_config as tc\nf = tc._get_platform_tools", "gateway/consumer.py"),
    ("hc, tc = _tools_mod('hermes_cli.config'), _tools_mod('hermes_cli.tools_config')\ntc._apply_toolset_change({}, 'cli', [], 'enable')", "tui_gateway/consumer.py"),
    ("import importlib as i\na = i.import_module('hermes_cli.tools_config')\nb = a\nf = getattr(b, '_save_platform_tools')", "gateway/consumer.py"),
    ("import importlib\ngetattr(importlib.import_module('hermes_cli.tools_config'), 'get_platform_tools')", "gateway/consumer.py"),
    ("from hermes_cli.config import load_config", "commands/domain.py"),
    ("c = _mod('hermes_cli.config')", "commands/domain.py"),
    ("from gateway.command_presentation import gateway_help_lines", "commands/domain.py"),
    ("from ..hermes_cli.config import save_config", "tools/platform_policy.py"),
    ("import model_tools", "tools/toolset_selection.py"),
    ("class CommandDef: pass", "hermes_cli/replacement.py"),
    ("COMMAND_REGISTRY = []", "gateway/replacement.py"),
    ("class CommandContext: pass", "hermes_cli/replacement.py"),
    ("class CommandReply: pass", "hermes_cli/replacement.py"),
    ("get_platform_tools = lambda: []", "hermes_cli/tools_config.py"),
    ("def _parse_enabled_flag(x): return bool(x)", "hermes_cli/tools_config.py"),
])
def test_rejects_retired_dependencies_and_duplicate_owners(source, path):
    assert scan_source(source, path)


@pytest.mark.parametrize("source,path", [
    ("from commands import CommandDef, resolve_command", "gateway/consumer.py"),
    ("from . import execution", "commands/__init__.py"),
    ("from tools import platform_policy as p\np.get_platform_tools({})", "gateway/consumer.py"),
    ("from hermes_cli.config_toolsets import save_platform_tools", "tui_gateway/consumer.py"),
    ("from hermes_cli.tools_config import _toolset_has_keys, gui_toolset_label", "hermes_cli/web_routers/tools.py"),
    ("from hermes_cli.commands_completion import SlashCommandCompleter", "tui_gateway/consumer.py"),
    ("import logging\nlogger = logging.getLogger('hermes_cli.commands')", "gateway/command_platforms.py"),
    ("class CommandDef: pass\nCOMMAND_REGISTRY = []", "commands/__init__.py"),
    ("class CommandContext: pass\nclass CommandReply: pass", "commands/execution.py"),
    ("from plugin_runtime.api import get_plugin_commands", "commands/__init__.py"),
    ("from plugin_runtime.lifecycle import get_plugin_toolset_keys_nowait", "tools/platform_policy.py"),
    ("def _current_platform_tools(config, platform): return set()", "hermes_cli/tools_config.py"),
])
def test_accepts_canonical_owners_and_retained_presentation(source, path):
    assert not scan_source(source, path)


def test_repository_satisfies_phase7_ownership():
    problems, count = audit()
    assert count > 0
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("retired", [
    "hermes_cli/slash_exec.py", "hermes_cli/toolset_scope.py",
    "hermes_cli/commands_platforms.py",
])
def test_a_forwarding_module_cannot_restore_a_retired_path(tmp_path, retired):
    path = tmp_path / retired
    path.parent.mkdir(parents=True)
    path.write_text("from commands import *\n", encoding="utf-8")
    problems, _ = audit(tmp_path)
    assert any("must stay deleted" in problem for problem in problems)


def test_invalid_python_cannot_silently_escape_the_audit():
    assert scan_source("from hermes_cli import (", "gateway/consumer.py")
