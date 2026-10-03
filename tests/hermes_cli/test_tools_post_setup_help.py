"""``hermes tools post-setup --help`` must list every valid key, without taxing every CLI start.

The help text used to hardcode a key list that drifted from the hook registry — runnable keys
(browser_use_cli, browserbase, lightpanda, faster_whisper) were missing, while the in-product
downgrade notice points users at exactly this command for the browser-use CLI. The key list is
rendered from the registry when the help is printed; the parser build (every CLI invocation)
must not import it.
"""

import argparse
import sys
import types

import pytest

from hermes_cli.subcommands.tools import build_tools_parser
from hermes_cli.tools_config_post_setup import valid_post_setup_keys


def _build_tools_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_tools_parser(subparsers, cmd_tools=lambda **kwargs: None)
    return parser


def _render_post_setup_help(parser: argparse.ArgumentParser, capsys) -> str:
    """Render ``hermes tools post-setup --help`` and capture its output."""
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["tools", "post-setup", "--help"])
    assert excinfo.value.code == 0
    return capsys.readouterr().out


def test_help_lists_every_valid_key(capsys):
    out = _render_post_setup_help(_build_tools_parser(), capsys)
    missing = sorted(key for key in valid_post_setup_keys() if key not in out)
    assert not missing, f"valid post-setup keys missing from --help: {missing}"


def test_help_imports_the_hook_registry_only_when_rendered(capsys, monkeypatch):
    """Parser assembly runs on every CLI invocation and must not import the hook registry;
    rendering this help is where the keys are read."""
    accessed = []
    spy = types.ModuleType("hermes_cli.tools_config_post_setup")

    def _spy_getattr(name):
        accessed.append(name)
        raise AttributeError(name)

    spy.__getattr__ = _spy_getattr
    monkeypatch.setitem(sys.modules, "hermes_cli.tools_config_post_setup", spy)

    parser = _build_tools_parser()
    assert accessed == []  # the parser build never touched the registry

    out = _render_post_setup_help(parser, capsys)
    assert "_POST_SETUP_HOOKS" in accessed  # rendered help reads the registry
    assert "browser_use_cli" in out  # and a broken registry degrades to the static list
