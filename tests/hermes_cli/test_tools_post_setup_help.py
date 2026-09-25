"""``hermes tools post-setup --help`` must list every valid key.

The help text used to hardcode a key list that drifted from the hook registry — runnable keys
(browser_use_cli, browserbase, lightpanda, faster_whisper) were missing, while the in-product
downgrade notice points users at exactly this command for the browser-use CLI.
"""

import argparse
import sys

import pytest

from hermes_cli.subcommands.tools import build_tools_parser
from hermes_cli.tools_config_post_setup import valid_post_setup_keys


def _post_setup_help(capsys) -> str:
    """Build the real parser tree and capture ``tools post-setup --help`` output."""
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_tools_parser(subparsers, cmd_tools=lambda **kwargs: None)
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["tools", "post-setup", "--help"])
    assert excinfo.value.code == 0
    return capsys.readouterr().out


def test_help_lists_every_valid_key(capsys):
    out = _post_setup_help(capsys)
    missing = sorted(key for key in valid_post_setup_keys() if key not in out)
    assert not missing, f"valid post-setup keys missing from --help: {missing}"


def test_help_degrades_to_static_list_when_registry_unavailable(capsys, monkeypatch):
    """A broken hook registry must never break the CLI's argparse tree build."""

    class _Boom:
        @staticmethod
        def valid_post_setup_keys():
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "hermes_cli.tools_config_post_setup", _Boom)
    out = _post_setup_help(capsys)
    assert "browser_use_cli" in out  # static fallback still names the real keys
