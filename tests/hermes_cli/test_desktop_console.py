"""Real child output must drain even when no interactive console is consuming it."""

import logging
import subprocess
import sys

import pytest

from hermes_cli.desktop_console import desktop_console_output, desktop_launch_notice


@pytest.mark.windows_only
def test_packaged_console_output_drains_both_streams(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="hermes_cli.desktop")

    def blocked_console(*args, **kwargs):
        raise AssertionError("packaged startup attempted a synchronous console write")

    monkeypatch.setattr("builtins.print", blocked_console)
    desktop_launch_notice("Starting Hermes")
    with desktop_console_output(source_mode=False) as streams:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.stdout.write('x'*131072); "
             "sys.stdout.flush(); sys.stderr.write('diagnostic\\n'); sys.exit(7)"],
            timeout=15, check=False, **streams,
        )
    assert result.returncode == 7
    messages = [record.getMessage() for record in caplog.records]
    assert sum(message.count("x") for message in messages) == 131072
    assert any("diagnostic" in message for message in messages)


def test_source_launch_keeps_interactive_streams():
    with desktop_console_output(source_mode=True) as streams:
        assert streams == {}
