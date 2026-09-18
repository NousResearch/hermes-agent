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


@pytest.mark.windows_only
@pytest.mark.parametrize("level", [logging.WARNING, logging.ERROR])
def test_stderr_survives_logging_threshold(caplog, level):
    caplog.set_level(level, logger="hermes_cli.desktop")
    with desktop_console_output(source_mode=False) as streams:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; print('ordinary-output'); "
             "sys.stderr.write('fatal-diagnostic\\n'); sys.exit(7)"],
            timeout=15, check=False, **streams,
        )
    assert result.returncode == 7
    messages = [record.getMessage() for record in caplog.records]
    assert any("fatal-diagnostic" in message for message in messages)
    assert not any("ordinary-output" in message for message in messages)


@pytest.mark.windows_only
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_complete_records_reach_redacted_logs(tmp_path, monkeypatch, caplog, stream):
    import hermes_logging

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    hermes_logging._reset_queued_handlers()
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="hermes_cli.desktop")
    secret = "synthetic-boundary-credential"
    payload = (
        b" " * (8192 - len(b"API_KEY=")) + b"API_KEY=" + secret.encode() + b"\n"
        + b"u" * 8191 + "\u20ac\n".encode()
        + b"oversized-start API_KEY=" + secret.encode() + b"z" * 262144
        + b"oversized-end\nrecovered\ntail " + "\u20ac".encode()
    )
    payload_path = tmp_path / "child-output.bin"
    payload_path.write_bytes(payload)
    try:
        log_dir = hermes_logging.setup_logging(hermes_home=tmp_path, log_level="INFO", mode="gui")
        with desktop_console_output(source_mode=False) as streams:
            result = subprocess.run(
                [sys.executable, "-c", "import pathlib, sys; "
                 "data = pathlib.Path(sys.argv[1]).read_bytes(); "
                 "out = getattr(sys, sys.argv[2]).buffer; "
                 "[(out.write(data[i:i+997]), out.flush()) for i in range(0, len(data), 997)]",
                 str(payload_path), stream],
                timeout=15, check=False, **streams,
            )
        assert result.returncode == 0
        hermes_logging.flush_log_queue()
        text = (log_dir / "agent.log").read_text(encoding="utf-8")
        assert secret not in text
        assert "u" * 8191 + "\u20ac" in text
        assert "\ufffd" not in text
        assert "oversized-start" not in text and "oversized-end" not in text
        assert "recovered" in text and "tail \u20ac" in text
        assert "omitted" in text.lower()
        gui_text = (log_dir / "gui.log").read_text(encoding="utf-8")
        assert "recovered" in gui_text and secret not in gui_text
    finally:
        hermes_logging._reset_queued_handlers()


def test_source_launch_keeps_interactive_streams():
    with desktop_console_output(source_mode=True) as streams:
        assert streams == {}
