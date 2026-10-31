from __future__ import annotations

import secrets
import subprocess
import sys

import pytest

from hermes_cli.gateway_systemctl import print_systemctl_status


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_status_output_redacts_credentials_from_a_real_child_process(capsys, stream):
    credential = secrets.token_hex(16)
    line = f"└─4242 node /opt/mcp/server.js --email ops@example.com --api-key {credential}"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1], file=getattr(sys, sys.argv[2]))",
            line,
            stream,
        ],
        capture_output=True,
        encoding="utf-8",
        check=True,
        timeout=5,
    )
    print_systemctl_status(result)
    captured = capsys.readouterr()
    output = captured.out if stream == "stdout" else captured.err
    assert "--api-key [REDACTED]" in output
    assert credential not in captured.out + captured.err
    assert "ops@example.com" not in output
    assert "└─4242 node /opt/mcp/server.js" in output


def test_status_without_output_prints_nothing_extra(capsys):
    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    print_systemctl_status(result)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
