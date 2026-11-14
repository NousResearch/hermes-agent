"""``hermes_cli.port_owners``: name the process behind an EADDRINUSE (port of cline/cline#14532)."""

import os
import socket

from hermes_cli import port_owners


def test_describe_names_the_listening_pid_and_clears_after_close():
    """The holder of a real LISTEN socket is reported as ``PID <n>``; a freed port reports nothing,
    so callers can append the fragment to a conflict message unconditionally."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        text = port_owners.describe_port_owners(port)
        assert f"held by PID {os.getpid()}" in text, text
        assert text.startswith(" (") and text.endswith(")")
    finally:
        holder.close()
    assert port_owners.describe_port_owners(port) == ""


def test_scan_without_tools_is_silent(monkeypatch):
    """No psutil attribution and no lsof/ss/netstat on PATH: the diagnostic degrades to an empty
    result instead of turning a port-conflict report into a second failure."""
    monkeypatch.setattr(port_owners, "_psutil_listening_pids", lambda port: [])

    def _missing(cmd, timeout, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(port_owners, "_run", _missing)
    assert port_owners.listening_pids(4242) == []
    assert port_owners.describe_port_owners(4242) == ""


def test_command_redacts_home_and_caps_length(monkeypatch):
    """The command line reaches logs and status payloads: the home prefix is redacted and the
    text is capped so a long argv cannot flood a one-line conflict message."""
    import psutil

    home = "/home/alice"
    monkeypatch.setattr(port_owners.Path, "home", lambda: port_owners.Path(home))

    class _Proc:
        def __init__(self, pid):
            pass

        def cmdline(self):
            return [f"{home}/.venv/bin/python", "-m", "hermes_cli.main", "serve", "--port", "9191", "x" * 400]

        def name(self):
            return "python"

    monkeypatch.setattr(psutil, "Process", _Proc)
    text = port_owners._describe_process(123)
    assert text.startswith("~/.venv/bin/python -m hermes_cli.main serve --port 9191")
    assert home not in text
    assert len(text) <= port_owners._MAX_COMMAND_CHARS
