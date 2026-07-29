"""Tests for acp_adapter.entry startup wiring."""

import sys

import acp
import pytest

from acp_adapter import entry


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool.discover_mcp_tools",
        lambda: discovery_calls.append(True),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert discovery_calls == []










def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]










def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1


def test_shield_stdin_redirects_fd0_to_devnull(tmp_path):
    """Children must inherit NUL, not the ACP JSON-RPC pipe (#73693).

    Runs in a subprocess so the fd surgery cannot disturb the test session's
    own stdin. The child reports what a grandchild would inherit on fd 0,
    plus whether the transport can still read the original stream.
    """
    import subprocess
    import sys as _sys

    payload = tmp_path / "probe.py"
    payload.write_text(
        "import os, sys\n"
        "from acp_adapter import entry\n"
        "entry._shield_stdin_from_children()\n"
        # What a child would inherit on fd 0:
        "inherited = os.read(0, 16)\n"
        # What the ACP transport still sees on the re-homed sys.stdin:
        "transport = sys.stdin.buffer.readline()\n"
        "print('INHERITED:' + repr(inherited))\n"
        "print('TRANSPORT:' + repr(transport))\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [_sys.executable, str(payload)],
        input=b"protocol-line\n",
        capture_output=True,
        timeout=60,
    )

    out = result.stdout.decode("utf-8", "replace")
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    # fd 0 now reads EOF (NUL), so nothing a child spawns can consume — or
    # block on — the protocol stream.
    assert "INHERITED:b''" in out
    # The transport keeps the real stdin through the private duplicate.
    assert r"TRANSPORT:b'protocol-line\n'" in out
