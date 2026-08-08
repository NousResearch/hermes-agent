"""Tests for acp_adapter.entry startup wiring."""

import os
import sys
from pathlib import Path

import acp
import pytest

from acp_adapter import entry

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_with_repo_first():
    """Environment for probe children with this tree first on the import path.

    Without it a child resolves ``acp_adapter`` against whatever the venv has
    installed — a worktree run against an install venv silently probes the
    wrong code and dies with ``AttributeError`` on the new symbols.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH")) if p
    )
    return env


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
        env=_env_with_repo_first(),
    )

    out = result.stdout.decode("utf-8", "replace")
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    # fd 0 now reads EOF (NUL), so nothing a child spawns can consume — or
    # block on — the protocol stream.
    assert "INHERITED:b''" in out
    # The transport keeps the real stdin through the private duplicate.
    assert r"TRANSPORT:b'protocol-line\n'" in out


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="STD_INPUT_HANDLE inheritance is Windows-specific",
)
def test_shield_stdin_denies_the_pipe_to_a_real_descendant(tmp_path):
    """A spawned descendant must inherit NUL, not the JSON-RPC pipe.

    The fd-0 probe above cannot prove this on Windows: a child launched with
    ``stdin=None`` inherits the process-wide ``STD_INPUT_HANDLE``, not fd 0,
    so only an actual descendant exercises the guarantee this shield makes.
    """
    import subprocess
    import sys as _sys

    payload = tmp_path / "descendant_probe.py"
    payload.write_text(
        "import subprocess, sys\n"
        "from acp_adapter import entry\n"
        "entry._shield_stdin_from_children()\n"
        # Launch a real grandchild with stdin=None so it inherits whatever
        # this process hands down — fd 0 on POSIX, STD_INPUT_HANDLE on Windows.
        "child = subprocess.run(\n"
        "    [sys.executable, '-c',\n"
        "     'import os,sys; sys.stdout.write(repr(os.read(0, 16)))'],\n"
        "    capture_output=True, timeout=30,\n"
        ")\n"
        "print('CHILD:' + child.stdout.decode('utf-8', 'replace'))\n"
        # The transport must still own the real stream afterwards.
        "print('TRANSPORT:' + repr(sys.stdin.buffer.readline()))\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [_sys.executable, str(payload)],
        input=b"protocol-line\n",
        capture_output=True,
        timeout=60,
        env=_env_with_repo_first(),
    )

    out = result.stdout.decode("utf-8", "replace")
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    # The descendant saw EOF: it cannot consume or block on the protocol pipe.
    assert "CHILD:b''" in out
    assert r"TRANSPORT:b'protocol-line\n'" in out


def test_shield_stdin_rolls_back_when_the_win32_handle_redirect_fails(tmp_path):
    """A failed ``SetStdHandle`` must not leave a half-shielded process.

    fd 0 is repointed at NUL *before* the Win32 std-handle redirect. If that
    redirect fails and we carry on, children still inherit the real pipe via
    ``STD_INPUT_HANDLE`` while the log claims the stdin was shielded. Restore
    the original fd 0 instead, so the process is left in a known state.
    """
    import subprocess
    import sys as _sys

    payload = tmp_path / "failing_redirect_probe.py"
    payload.write_text(
        "import os, sys\n"
        "from acp_adapter import entry\n"
        # Simulate SetStdHandle returning FALSE, on any platform.
        "entry._point_win32_stdin_at = lambda fd: False\n"
        "entry._shield_stdin_from_children()\n"
        # Rolled back: fd 0 is the original stream again, not NUL.
        "print('FD0:' + repr(os.read(0, 32)))\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [_sys.executable, str(payload)],
        input=b"protocol-line\n",
        capture_output=True,
        timeout=60,
        env=_env_with_repo_first(),
    )

    out = result.stdout.decode("utf-8", "replace")
    err = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, err
    # Rollback restored the original stdin on fd 0 rather than leaving NUL.
    assert r"FD0:b'protocol-line\n'" in out, out
    # And the failure was reported rather than logged as a success.
    assert "shielded" not in err.lower(), err
