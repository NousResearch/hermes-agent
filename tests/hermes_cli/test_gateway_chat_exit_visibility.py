"""A supervisor killing or losing a `hermes chat -q` client gets one stderr line naming why and which session."""
import argparse
import asyncio
import os
import signal

import pytest

from hermes_cli import gateway_chat


def _args():
    return argparse.Namespace(query="hello", quiet=True)


@pytest.fixture(autouse=True)
def _no_setup(monkeypatch):
    monkeypatch.setattr("hermes_cli.gateway_chat_startup.ensure_launch_provider", lambda args: True)
    monkeypatch.setattr(gateway_chat, "_register_terminal_process", lambda: None)


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="POSIX signal delivery")
@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_signal_detaches_with_reason_session_and_nonzero_exit(monkeypatch, capsys, signame):
    signum = getattr(signal, signame)

    async def fake_run(args, emitter=None):
        gateway_chat._BOUND_SESSION["id"] = "20260101_000000_abcd"
        os.kill(os.getpid(), signum)
        await asyncio.sleep(5)  # the handler raises before this returns
        return 0

    monkeypatch.setattr(gateway_chat, "run_gateway_chat", fake_run)
    before = signal.getsignal(signum)
    assert gateway_chat.launch_from_args(_args()) == 128 + signum
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1
    assert err[0].startswith(f"Detached by {signame} (session 20260101_000000_abcd)")
    assert "hermes sessions interrupt 20260101_000000_abcd" in err[0]
    assert signal.getsignal(signum) == before  # the previous handler is restored


def test_ctrl_c_keeps_exit_130_and_names_the_session(monkeypatch, capsys):
    async def fake_run(args, emitter=None):
        gateway_chat._BOUND_SESSION["id"] = "s-1"
        raise KeyboardInterrupt

    monkeypatch.setattr(gateway_chat, "run_gateway_chat", fake_run)
    assert gateway_chat.launch_from_args(_args()) == 130
    assert "Detached by SIGINT (session s-1)" in capsys.readouterr().err


def test_connection_drop_names_the_session_and_exits_1(monkeypatch, capsys):
    async def fake_run(args, emitter=None):
        gateway_chat._BOUND_SESSION["id"] = "s-2"
        raise ConnectionResetError("peer went away")

    monkeypatch.setattr(gateway_chat, "run_gateway_chat", fake_run)
    assert gateway_chat.launch_from_args(_args()) == 1
    err = capsys.readouterr().err
    assert "Gateway connection/read failed; no local fallback (session s-2)" in err
    assert "peer went away" not in err  # transport text may carry credentials


def test_failure_before_a_session_is_bound_has_no_stale_session(monkeypatch, capsys):
    gateway_chat._BOUND_SESSION["id"] = "stale-from-a-previous-launch"

    async def fake_run(args, emitter=None):
        raise ConnectionRefusedError()

    monkeypatch.setattr(gateway_chat, "run_gateway_chat", fake_run)
    assert gateway_chat.launch_from_args(_args()) == 1
    assert "session" not in capsys.readouterr().err
