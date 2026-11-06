"""Opt-in cancellation for finite gateway clients, without changing detach defaults."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from hermes_cli import gateway_chat_startup, main
from tests.waiting import wait_until


def test_interrupt_on_exit_parses_and_routes_text_chat_to_gateway(monkeypatch):
    from hermes_cli._parser import build_top_level_parser

    args = build_top_level_parser()[0].parse_args([
        "chat", "--oneshot", "--interrupt-on-exit", "-q", "work", "--source", "tool",
    ])
    seen = []
    monkeypatch.setattr(gateway_chat_startup, "launch_gateway_chat", lambda args: seen.append(args) or 130)
    with pytest.raises(SystemExit) as exit_info:
        main.cmd_chat(args)
    assert exit_info.value.code == 130
    assert seen == [args] and args.interrupt_on_exit


_SIGNAL_CLIENT = r'''
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import signal
import sys

from hermes_cli import gateway_chat, gateway_chat_startup
from hermes_cli._parser import build_top_level_parser
from hermes_cli.gateway_client import GatewayClientError
from tests.waiting import async_wait_until

state, scenario, enabled = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'yes'
calls = []
class Peer:
    def __init__(self):
        self.events = asyncio.Queue()
        self.receipts = 0
    async def rpc(self, method, **params):
        calls.append([method, params])
        (state / 'calls.json').write_text(json.dumps(calls))
        if method == 'runtime.describe':
            return {'session_create': {'sources': ['tool'], 'parameters': ['cwd', 'source']}}
        if method == 'session.create':
            return {'stored_session_id': 'own-session', 'execution_generation': 3}
        if method == 'prompt.submit':
            if scenario == 'complete':
                self.events.put_nowait({'params': {'session_id': 'own-session',
                    'admission_id': 'own-input', 'type': 'message.complete',
                    'payload': {'outcome': 'completed', 'text': 'finished'}}})
            else:
                asyncio.get_running_loop().call_soon((state / 'ready').write_text, 'ready')
            if scenario == 'inflight':
                await async_wait_until(lambda: (state / 'release').exists(), timeout=5)
            return {'admission_id': 'own-input'}
        if method == 'prompt.receipt':
            assert params['admission_id'] == 'own-input' and params['session_id'] == 'own-session'
            self.receipts += 1
            status = 'started'
            if scenario in ('complete', 'complete-race') or (scenario == 'stale' and self.receipts > 1):
                status = 'terminal'
            elif scenario == 'unknown':
                status = 'unknown'
            elif scenario == 'queued' or (scenario == 'queue-race' and self.receipts == 1):
                status = 'queued'
            return {'admission_id': 'own-input', 'status': status, 'execution_generation': 7,
                    'result': {'completed': True, 'final_response': 'finished'}}
        if method == 'prompt.cancel':
            if scenario == 'queue-race':
                raise GatewayClientError('stale_generation')
            assert scenario == 'queued' and params == {'session_id': 'own-session', 'admission_id': 'own-input'}
            return {'status': 'terminal', 'outcome': 'cancelled'}
        if method == 'session.interrupt':
            if scenario == 'finish-on-stop':
                self.events.put_nowait({'params': {'session_id': 'own-session',
                    'admission_id': 'own-input', 'type': 'message.complete',
                    'payload': {'outcome': 'completed', 'text': 'finished'}}})
                loop = asyncio.get_running_loop()
                yielded = loop.create_future()
                loop.call_soon(loop.call_soon, yielded.set_result, None)
                await yielded
            if scenario == 'refused':
                raise GatewayClientError('permission_denied')
            if scenario in ('stale', 'stale-successor'):
                raise GatewayClientError('stale_generation')
            if scenario == 'timeout':
                await asyncio.Event().wait()
            assert params == {'session_id': 'own-session', 'execution_generation': 7}, params
            return {'execution_generation': 7}
        raise AssertionError(method)

@asynccontextmanager
async def connected():
    yield Peer()

gateway_chat.connect_gateway = connected
gateway_chat_startup.ensure_launch_provider = lambda args: True
argv = ['chat', '--oneshot', '-q', 'work', '--source', 'tool']
if scenario != 'text':
    argv += ['--format', 'stream-json']
if enabled:
    argv.append('--interrupt-on-exit')
args = build_top_level_parser()[0].parse_args(argv)
previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
code = gateway_chat.launch_from_args(args)
assert previous == {sig: signal.getsignal(sig) for sig in previous}, 'signal handlers leaked'
raise SystemExit(code)
'''


def _signal_client(tmp_path, scenario="running", signal_name: str | None = "SIGTERM", enabled=True):
    root = Path(__file__).resolve().parents[2]
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TZ", "TMPDIR") if key in os.environ}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(tmp_path / "home"),
               PYTHONPATH=str(root), PYTHONUNBUFFERED="1")
    child = subprocess.Popen([sys.executable, "-c", _SIGNAL_CLIENT, str(tmp_path), scenario, "yes" if enabled else "no"],
                             cwd=root, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if signal_name:
            assert wait_until(lambda: (tmp_path / "ready").exists() or child.poll() is not None, timeout=15)
            assert child.poll() is None, child.communicate()
            child.send_signal(getattr(signal, signal_name))
            if scenario == "inflight":
                (tmp_path / "release").write_text("release")
        out, err = child.communicate(timeout=10)
        calls = json.loads((tmp_path / "calls.json").read_text())
        return child.returncode, out, err, calls
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("signal_name,expected", [("SIGINT", 130), ("SIGTERM", 143)])
def test_signal_interrupts_only_own_generation_and_closes_jsonl(tmp_path, signal_name, expected):
    code, out, err, calls = _signal_client(tmp_path, signal_name=signal_name)
    assert code == expected, (out, err)
    assert [name for name, _ in calls][-2:] == ["prompt.receipt", "session.interrupt"]
    events = [json.loads(line) for line in out.splitlines()]
    assert events[-1]["type"] == "result" and events[-1]["exit_code"] == expected
    assert events[-1]["session_id"] == "own-session"
    assert "accepted work continues" not in err


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("scenario", ["queued", "queue-race", "inflight", "complete-race", "stale"])
def test_signal_cancels_owned_input_across_admission_races(tmp_path, scenario):
    code, out, err, calls = _signal_client(tmp_path, scenario)
    assert code == 143, (out, err, calls)
    interrupts = [params for method, params in calls if method == "session.interrupt"]
    if scenario in ("queued", "complete-race"):
        assert not interrupts
    else:
        assert interrupts == [{"session_id": "own-session", "execution_generation": 7}]
    if scenario in ("queued", "queue-race"):
        assert any(method == "prompt.cancel" for method, _ in calls)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("scenario", ["refused", "timeout", "unknown", "stale-successor"])
def test_unconfirmed_cancel_fails_without_claiming_stop(tmp_path, scenario):
    code, out, err, calls = _signal_client(tmp_path, scenario)
    assert code == 1, (out, err)
    result = json.loads(out.splitlines()[-1])
    assert result["exit_code"] == 1 and result["error"]
    assert "could not be confirmed" in err and "work may continue" in err
    assert len([method for method, _ in calls if method == "session.interrupt"]) <= 1


@pytest.mark.platforms("posix")
def test_completion_during_stop_emits_only_one_terminal_result(tmp_path):
    code, out, err, _ = _signal_client(tmp_path, "finish-on-stop")
    results = [event for line in out.splitlines() if (event := json.loads(line))["type"] == "result"]
    assert len(results) == 1 and results[0]["exit_code"] == code == 143, (out, err)


@pytest.mark.platforms("posix")
def test_default_sigint_still_detaches_without_interrupting(tmp_path):
    code, out, err, calls = _signal_client(tmp_path, signal_name="SIGINT", enabled=False)
    assert code == 130 and "accepted work continues" in err
    assert not any(method in ("prompt.cancel", "session.interrupt") for method, _ in calls)
    assert json.loads(out.splitlines()[-1])["exit_code"] == 130


@pytest.mark.platforms("posix")
def test_opt_in_normal_completion_does_not_cancel(tmp_path):
    code, out, err, calls = _signal_client(tmp_path, "complete", signal_name=None)
    assert code == 0, (out, err)
    assert not any(method in ("prompt.cancel", "session.interrupt") for method, _ in calls)
    assert json.loads(out.splitlines()[-1])["exit_code"] == 0


@pytest.mark.platforms("posix")
def test_text_mode_signal_has_no_json_output(tmp_path):
    code, out, err, _ = _signal_client(tmp_path, "text")
    assert code == 143 and not out, (out, err)
    assert "Cancellation requested" in err


@pytest.mark.parametrize("options", [{}, {"oneshot_exit": True, "tui": True}])
def test_invalid_opt_in_fails_before_gateway_connection(monkeypatch, capsys, options):
    import argparse
    from hermes_cli import gateway_chat

    monkeypatch.setattr(gateway_chat, "connect_gateway", lambda: pytest.fail("connected"))
    assert gateway_chat.launch_from_args(argparse.Namespace(interrupt_on_exit=True, **options)) == 2
    assert "requires --oneshot" in capsys.readouterr().err


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_signal_handlers_use_the_event_loop_and_restore_on_completion(monkeypatch):
    import asyncio
    from hermes_cli.gateway_chat_signals import run_interruptible

    loop = asyncio.get_running_loop()
    installed = []
    add = loop.add_signal_handler

    def tracked(sig, callback, *args):
        installed.append(sig)
        return add(sig, callback, *args)

    class View:
        async def run(self, query, *, oneshot):
            assert oneshot and query == "work"
            return 0

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(loop, "add_signal_handler", tracked)
    assert await run_interruptible(View(), "work") == 0
    assert installed == [signal.SIGINT, signal.SIGTERM]
    assert previous == {sig: signal.getsignal(sig) for sig in previous}
