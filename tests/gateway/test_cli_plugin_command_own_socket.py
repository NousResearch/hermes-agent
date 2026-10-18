"""A classic CLI plugin command never holds the CLI's own gateway connection (#134889 PLG-7).

The gateway answers one socket's requests one at a time, in arrival order (``tui_gateway/ws.py``
``_dispatch_loop``), and a plugin handler may run for minutes. So ``slash.exec`` for a plugin command
goes over a socket of its own to the same gateway, for the same session: while the handler runs, even
after Ctrl+C stopped the wait, the CLI's reads, prompt admissions and interrupts are answered at once.
Main's CLI ran the handler itself, without occupying any gateway connection.
"""
import os
import signal
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, daemon
from tests.gateway.test_cli_plugin_command_in_gateway import (
    _SLOW, _dial_running, _no_local_plugins, _plugin_home, _release, _typed_in_thread,
)
from tests.waiting import wait_until

# The command socket's thread (hermes_cli/gateway_connection.py); none may outlive its command.
_COMMAND_THREAD = 'hermes-gateway-command'


def _command_threads():
    return [thread for thread in threading.enumerate() if thread.name == _COMMAND_THREAD]


def _cli(monkeypatch, home):
    """``(cli, connection)``: a classic CLI on its own production connection to ``home``."""
    import cli as cli_mod
    from cli import HermesCLI
    from hermes_cli.gateway_connection import GatewayConnection
    from hermes_cli.gateway_session_store import GatewaySessionStore
    _no_local_plugins(monkeypatch)
    monkeypatch.setattr(cli_mod, '_ensure_skill_commands', dict)
    monkeypatch.setattr(cli_mod, 'get_skill_bundles', dict)
    monkeypatch.setenv('HERMES_HOME', str(home))
    conn = GatewayConnection(connect=_dial_running(home))
    c = HermesCLI()
    c.agent, c._session_db, c._gateway_session_open = None, GatewaySessionStore(conn), False
    return c, conn


def _slow_source(tmp_path):
    gate, marks = tmp_path / 'gate', tmp_path / 'marks'
    os.mkfifo(gate)
    marks.mkdir()
    return gate, marks, _SLOW.replace('GATE', repr(str(gate))).replace('MARKS', repr(str(marks)))


def _ctrl_c(c):
    """The classic TUI's Ctrl+C binding, with an empty draft."""
    from prompt_toolkit.buffer import Buffer
    app = SimpleNamespace(current_buffer=Buffer(), invalidate=lambda: None,
                          exit=lambda: setattr(c, '_should_exit', True))
    c._tui_handle_ctrl_c(SimpleNamespace(app=app))


@pytest.mark.platforms("linux")
def test_a_stopped_plugin_command_leaves_the_cli_connection_free(tmp_path, monkeypatch, capsys):
    """Ctrl+C stops the wait while the handler stays blocked in the gateway. Before it is released, the
    CLI's connection answers a prompt admission, a receipt read and the interrupt of that turn, in
    order. Released, the handler's late reply is never shown and no command socket or pending
    request is left."""
    from hermes_cli import gateway_plugin_commands
    from hermes_cli.gateway_client import GatewayClientError
    models, model_post = [], Model.do_POST

    def seen_post(handler):  # the loopback model server, to hold and release the admitted turn
        models.append(handler.server)
        model_post(handler)
    monkeypatch.setattr(Model, 'do_POST', seen_post)
    gate, marks, source = _slow_source(tmp_path)
    outcomes = {}

    def step(name, method, **params):  # the primary connection's ordinary call, its default budget
        try:
            outcomes[name] = conn.call(method, **params)
        except GatewayClientError as exc:
            outcomes[name] = exc
        return outcomes[name]
    with _plugin_home(tmp_path, source) as (root, home, env):
        c, conn = _cli(monkeypatch, home)
        try:
            with daemon(root, home, env, barrier=True, fixture='plugin_mcp_daemon.py') as (proc, _desc):
                c._ensure_gateway_session()
                sid = c.session_id
                assert 'ppslow' in (gateway_plugin_commands.names(c) or {})  # cached: no catalog read below
                capsys.readouterr()
                stopped = _typed_in_thread(c, '/ppslow held')
                wait_until(lambda: (marks / 'started-held').exists(), timeout=30, what='the handler to start')
                _ctrl_c(c)
                stopped.join(60)
                stop_out, stop_alive = capsys.readouterr().out, stopped.is_alive()
                try:
                    receipt = step('submit', 'prompt.submit', session_id=sid, submission_id='held-turn',
                                   text='BLOCK_STARTED')
                    if isinstance(receipt, dict):
                        wait_until(lambda: models and models[-1].blocked.is_set(), timeout=60,
                                   what='the admitted turn to reach the model')
                        read = step('read', 'prompt.receipt', session_id=sid, admission_id=receipt['admission_id'])
                        if isinstance(read, dict):
                            step('interrupt', 'session.interrupt', session_id=sid,
                                 execution_generation=read['execution_generation'])
                    handler_held = not (marks / 'done-held').exists()
                finally:
                    for model in models:
                        model.release.set()
                    _release(gate, 'late')
                wait_until(lambda: (marks / 'done-held').exists(), timeout=30, what='the handler to finish')
                after = c._process_unregistered_slash('/ppquick', '/ppquick'), capsys.readouterr().out
                wait_until(lambda: not _command_threads(), timeout=30, what='the command sockets to close')
                pending = dict(conn.client.pending) if conn.client is not None else {}
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=20)
        finally:
            conn.close()
    assert not stop_alive and not c._should_exit, (stop_alive, c._should_exit)
    assert 'stopped waiting' in stop_out and 'keeps running in the gateway' in stop_out, stop_out
    # answered while the stopped handler was still blocked in the gateway
    assert handler_held, 'the handler finished before the requests were answered'
    assert isinstance(outcomes.get('submit'), dict), outcomes
    assert outcomes['submit']['status'] in {'queued', 'started'}, outcomes
    assert isinstance(outcomes.get('read'), dict) and outcomes['read']['status'] == 'started', outcomes
    assert isinstance(outcomes.get('interrupt'), dict), outcomes
    assert after[0] is True and 'QUICK_DONE' in after[1] and 'SLOW_DONE' not in after[1], after
    assert pending == {}, pending


@pytest.mark.platforms("linux")
def test_a_gateway_hang_up_during_a_plugin_command_ends_the_wait_and_leaves_nothing_pending(
        tmp_path, monkeypatch, capsys):
    """The gateway dies while the handler runs: the wait ends with the transport's error (the command
    is not sent again), and no command socket, thread or pending request is left."""
    _gate, marks, source = _slow_source(tmp_path)
    with _plugin_home(tmp_path, source) as (root, home, env):
        c, conn = _cli(monkeypatch, home)
        try:
            with daemon(root, home, env, barrier=True, fixture='plugin_mcp_daemon.py') as (proc, _desc):
                c._ensure_gateway_session()
                from hermes_cli import gateway_plugin_commands
                assert 'ppslow' in (gateway_plugin_commands.names(c) or {})
                capsys.readouterr()
                waiting = _typed_in_thread(c, '/ppslow hangup')
                wait_until(lambda: (marks / 'started-hangup').exists(), timeout=30, what='the handler to start')
                proc.kill()
                proc.wait(timeout=20)
                waiting.join(60)
                out, alive = capsys.readouterr().out, waiting.is_alive()
                wait_until(lambda: not _command_threads(), timeout=30, what='the command socket to close')
                pending = dict(conn.client.pending) if conn.client is not None else {}
        finally:
            conn.close()
    assert not alive, 'the wait outlived the gateway'
    assert '/ppslow: Gateway disconnected' in out and 'stopped waiting' not in out, out
    assert out.count('/ppslow') == 1 and 'SLOW_DONE' not in out, out
    wait = getattr(c, '_plugin_command_wait', None)
    assert wait is None and pending == {}, (wait, pending)
