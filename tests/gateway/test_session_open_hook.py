"""``on_session_open`` fires in the gateway when a client creates or opens a local session (#89385).

A real daemon (``gateway.run``) loads a user plugin whose ``on_session_open`` hook appends one JSON
line per call. Clients reach it over the canonical websocket exactly as the CLI, Ink TUI and Desktop
do (``session.create`` / ``session.resume``).
"""
import io
import json
from pathlib import Path

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model
from tests.gateway.test_cli_plugin_command_in_gateway import _Wire
import tests.gateway.test_pre_command_in_gateway as harness
from tests.gateway.test_pre_command_in_gateway import _ProfileWire, _gateway
from tests.waiting import wait_until

_PLUGIN = '''import json
import os
import time
from pathlib import Path


def register(ctx):
    def opened(**kw):
        from hermes_constants import get_hermes_home
        log = Path(%r)
        gate = log.with_suffix(".gate")
        with open(log, "a") as out:
            out.write(json.dumps(dict(kw, home=str(get_hermes_home()), phase="start")) + "\\n")
        deadline = time.monotonic() + 5
        while gate.exists() and not gate.with_suffix(".open").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        with open(log, "a") as out:
            out.write(json.dumps(dict(kw, home=str(get_hermes_home()), phase="done")) + "\\n")
    ctx.register_hook("on_session_open", opened)
'''


def _done(log):
    lines = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return [(r['session_id'], r['platform'], r['reason'], r['home']) for r in lines if r['phase'] == 'done']


def _wait_done(log, count):
    return wait_until(lambda: _done(log) if len(_done(log)) >= count else [], timeout=20,
                      what=f'{count} on_session_open lines in {log.name}')


@pytest.mark.platforms("linux")
def test_each_open_fires_once_in_the_owning_profile(tmp_path, monkeypatch):
    """A gateway serving two profiles: create and a later reopen each fire once, in the session's own
    profile (A -> B -> A); a snapshot poll on an attachment already held fires nothing."""
    monkeypatch.setattr(harness, '_PLUGIN', _PLUGIN)
    with _gateway(tmp_path, beta=True) as (_proc, desc, home, work, _env, root_log, _peer):
        beta_home = (home / 'profiles' / 'beta').resolve()
        beta_log = tmp_path / 'beta.jsonl'
        root, beta = _Wire(home, desc), _ProfileWire(home, desc, beta_home)
        try:
            root_sid = root.call('session.create', request_id='root', source='cli', toolsets=[], cwd=str(work))['session_id']
            beta_sid = beta.call('session.create', request_id='beta', source='tui', toolsets=[], cwd=str(work))['session_id']
            root.call('session.resume', session_id=root_sid)  # a poll on the held attachment
            _wait_done(beta_log, 1)
            root.close()
            again = _Wire(home, desc)  # `hermes --resume`: a new client opens the stored session
            try:
                again.call('session.resume', session_id=root_sid)
                root_fired = _wait_done(root_log, 2)
            finally:
                again.close()
        finally:
            beta.close()
    assert root_fired == [(root_sid, 'cli', 'create', str(home.resolve())),
                          (root_sid, 'cli', 'resume', str(home.resolve()))], root_fired
    assert _done(beta_log) == [(beta_sid, 'tui', 'create', str(beta_home))]


class _OrderModel(Model):
    """The model endpoint, recording which open deliveries had finished when each turn reached it."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.rfile = io.BytesIO(body)
        if (log := getattr(self.server, 'open_log', None)) is not None and b'TURN_ONE' in body:
            self.server.seen_at_turn.append(_done(log))
            log.with_suffix('.open').write_text('')  # a hook still held here is released
        super().do_POST()


@pytest.mark.platforms("linux")
def test_the_reply_never_waits_on_the_hook_and_the_first_turn_does(tmp_path, monkeypatch):
    """``session.create`` answers while the hook is still running; a prompt submitted at once reaches
    the model only after the hook finished."""
    monkeypatch.setattr(harness, '_PLUGIN', _PLUGIN)
    monkeypatch.setattr(harness, 'Model', _OrderModel)
    with _gateway(tmp_path) as (_proc, desc, home, work, _env, log, peer):
        log.with_suffix('.gate').write_text('')  # the hook holds (see the plugin)
        peer.seen_at_turn, peer.open_log = [], log
        wire = _Wire(home, desc)
        try:
            sid = wire.call('session.create', request_id='slow', source='gui', toolsets=[], cwd=str(work))['session_id']
            wait_until(lambda: log.exists(), timeout=20, what='the hook to start')
            assert _done(log) == []  # the create reply came back while the hook still runs
            # The hook holds until the first turn reaches the model or 5 s pass: only a turn that
            # waits for the hook sees it finished.
            wire.call('prompt.submit', session_id=sid, input_id='first', text='TURN_ONE')
            wait_until(lambda: peer.seen_at_turn, timeout=30, what='the first turn')
        finally:
            wire.close()
    assert peer.seen_at_turn[0] == [(sid, 'desktop', 'create', str(Path(home).resolve()))], peer.seen_at_turn
