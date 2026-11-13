"""``on_session_open`` fires in the gateway when a client creates or opens a local session (#89385).

A real daemon (``hermes_agent.gateway.run``) loads a user plugin whose ``on_session_open`` hook appends one JSON
line per call. Clients reach it over the canonical websocket exactly as the CLI, Ink TUI and Desktop
do (``session.create`` / ``session.resume``).
"""
import io
import json
from pathlib import Path
import sqlite3

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model
from tests.gateway.test_cli_plugin_command_in_gateway import _Wire
import tests.gateway.test_pre_command_in_gateway as harness
from tests.gateway.test_pre_command_in_gateway import _ProfileWire, _gateway
from tests.waiting import wait_until

_PLUGIN = '''import json
import os
from pathlib import Path


def register(ctx):
    def opened(**kw):
        from hermes_constants import get_hermes_home
        log = Path(%r)
        gate = log.with_suffix(".gate")
        with open(log, "a") as out:
            out.write(json.dumps(dict(kw, home=str(get_hermes_home()), phase="start")) + "\\n")
        if gate.exists():
            from tests.waiting import wait_until
            try:
                wait_until(lambda: gate.with_suffix(".open").exists(), timeout=5, what="the gate")
            except AssertionError:
                pass
        with open(log, "a") as out:
            out.write(json.dumps(dict(kw, home=str(get_hermes_home()), phase="done")) + "\\n")
    ctx.register_hook("on_session_open", opened)
'''


def _done(log):
    lines = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return [(r['session_id'], r['platform'], r['reason'], r['home']) for r in lines if r['phase'] == 'done']


def _message_count(home, sid):
    with sqlite3.connect(home / 'state.db') as db:
        row = db.execute('SELECT message_count FROM sessions WHERE id=?', (sid,)).fetchone()
    return row and row[0]


def _wait_done(log, count):
    return wait_until(lambda: _done(log) if len(_done(log)) >= count else [], timeout=20,
                      what=f'{count} on_session_open lines in {log.name}')


@pytest.mark.platforms("linux")
def test_each_open_fires_once_in_the_owning_profile(tmp_path, monkeypatch):
    """A gateway serving two profiles: create and a later reopen of the stored chat each fire once, in
    the session's own profile (A -> B -> A); a snapshot poll on an attachment already held and a reconnect fire nothing."""
    monkeypatch.setattr(harness, '_PLUGIN', _PLUGIN)
    with _gateway(tmp_path, beta=True) as (_proc, desc, home, work, _env, root_log, _peer):
        beta_home = (home / 'profiles' / 'beta').resolve()
        beta_log = tmp_path / 'beta.jsonl'
        root, beta = _Wire(home, desc), _ProfileWire(home, desc, beta_home)
        try:
            root_sid = root.call('session.create', request_id='root', source='cli', toolsets=[], cwd=str(work))['session_id']
            beta_sid = beta.call('session.create', request_id='beta', source='tui', toolsets=[], cwd=str(work))['session_id']
            root.call('session.resume', session_id=root_sid)  # a poll on the held attachment
            root.call('prompt.submit', session_id=root_sid, input_id='turn', text='hello')
            _wait_done(beta_log, 1)
            wait_until(lambda: _message_count(home, root_sid), timeout=30, what='the first turn to be stored')
            root.close()
            again = _Wire(home, desc)  # `hermes --resume`: a new client opens the stored session
            try:
                again.call('session.resume', session_id=root_sid)
            finally:
                again.close()
            # A reconnect (catch-up, then a snapshot re-attach) is no new open; the next real open is the barrier.
            for reconnect in (True, False):
                client = _Wire(home, desc)
                try:
                    if reconnect:
                        client.call('session.events.since', session_id=root_sid, last_sequence=0, replay_epoch='gone')
                    client.call('session.resume', session_id=root_sid)
                finally:
                    client.close()
            # A viewer that detached (a backgrounded tab) and then probes the ring (Desktop's clarify
            # recovery) is no reconnect: its later reopen is an open.
            tab = _Wire(home, desc)
            try:
                held = tab.call('session.resume', session_id=root_sid)
                tab.call('session.detach', session_id=root_sid, subscription_id=held['subscription_id'])
                tab.call('session.events.since', session_id=root_sid, last_seen=2**53 - 1)
                tab.call('session.resume', session_id=root_sid)
            finally:
                tab.close()
            root_fired = _wait_done(root_log, 5)
        finally:
            beta.close()
    assert root_fired == [(root_sid, 'cli', 'create', str(home.resolve()))] + [
        (root_sid, 'cli', 'resume', str(home.resolve()))] * 4, root_fired
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


async def _hung_open(monkeypatch, sid, release):
    """A real pending open whose delivery hangs until *release* (as under ``hook_callback_timeout: 0``,
    where the dispatcher never abandons a callback)."""
    import threading
    from types import SimpleNamespace
    from hermes_agent.gateway import session_open
    from hermes_agent.gateway.session_contract import SessionRef
    monkeypatch.setattr(session_open, '_UNBOUNDED_TURN_WAIT_S', 0.5, raising=False)
    entered = threading.Event()
    delivered = []
    monkeypatch.setattr(session_open._Open, '_deliver',
                        lambda self, *a: (delivered.append(a), entered.set(), release.wait(20)))
    ref = SessionRef('p', sid)
    gate = session_open.open_local(SimpleNamespace(db=None), ref)
    gate.fire(SimpleNamespace(ignore_user_config=False), 'route')
    opening = gate.delivery
    import asyncio
    assert await asyncio.to_thread(entered.wait, 10), 'the open delivery never started'
    return ref, opening, delivered


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_a_hung_open_never_wedges_the_turn_or_its_executing_stamp(monkeypatch, cancel):
    """With callbacks never abandoned (``hook_callback_timeout: 0``) a hung ``on_session_open`` delays
    the next turn by a bounded wait only, and a drain cancelled during that wait leaves no executing
    stamp behind (which would pin the session against retirement forever)."""
    import asyncio
    import threading
    from types import SimpleNamespace
    from hermes_agent.gateway import session_finite
    from hermes_agent.gateway.session_authority import LiveSession, SessionAuthority
    release = threading.Event()
    ran = []

    async def turn(authority, ref, row):
        ran.append(row['admission_id'])
        return 'ok'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', turn)
    try:
        ref, opening, _delivered = await _hung_open(monkeypatch, 'hung-open', release)
        live = LiveSession(SimpleNamespace(chat_id=ref.session_id), 'route')
        owner = SimpleNamespace(executing={ref.session_id: 7})  # as _stamp_claim left it
        task = asyncio.ensure_future(SessionAuthority._execute_claimed(owner, ref, live, {'admission_id': 'a'}))
        if cancel:
            for _ in range(5):  # yields only: the turn runs up to its wait on the open
                await asyncio.sleep(0)
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=5)
        assert done, 'the turn waited on the hung open without bound'
        assert not opening.task.done(), 'the open delivery keeps running; only the turn stopped waiting'
    finally:
        release.set()
    assert ref.session_id not in owner.executing, owner.executing
    assert task.cancelled() if cancel else (task.result() == ('ok', 'completed') and ran == ['a'])
    await asyncio.wait({opening.task}, timeout=5)


@pytest.mark.asyncio
async def test_opens_behind_a_hung_open_stay_bounded_and_fire_once_after_it(monkeypatch):
    """With callbacks never abandoned, a session reopened 40 times behind a hung ``on_session_open``
    holds one queued delivery, not 40; once the hang ends the opens meanwhile fire once, after it."""
    import asyncio
    import threading
    from types import SimpleNamespace
    from hermes_agent.gateway import session_open
    release = threading.Event()
    try:
        ref, first, delivered = await _hung_open(monkeypatch, 'reopened', release)
        before = len(asyncio.all_tasks())
        for n in range(40):
            gate = session_open.open_local(SimpleNamespace(db=None), ref)
            gate.fire(SimpleNamespace(ignore_user_config=False), f'route-{n}')
        for _ in range(20):  # yields only: each queued delivery reaches its wait on the hung one
            await asyncio.sleep(0)
        held = len(asyncio.all_tasks()) - before
    finally:
        release.set()
    queued = session_open._PENDING[(ref.profile_id, ref.session_id)]
    await asyncio.wait({first.task, queued.task}, timeout=10)
    assert held == 1, f'{held} deliveries queued behind the hung one'
    assert [route for _policy, route in delivered] == ['route', 'route-39'], delivered


@pytest.mark.asyncio
async def test_a_finished_open_is_freed_while_the_next_one_still_runs(monkeypatch):
    """A session reopened while each delivery runs keeps only live deliveries: a finished one is not
    held by its successor (it would chain every open since the queue was last idle)."""
    import asyncio
    import gc
    import threading
    import weakref
    from types import SimpleNamespace
    from hermes_agent.gateway import session_open
    release, hold = threading.Event(), threading.Event()
    try:
        ref, first, _delivered = await _hung_open(monkeypatch, 'chained', release)
        gate = session_open.open_local(SimpleNamespace(db=None), ref)
        gate.fire(SimpleNamespace(ignore_user_config=False), 'again')
        second, first_ref = gate.delivery, weakref.ref(first)
        monkeypatch.setattr(session_open._Open, '_deliver', lambda self, *a: hold.wait(20))
        release.set()
        await asyncio.wait({first.task}, timeout=5)
        del first, _delivered
        for _ in range(20):  # yields only: the second delivery leaves its wait on the first
            await asyncio.sleep(0)
        gc.collect()
        assert not second.task.done()
        assert first_ref() is None, 'a finished delivery stays reachable from the one after it'
    finally:
        release.set()
        hold.set()
    await asyncio.wait({second.task}, timeout=5)


@pytest.mark.asyncio
async def test_a_profile_with_no_consumer_starts_no_delivery_thread(monkeypatch, tmp_path):
    """A profile whose plugins are loaded with no ``on_session_open`` callback spends no thread on an
    open, and one that registers a callback does."""
    import asyncio
    from types import SimpleNamespace
    from hermes_agent.gateway import session_commands, session_open
    from hermes_agent.gateway.session_contract import SessionRef
    from hermes_agent.hermes_cli import plugins
    from hermes_agent.hermes_constants import reset_hermes_home_override, set_hermes_home_override
    threads = []

    async def own_thread(fn):
        threads.append(fn)
    monkeypatch.setattr(session_commands, '_on_own_thread', own_thread)
    token = set_hermes_home_override(str(tmp_path))
    try:
        manager = plugins.get_plugin_manager()
    finally:
        reset_hermes_home_override(token)
    manager._discovered = True
    for consumer in (False, True):
        if consumer:
            manager._hooks['on_session_open'] = [lambda **kw: None]
        gate = session_open.open_local(SimpleNamespace(db=None), SessionRef(str(tmp_path), f's-{consumer}'))
        gate.fire(SimpleNamespace(ignore_user_config=False), 'route')
        await asyncio.wait({gate.delivery.task}, timeout=5)
        assert len(threads) == consumer, (consumer, threads)
    # A (re)load in flight on another thread (it clears and re-registers callbacks under the discovery
    # lock, with ``_discovered`` already set) is "unknown", never "unused": the open still gets a thread.
    manager._hooks.pop('on_session_open')
    import threading
    held, done = threading.Event(), threading.Event()

    def sweep():
        with manager._discovery_lock:
            held.set()
            done.wait(5)
    t = threading.Thread(target=sweep)
    t.start()
    held.wait(5)
    try:
        gate = session_open.open_local(SimpleNamespace(db=None), SessionRef(str(tmp_path), 's-reload'))
        gate.fire(SimpleNamespace(ignore_user_config=False), 'route')
        await asyncio.wait({gate.delivery.task}, timeout=5)
        assert len(threads) == 2, threads
    finally:
        done.set()
        t.join(5)
