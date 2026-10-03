"""Three real gateways in majority mode: the host is killed and a standby takes the group over by itself.

H hosts a Group Chat of its two Bots. S and T are custodian-only backups: each consented to continue
it, the owner designated them in that order, and each says it is always on (``group_chat.always_on``),
so the voters are [H, S, T] and the group moves by itself. Every exchange between
the three is real HTTP between separate processes, each with its own ``HERMES_HOME``.

Once the configuration has settled and H holds its lease, a send comes back ``protected`` and H's Bot
takes the turn. H is then killed (SIGKILL): after the turn settled and every voter holds the whole log,
or mid-turn. A global observer polls every live computer's public status twice a second for the
whole run: at no observed instant may two computers admit work, nor at two different epochs. A
standby must take over within the bound (the ~20 s lease, 10 s grace, then promises and catch-up)
with a certified, automatic move; its history holds the protected message and the turn's admission,
and the other standby follows it. H then restarts: it refuses work, rejoins as a copy of the new
host, and never serves. H's turn runs once, never again there or anywhere else.

Real timing: about 80 s of wall clock per case, like the other daemon tests' grant renewal waits.
"""
import asyncio
from contextlib import ExitStack, suppress
import json
from pathlib import Path
import socket
import time

import pytest
from websockets.exceptions import ConnectionClosed

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_session_group_peer_daemons import _gateway, _last_user_text, _model

ROOM = 'linked'
MEMBERS = [{'member_id': 'host', 'profile': 'default', 'handle': 'host'},
           {'member_id': 'helper', 'profile': 'helper', 'handle': 'helper'}]
# H's model holds the turn for a message carrying this marker: H dies with that turn running.
HOLD = 'HOLD_TURN'
POLL_SECONDS = .5
# Lease (20 s) and grace (10 s) after the standby last heard from the host, its rank's delay and
# jitter, then the promises, catch-up and the transition; generous for a shared machine.
TAKEOVER_BOUND_SECONDS = 90


def _free_ports(count):
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _always_on_gateway(tmp_path, name, model, root, port):
    """A default-profile gateway with its own API server, named for the status, that says it is always on,
    and a second local profile for the room's second Bot."""
    home, env = _gateway(tmp_path, name, model, root, api_port=port)
    config = json.loads((home / 'config.yaml').read_text())
    config['gateway']['display_name'] = name.upper()
    config['group_chat'] = {'always_on': True}
    (home / 'profiles' / 'helper').mkdir(parents=True)
    config['hosted_rooms'] = {'profiles': {'helper': str(home / 'profiles' / 'helper')}}
    (home / 'config.yaml').write_text(json.dumps(config))
    return home, env


class _Computer:
    """One gateway process: its home, and one websocket shared by the scenario and the observer."""

    def __init__(self, name, home, env):
        self.name, self.home, self.env = name, home, env
        self.proc = self.ws = None
        self.live = False
        self.lock = asyncio.Lock()
        self.stack = ExitStack()

    async def start(self, root):
        self.proc, desc = await asyncio.to_thread(
            self.stack.enter_context, daemon(root, self.home, self.env, barrier=False))
        self.ws = await websocket(self.home, desc)
        self.live = True

    async def call(self, method, **params):
        async with self.lock:
            return await rpc(self.ws, method, **params)

    async def result(self, method, **params):
        reply = await self.call(method, **params)
        assert 'result' in reply, (self.name, method, reply)
        return reply['result']

    async def kill(self):
        self.live = False
        self.proc.kill()
        await asyncio.to_thread(self.proc.wait, 10)
        with suppress(Exception):
            await asyncio.wait_for(self.ws.close(), 5)
        await asyncio.to_thread(self.stack.close)

    def log_tail(self, chars=6000):
        return {path.name: path.read_text(errors='replace')[-chars:]
                for path in sorted(self.home.glob('*.log'))}


class _Observer:
    """Every live computer's public view of the room, twice a second, through the whole run.

    A computer admits work while it hosts the room and its status is ``ok`` (not paused, moving or
    conflicted); its epoch is the room's ``authority_epoch`` as it reports it. ``changes`` keeps each
    computer's (role, state, epoch, host) whenever it changes; ``rounds`` every round's admitting set.
    """

    def __init__(self, computers, started):
        self.computers, self.started = computers, started
        self.changes, self.rounds, self.violations = [], [], []
        self._last = {}
        self.ids = {}

    async def admitting(self):
        """What the latest round saw admitting work: ``{(name, epoch)}``."""
        return self.rounds[-1][1] if self.rounds else set()

    async def _view(self, computer):
        try:
            status = await computer.call('groups.succession.status', room_id=ROOM)
            state = await computer.call('groups.state', room_id=ROOM)
        except (ConnectionClosed, OSError, TimeoutError):
            return None  # killed, or not up yet
        if 'result' not in status or 'result' not in state:
            return ('error', (status.get('error') or state.get('error') or {}).get('message'), None, None)
        room = state['result']['room']
        host = self.ids.get(room['authority_gateway_id'], room['authority_gateway_id'])
        return (status['result']['this_install']['role'], status['result']['state'], room['authority_epoch'], host)

    async def sample(self):
        admitting = set()
        for name, computer in self.computers.items():
            if not computer.live:
                continue
            view = await self._view(computer)
            at = round(time.monotonic() - self.started, 1)
            if view is None or not computer.live:
                continue
            if self._last.get(name) != view:
                self._last[name] = view
                self.changes.append((at, name, *view))
            role, state, epoch, _ = view
            if role == 'host' and state == 'ok':
                admitting.add((name, epoch))
        at = round(time.monotonic() - self.started, 1)
        self.rounds.append((at, admitting))
        if len(admitting) > 1 or len({epoch for _, epoch in admitting}) > 1:
            self.violations.append((at, sorted(admitting)))

    async def run(self, stop):
        while not stop.is_set():
            await self.sample()
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), POLL_SECONDS)


async def _until(read, accept, timeout, what):
    value = None
    try:
        async with asyncio.timeout(timeout):
            while not accept(value := await read()):
                await asyncio.sleep(POLL_SECONDS)
    except TimeoutError as exc:
        raise AssertionError(f'{what}: not within {timeout} s; last {value!r}') from exc
    return value


async def _events(computer):
    """The room's whole log as this computer holds it, page by page."""
    found, since = [], 0
    while True:
        page = await computer.result('groups.log', room_id=ROOM, since_seq=since, limit=100)
        found += page['events']
        if not page['events'] or not page.get('has_more'):
            return found
        since = page['events'][-1]['seq']


@pytest.mark.parametrize('turn', ['settled', 'running'])
def test_a_killed_host_is_replaced_by_a_majority_within_the_bound_and_never_by_two(tmp_path, turn):
    root = Path(__file__).resolve().parents[2]
    message = f'@host {HOLD if turn == "running" else "SETTLED"} protected before the kill'
    models = {'h': _model('H_REPLY', HOLD), 's': _model('S_REPLY'), 't': _model('T_REPLY')}
    computers = {name: _Computer(name, *_always_on_gateway(tmp_path, name, models[name], root, port))
                 for name, port in zip('hst', _free_ports(3))}
    h, s, t = computers.values()
    observer = _Observer(computers, time.monotonic())
    report = {'turn': turn}

    async def scenario():
        for computer in computers.values():
            await computer.start(root)
        stop = asyncio.Event()
        watching = asyncio.create_task(observer.run(stop))
        try:
            await setup()
            await kill_and_take_over()
            await restart_old_host()
        finally:
            stop.set()
            await watching

    async def setup():
        room = (await h.result('groups.create', room_id=ROOM, name='Linked', members=MEMBERS))['room']
        ids = {'h': room['authority_gateway_id']}
        for backup in (s, t):  # the owner's order of standbys: S first
            link = await _until(lambda: backup.result('groups.capabilities'), lambda c: c['room_link']['enabled'],
                                30, f'{backup.name} room link')
            invited = await backup.result(
                'groups.peer.invite', room_id=ROOM, home_install_id=ids['h'], authority_gateway_id=ids['h'],
                authority_epoch=room['authority_epoch'], custody_only=True, successor=True)
            added = await h.result('groups.custody.add', room_id=ROOM, target_url=link['room_link']['endpoint']['url'],
                                   catalog=invited['catalog'], grant=invited['grant'], successor=True)
            ids[backup.name] = added['install_id']
        observer.ids.update({install_id: name for name, install_id in ids.items()})
        report['ids'] = ids
        voters = [ids[name] for name in 'hst']

        # One voter joins per acknowledged round; the host serves once a majority grants its lease.
        await _until(lambda: h.result('groups.custody.status', room_id=ROOM),
                     lambda c: c['mode'] == 'majority' and c['voters'] == voters and len(c['voter_sets']) == 1,
                     90, 'the voters to settle as [H, S, T]')
        for computer in computers.values():
            current = await _until(lambda: computer.result('groups.succession.status', room_id=ROOM),
                                   lambda c: c['automatic']['mode'] == 'majority' and c['state'] == 'ok',
                                   60, f'{computer.name} to see majority mode')
            assert [voter['install_id'] for voter in current['automatic']['voters']] == voters, current
        hosting = await h.result('groups.succession.status', room_id=ROOM)
        assert hosting['this_install']['role'] == 'host' and hosting['automatic']['state'] == 'ready', hosting
        assert hosting['automatic']['standby']['install_id'] == ids['s'], hosting
        await _until(observer.admitting, lambda a: a == {('h', 1)}, 10, 'the observer to see H admit at epoch 1')

    async def kill_and_take_over():
        sent = await h.result('groups.send', room_id=ROOM, event_id='before-kill',
                              payload={'text': message, 'thread_id': 'thread'})
        assert sent['accepted'] and sent.get('protected') is True, sent
        report['message_event_id'] = sent['event']['event_id']
        if turn == 'running':
            # H starts its Bot's turn only once a majority stored the turn's admission; it dies mid-turn.
            assert await asyncio.to_thread(models['h'].gates[HOLD][0].wait, 30), 'H never started the turn'
        else:
            # H's Bot answered, and every voter holds the log through the answer: nothing is left to reconcile.
            async def held():
                history = await _events(h)  # first: the voters' watermarks read after it must cover it all
                custody = await h.result('groups.custody.status', room_id=ROOM)
                return history, [(c['watermark'] or {}).get('seq') or 0 for c in custody['custodians']]
            await _until(held, lambda v: any(event['kind'] == 'turn.settled' for event in v[0])
                         and len(v[1]) == 2 and min(v[1]) >= v[0][-1]['seq'], 60,
                         "H's turn to settle and reach every voter")
        await h.kill()
        killed = time.monotonic()

        def taken_over(admitting):
            return any(name in {'s', 't'} for name, _ in admitting)
        admitting = await _until(observer.admitting, taken_over, TAKEOVER_BOUND_SECONDS,
                                 'a standby to admit work as the new host')
        took = time.monotonic() - killed
        (name, epoch), = admitting
        report.update(new_host=name, epoch=epoch, took_s=round(took, 1))
        assert epoch > 1, admitting
        new, other = computers[name], computers['t' if name == 's' else 's']

        history = await _events(new)
        assert any(event['kind'] == 'message.user' and event['event_id'] == report['message_event_id']
                   and event['payload'].get('text') == message for event in history), history
        # The successor knows the turn H started: its admission was stored on a majority first.
        assert any(event['kind'] == 'task.admitted' for event in history), history
        moved = [event for event in history if event['kind'] == 'authority.transition']
        assert moved and moved[-1]['payload']['to_epoch'] == epoch, moved
        assert moved[-1]['payload']['proof_kind'] == 'certified', moved[-1]
        assert moved[-1]['payload']['reason'] == 'automatic', moved[-1]
        current = await new.result('groups.succession.status', room_id=ROOM)
        assert current['moved_in']['proof_kind'] == 'certified', current
        assert current['moved_in']['from']['install_id'] == report['ids']['h'], current

        # The other standby follows the new host (which pushes to it about every 5 s) and keeps the message.
        await _until(lambda: other.result('groups.state', room_id=ROOM),
                     lambda c: (c['room']['authority_gateway_id'], c['room']['authority_epoch'])
                     == (report['ids'][name], epoch), 30, f'{other.name} to follow the new host')
        assert any(event['event_id'] == report['message_event_id'] for event in await _events(other))

    async def restart_old_host():
        h.home.joinpath('restart.log').rename(h.home / 'killed.log')
        await h.start(root)
        # Back from the dead, still recorded as the host at epoch 1: it refuses work at once.
        refused = await h.call('groups.send', room_id=ROOM, event_id='after-restart',
                               payload={'text': 'MUST_NOT_BE_ADMITTED', 'thread_id': 'thread'})
        assert 'error' in refused, refused
        current = await _until(lambda: h.result('groups.succession.status', room_id=ROOM),
                               lambda c: c['this_install']['role'] == 'backup' and c['state'] == 'moved_away',
                               90, 'H to rejoin as a copy')
        assert current['moved']['to']['install_id'] == report['ids'][report['new_host']], current
        # Its copy then catches up from the new host, on its next upkeep (every 15 s).
        await _until(lambda: h.result('groups.state', room_id=ROOM),
                     lambda c: c['room'].get('copy') is True and (c['room']['authority_gateway_id'],
                     c['room']['authority_epoch']) == (report['ids'][report['new_host']], report['epoch']),
                     60, "H's copy to follow the new host")
        assert 'MUST_NOT_BE_ADMITTED' not in json.dumps(await _events(computers[report['new_host']]))
        # The new host keeps serving while the old one rejoins.
        await asyncio.sleep(3 * POLL_SECONDS)
        assert await observer.admitting() == {(report['new_host'], report['epoch'])}, observer.rounds[-3:]

    try:
        asyncio.run(scenario())
        assert not observer.violations, observer.violations
        # H's turn ran once, on H, and never again: neither when H came back nor on another computer.
        assert [message in _last_user_text(r) for r in models['h'].requests] == [True], models['h'].requests
        assert models['s'].requests == models['t'].requests == [], (models['s'].requests, models['t'].requests)
        print('takeover', json.dumps({k: report[k] for k in ('turn', 'new_host', 'epoch', 'took_s')}))
    except BaseException as exc:
        details = {'report': report, 'violations': observer.violations, 'changes': observer.changes,
                   'logs': {name: computer.log_tail() for name, computer in computers.items()}}
        raise AssertionError(json.dumps(details, indent=1, default=str)) from exc
    finally:
        (tmp_path / 'takeover.json').write_text(json.dumps({**report, 'changes': observer.changes}, default=str))
        for model in models.values():
            for _, release in model.gates.values():
                release.set()
            model.shutdown()
            model.server_close()
        for computer in computers.values():
            computer.stack.close()
