"""A real gateway, a file-backed Telegram double and the real CLI: a private and a shared chat
list, send to, answer approvals in (once, always, deny) and stop canonical Group Chats.

Stop is the last turn in each gateway: on the base runtime a late Stop can still cancel the
member's next turn (#109338), which is not what these journeys are about.
"""
import asyncio
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket

COMMANDS = {'RUN_ALWAYS': 'rm -rf ./build-cache', 'RUN_DENY': 'chmod -R 777 ./scratch',
            'RUN_ONCE': 'rm -r ./old-logs'}
MEMBERS = [{'member_id': 'ada', 'profile': 'default', 'handle': 'ada', 'display_name': 'Ada'},
           {'member_id': 'two', 'profile': 'two', 'handle': 'two'}]


class Model(BaseHTTPRequestHandler):
    """Ada runs the terminal command a message names; every other turn passes."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        users = [str(m.get('content') or '') for m in messages if m['role'] == 'user']
        latest, after_tool = (users[-1] if users else ''), bool(messages) and messages[-1]['role'] == 'tool'
        message = {'role': 'assistant', 'content': 'PASS'}
        if after_tool:
            message['content'] = 'DONE ' + str(messages[-1].get('content'))[:80]
        elif 'You are @ada' in ' '.join(users):
            if 'BLOCK_STOP' in latest:
                self.server.blocked.set()
                self.server.release.wait(120)
            command = next((c for marker, c in COMMANDS.items() if marker in latest), None)
            if command:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'index': 0, 'id': 'call-' + command.split()[0], 'type': 'function',
                    'function': {'name': 'terminal', 'arguments': json.dumps({'command': command})}}]}
        finish = 'tool_calls' if 'tool_calls' in message else 'stop'
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
            self.end_headers()
            if body.get('stream'):
                for delta, done in ((message, None), ({}, finish)):
                    frame = {'id': 'm', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'm',
                             'choices': [{'index': 0, 'delta': delta, 'finish_reason': done}]}
                    self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
            else:
                self.wfile.write(json.dumps({'id': 'm', 'choices': [{'index': 0, 'message': message,
                                                                     'finish_reason': finish}],
                                             'usage': {'prompt_tokens': 1, 'completion_tokens': 1,
                                                       'total_tokens': 2}}).encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # an interrupted request (Stop) closes its stream first


class Chat:
    """One person in one Telegram chat, talking to the gateway through the file double."""

    def __init__(self, home, user, chat, chat_type, chat_name=None):
        self.home, self.user, self.chat, self.chat_type, self.chat_name = home, user, chat, chat_type, chat_name

    def ask(self, text, timeout=60):
        with self.home.lock:
            self.home.sent += 1
            number = self.home.sent
        item = dict(text=text, user_id=self.user, user_name=self.user.title(), chat_id=self.chat,
                    chat_type=self.chat_type, chat_name=self.chat_name, message_id=f'msg-{number}')
        inbox = self.home.path / 'fake-inbox'
        staged = inbox / f'{number:04d}.tmp'
        staged.write_text(json.dumps(item), encoding='utf-8')
        staged.rename(inbox / f'{number:04d}.json')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            replies = self.home.replies()
            if len(replies) >= number:
                return replies[number - 1]['content']
            time.sleep(0.05)
        raise AssertionError(f'no reply to {text!r}: {self.home.replies()[number - 1:]}')


class Home:
    def __init__(self, path):
        self.path, self.sent, self.lock = path, 0, threading.Lock()

    def replies(self):
        path = self.path / 'fake-outbox.jsonl'
        lines = path.read_text(encoding='utf-8-sig').splitlines() if path.exists() else []
        # The gateway's own "connected" notice is not an answer to a message.
        return [line for line in map(json.loads, lines) if not line['content'].startswith('This chat is connected')]


@contextmanager
def gateway(tmp_path):
    """The Telegram Bot's gateway (profile default) and the second member's standalone gateway."""
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    two = home / 'profiles' / 'two'
    two.mkdir(parents=True)
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.blocked, model.release = threading.Event(), threading.Event()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    config = {'hosted_rooms': {'profiles': {'two': str(two)}},
              'model': {'provider': 'custom', 'default': 'gpt-4o', 'base_url': base},
              'platform_toolsets': {'gui': ['terminal'], 'bot_room': [], 'telegram': []},
              'approvals': {'mode': 'manual', 'timeout': 120},
              'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)},
              'platforms': {'telegram': {'enabled': True, 'token': 'fixture-token', 'extra': {
                  'allow_admin_from': ['alice'], 'group_allow_admin_from': ['bob']}}}}
    (home / 'config.yaml').write_text(json.dumps(config), encoding='utf-8')
    (two / 'config.yaml').write_text(json.dumps({**config, 'platforms': {}, 'gateway': {'standalone': True}}),
                                     encoding='utf-8')
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
                             OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1',
                             TELEGRAM_ALLOWED_USERS='alice,bob,carol', TELEGRAM_BOT_TOKEN='fixture-token')
    try:
        with daemon(root, two, env | {'HERMES_HOME': str(two)}, barrier=False), \
                daemon(root, home, env, barrier=True, fixture='group_chat_messaging_daemon.py') as (_, desc):
            yield _Journey(root, Home(home), env, model, desc)
    finally:
        model.release.set()
        model.shutdown()
        model.server_close()


class _Journey:
    def __init__(self, root, home, env, model, desc):
        self.root, self.home, self.env, self.model, self.desc = root, home, env, model, desc
        self.desktop = None

    def chat(self, user, chat, chat_type, chat_name=None):
        return Chat(self.home, user, chat, chat_type, chat_name)

    @asynccontextmanager
    async def open(self, *rooms):
        async with websocket(self.home.path, self.desc) as self.desktop:
            for room_id in rooms:
                created = await rpc(self.desktop, 'groups.create', room_id=room_id, name=room_id.title(),
                                    members=MEMBERS)
                assert 'result' in created, created
            yield self

    def cli(self, *args):
        done = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'groups', *args], env=self.env,
                              cwd=self.root, capture_output=True, text=True, timeout=180)
        assert done.returncode == 0, (done.stdout, done.stderr[-3000:])
        return done.stdout

    async def ask(self, chat, text):
        return await asyncio.to_thread(chat.ask, text)

    async def connect(self, chat):
        reply = await self.ask(chat, '/group')
        code = reply.split('hermes groups allow ')[1].split()[0]
        return reply, await asyncio.to_thread(self.cli, 'allow', code, '--yes')

    async def number(self, chat, name):
        listing = await self.ask(chat, '/group')
        return int(re.search(rf'^(\d+)\. {name} ·', listing, re.MULTILINE).group(1))

    async def until(self, check, what, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if found := await check():
                return found
            await asyncio.sleep(0.2)
        raise AssertionError('timed out waiting for ' + what)

    async def state(self, room):
        return (await rpc(self.desktop, 'groups.state', room_id=room))['result']['driver_status']

    async def approval(self, room):
        async def waiting():
            status = await self.state(room)
            return next((a for a in status['pending_actions'] if a['kind'] == 'approval'), None)
        return await self.until(waiting, f'an approval in {room}')

    async def replies(self, room, count):
        async def enough():
            events = (await rpc(self.desktop, 'groups.log', room_id=room, limit=500))['result']['events']
            done = [e['payload']['text'] for e in events if e['kind'] == 'message.member'
                    and e['payload']['member_id'] == 'ada' and e['payload']['text'].startswith('DONE')]
            return done if len(done) >= count else None
        return await self.until(enough, f'{count} replies in {room}')

    async def code(self, chat, ref, room):
        await self.approval(room)
        detail = await self.ask(chat, f'/group {ref}')
        return re.search(r'Approval (\w{6}) ', detail).group(1), detail

    async def always(self, chat, ref, room, replies):
        """Send, choose "always" (warning, then confirm), then let the rule answer the next one."""
        await self.ask(chat, f'/group {ref} send @ada RUN_ALWAYS first')
        code, detail = await self.code(chat, ref, room)
        assert f'approve {code} once|always|deny' in detail and '```\nrm -rf ./build-cache\n```' in detail
        assert 'Always allow this in this chat?' in await self.ask(chat, f'/group {ref} approve {code} always')
        assert (await self.ask(chat, f'/group {ref} approve {code} always confirm')).startswith('Allowed. Ada may')
        await self.replies(room, replies + 1)
        await self.ask(chat, f'/group {ref} send @ada RUN_ALWAYS again')  # nobody answers this one
        assert (await self.replies(room, replies + 2))[-1].startswith('DONE {"output": "", "exit_code": 0')
        assert 'used 1 time' in await self.ask(chat, f'/group {ref}')

    async def once_and_deny(self, chat, ref, room, replies):
        await self.ask(chat, f'/group {ref} send @ada RUN_ONCE please')
        code, _ = await self.code(chat, ref, room)
        assert await self.ask(chat, f'/group {ref} approve {code} once') == 'Allowed once for Ada.'
        assert 'old-logs' in (await self.replies(room, replies + 1))[-1]
        await self.ask(chat, f'/group {ref} send @ada RUN_DENY please')
        code, _ = await self.code(chat, ref, room)
        assert await self.ask(chat, f'/group {ref} approve {code} deny') == 'Denied for Ada.'
        assert 'BLOCKED' in (await self.replies(room, replies + 2))[-1]

    async def stop(self, chat, ref, room):
        self.model.blocked.clear()
        self.model.release.clear()
        await self.ask(chat, f'/group {ref} send @ada BLOCK_STOP now')
        assert await asyncio.to_thread(self.model.blocked.wait, 120)
        reply = await self.ask(chat, f'/group {ref} stop')
        assert re.fullmatch(rf'Stopping work in Group {ref} \(\d+ tasks?\)\.', reply), reply
        self.model.release.set()

        async def stopped():
            status = await self.state(room)
            return status['counts'].get('cancelled') and not status['working']
        await self.until(stopped, f'Stop in {room}')

    async def authors(self, room):
        events = (await rpc(self.desktop, 'groups.log', room_id=room, limit=500))['result']['events']
        return {e['payload']['text']: e['actor'] for e in events if e['kind'] == 'message.user'}


def test_a_private_chat_lists_sends_answers_approvals_and_stops(tmp_path):
    async def journey(run):
        async with run.open('research'):
            alice = run.chat('alice', 'alice', 'dm')
            reply, allowed = await run.connect(alice)
            assert 'Everyone' not in reply and 'private chat with Alice (user ID alice)' in allowed
            research = await run.number(alice, 'Research')
            await run.always(alice, research, 'research', 0)
            await run.once_and_deny(alice, research, 'research', 2)
            sent = await rpc(run.desktop, 'groups.send', room_id='research', event_id='desktop-1',
                             payload={'text': 'from Desktop', 'thread_id': 'desktop-1'})
            assert sent['result']['accepted'], sent
            authors = await run.authors('research')
            assert authors['@ada RUN_ALWAYS first'] == {'kind': 'user', 'id': 'telegram:alice',
                                                        'display_name': 'Alice via Telegram'}
            assert authors['from Desktop'] == {'kind': 'user', 'id': 'desktop'}
            await run.stop(alice, research, 'research')

    with gateway(tmp_path) as run:
        asyncio.run(journey(run))


def test_a_shared_chat_is_its_own_audience_rules_and_grant(tmp_path):
    async def journey(run):
        async with run.open('research', 'ops'):
            alice = run.chat('alice', 'alice', 'dm')
            bob = run.chat('bob', 'team', 'group', 'Team')
            carol = run.chat('carol', 'team', 'group', 'Team')
            # The private chat's rule, to show it never applies elsewhere and survives a revoke.
            await run.connect(alice)
            research = await run.number(alice, 'Research')
            await run.always(alice, research, 'research', 0)

            # Only the group admins may use /group in the shared chat; everyone there can read it.
            assert 'admin-only' in await run.ask(carol, '/group')  # the Bot's own slash gate refuses
            reply, allowed = await run.connect(bob)
            assert 'Everyone in this chat will be able to read' in reply
            assert 'shared chat "Team" (chat ID team)' in allowed and 'Everyone in that chat' in allowed
            assert "group_allow_admin_from list" in allowed
            assert 'admin-only' in await run.ask(carol, '/group 1')
            ops = await run.number(bob, 'Ops')
            await run.ask(bob, f'/group {ops} send @ada RUN_ALWAYS from the team')
            code, detail = await run.code(bob, ops, 'ops')
            assert 'Always allowed' not in detail  # the private chat's rule is not this chat's
            await run.ask(bob, f'/group {ops} approve {code} always')
            assert (await run.ask(bob, f'/group {ops} approve {code} always confirm')).startswith('Allowed.')
            await run.replies('ops', 1)
            await run.ask(bob, f'/group {ops} send @ada RUN_ALWAYS again')
            await run.replies('ops', 2)
            await run.once_and_deny(bob, ops, 'ops', 2)
            assert (await run.authors('ops'))['@ada RUN_ALWAYS from the team'] == {
                'kind': 'user', 'id': 'telegram:bob', 'display_name': 'Bob via Telegram'}

            # The owner sees both chats and what they always allow, then revokes the shared one.
            chats = await asyncio.to_thread(run.cli, 'chats')
            assert chats.count('always allowed in') == 2, chats
            team = re.search(r'^(\w{8})  telegram shared chat "Team"', chats, re.MULTILINE).group(1)
            assert 'Revoked' in await asyncio.to_thread(run.cli, 'revoke', team)
            assert 'hermes groups allow' in await run.ask(bob, '/group')
            await run.ask(alice, f'/group {research} send @ada RUN_ALWAYS still mine')
            await run.replies('research', 3)  # the private chat's own rule still applies
            await rpc(run.desktop, 'groups.send', room_id='ops', event_id='desktop-2',
                      payload={'text': '@ada RUN_ALWAYS after revoke', 'thread_id': 'desktop-2'})
            action = await run.approval('ops')  # the revoked chat's rule ended with it
            denied = await rpc(run.desktop, 'groups.approve', room_id='ops', choice='deny', **{
                k: action[k] for k in ('member_id', 'task_id', 'execution_generation', 'request_id')})
            assert 'result' in denied, denied
            await run.replies('ops', 5)

            # Connected again, the shared chat stops work too.
            await run.connect(bob)
            await run.stop(bob, await run.number(bob, 'Ops'), 'ops')

    with gateway(tmp_path) as run:
        asyncio.run(journey(run))
