"""Participant YOLO reaches a separate owner, active tools and cold restart."""
import asyncio
import json
from pathlib import Path
import shlex
import threading

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until


@pytest.mark.platforms('posix')
@pytest.mark.live_system_guard_bypass
def test_member_yolo_controls_real_owner_and_survives_restart(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    target = home / 'profiles' / 'two'
    target.mkdir(parents=True)
    user.mkdir()
    entered, release = threading.Event(), threading.Event()
    effects = target / 'effects'
    command = f'rm -rf {shlex.quote(str(target / "disposable"))}; echo ran >> {shlex.quote(str(effects))}'

    def answer(request):
        messages = request['body']['messages']
        last = max(i for i, message in enumerate(messages) if message['role'] == 'user')
        text = str(messages[last]['content'])
        if any(message['role'] == 'tool' for message in messages[last + 1:]):
            return Text('PASS')
        if 'BLOCK_YOLO' in text:
            entered.set()
            assert release.wait(30), 'test never released the model'
            return ToolCall('terminal', {'command': command})
        if 'ASK_APPROVAL' in text:
            return ToolCall('terminal', {'command': command})
        return Text('PASS')

    async def wait_state(ws, predicate):
        result = None
        async def ready():
            nonlocal result
            reply = await rpc(ws, 'groups.state', room_id='room')
            assert 'result' in reply, reply
            result = reply['result']['driver_status']
            return predicate(result)
        try:
            await async_wait_until(ready, timeout=40, what='room did not reach expected state')
        except AssertionError:
            pytest.fail(f'room did not settle: {result}')
        return result

    async def yolo(ws, member='two', **params):
        reply = await rpc(ws, 'groups.member.yolo', room_id='room', member_id=member, **params)
        assert 'result' in reply, reply
        return reply['result']

    async def send(ws, event_id, text):
        reply = await rpc(ws, 'groups.send', room_id='room', event_id=event_id,
                          payload={'text': text, 'thread_id': 'thread'})
        assert reply.get('result', {}).get('accepted'), reply

    async def exercise(desc, restart):
        async with websocket(home, desc) as ws:
            if restart:
                assert (await yolo(ws))['enabled'] is False
                assert (await yolo(ws, 'one'))['enabled'] is True
                assert effects.read_text().splitlines() == ['ran']
                return
            caps = (await rpc(ws, 'groups.capabilities'))['result']
            assert 'groups.member.yolo' in caps['methods']
            members = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                       {'member_id': 'two', 'profile': 'two', 'handle': 'two'}]
            assert 'result' in await rpc(ws, 'groups.create', room_id='room', name='Room', members=members)
            missing = await rpc(ws, 'groups.member.yolo', room_id='room', member_id='two', enabled=True)
            assert missing['error']['message'] == 'not_found', missing
            await send(ws, 'warm', 'WARM')
            await wait_state(ws, lambda state: state['counts'].get('settled') == 2)
            assert (await yolo(ws))['enabled'] is False
            await send(ws, 'active', '@two BLOCK_YOLO')
            assert await asyncio.to_thread(entered.wait, 30), 'member never reached the model'
            on = await yolo(ws, enabled=True)
            assert on['enabled'] and on['effective_enabled']
            assert await yolo(ws, enabled=True) == on
            assert (await yolo(ws, 'one'))['enabled'] is False
            release.set()
            await wait_state(ws, lambda state: state['counts'].get('settled') == 3)
            assert effects.read_text().splitlines() == ['ran']
            await yolo(ws, enabled=False)
            await send(ws, 'approval', '@two ASK_APPROVAL')
            state = await wait_state(ws, lambda state: bool(state['pending_actions']))
            pending, = state['pending_actions']
            await yolo(ws, enabled=True)
            state = (await rpc(ws, 'groups.state', room_id='room'))['result']['driver_status']
            assert state['pending_actions'] == [pending], 'YOLO must not answer an already pending approval'
            params = {key: pending[key] for key in ('member_id', 'task_id', 'execution_generation', 'request_id')}
            reply = await rpc(ws, 'groups.approve', room_id='room', choice='deny', **params)
            assert reply['result']['approved'], reply
            await wait_state(ws, lambda state: state['counts'].get('settled') == 4)
            await yolo(ws, enabled=False)
            await yolo(ws, 'one', enabled=True)
            assert effects.read_text().splitlines() == ['ran']

    with FakeLLMServer(answer) as model:
        config = {
            'hosted_rooms': {'profiles': {'two': str(target)}},
            'model': {'provider': 'custom', 'default': 'gpt-4o', 'base_url': model.base_url},
            'platform_toolsets': {'gui': ['terminal'], 'bot_room': ['terminal']},
            'auxiliary': {'title_generation': {'enabled': False}},
            'approvals': {'mode': 'manual'}, 'terminal': {'cwd': str(target)},
        }
        (home / 'config.yaml').write_text(json.dumps(config))
        (target / 'config.yaml').write_text(json.dumps({**config, 'gateway': {'standalone': True}}))
        env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
                                PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only')
        try:
            for restart in (False, True):
                with (daemon(root, target, env | {'HERMES_HOME': str(target)}, barrier=False),
                      daemon(root, home, env, barrier=False) as (_, desc)):
                    asyncio.run(exercise(desc, restart))
        finally:
            release.set()
