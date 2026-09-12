"""Request binding at actual PR dispatch/confirmation seams, with fixture transports.

No real model, account or platform is used. The interleavings are controlled.
"""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

H = runpy.run_path(str(Path(__file__).with_name('test_model_multiline_payload.py')))


def setup(tmp_path, monkeypatch):
    from tools import slash_confirm
    H['_setup_isolated_home'](tmp_path, monkeypatch)
    H['_quiet_model_switch_guards'](monkeypatch)
    H['_fire_selection_guard'](monkeypatch)
    H['_fake_switch_model'](monkeypatch, seen_raw_inputs=[])
    runner, adapter = H['_make_runner']()
    event = lambda text, thread='binding': H['_make_event'](text, thread_id=thread)
    key = runner._session_key_for_source(event('').source)
    slash_confirm.clear(key)
    return runner, adapter, event, key, slash_confirm


def record(case, **values):
    target = os.environ.get('MODEL_BINDING_OBSERVATIONS')
    if target:
        with open(target, 'a', encoding='utf-8') as out:
            out.write(json.dumps(dict(case=case, **values), ensure_ascii=False)+'\n')


def routed(runner):
    return [call.args[0].text for call in runner._handle_message_with_agent.await_args_list]


def test_request_whitespace_survives_approval(tmp_path, monkeypatch):
    async def scenario():
        r, _, event, key, sc = setup(tmp_path, monkeypatch)
        payload = '  한국어 요청\n    keep indentation\n'
        try:
            await r._handle_message(event('/model proposal\n'+payload))
            await r._handle_message(event('!approve'))
            record('exact_text', expected=payload, received=routed(r))
            assert routed(r) == [payload]
        finally:
            sc.clear(key)
    asyncio.run(scenario())


def test_failed_new_proposal_does_not_replace_pending_request(tmp_path, monkeypatch):
    async def scenario():
        r, _, event, key, sc = setup(tmp_path, monkeypatch)
        try:
            await r._handle_message(event('/model first\nFIRST'))
            previous = sc.get_pending(key)
            r._perform_model_switch = AsyncMock(return_value=(None, 'Error: rejected proposal'))
            await r._handle_message(event('/model bad\nUNACCEPTED'))
            assert sc.get_pending(key)['confirm_id'] == previous['confirm_id']
            await r._handle_message(event('!approve'))
            record('failed_proposal', received=routed(r))
            assert routed(r) == ['FIRST']
        finally:
            sc.clear(key)
    asyncio.run(scenario())


def test_help_keeps_the_existing_confirmation(tmp_path, monkeypatch):
    async def scenario():
        r, _, event, key, sc = setup(tmp_path, monkeypatch)
        try:
            await r._handle_message(event('/model first\nFIRST'))
            previous = sc.get_pending(key)
            r._model_listing_reply = AsyncMock(return_value='Available models')
            assert await r._handle_message(event('/model')) == 'Available models'
            assert sc.get_pending(key)['confirm_id'] == previous['confirm_id']
            await r._handle_message(event('!approve'))
            record('help', received=routed(r))
            assert routed(r) == ['FIRST']
        finally:
            sc.clear(key)
    asyncio.run(scenario())


def test_late_prompt_return_cannot_overwrite_newer_confirmation(tmp_path, monkeypatch):
    async def scenario():
        r, adapter, event, key, sc = setup(tmp_path, monkeypatch)
        ready, release = asyncio.Event(), asyncio.Event()
        sends = [0]
        async def present(**kwargs):
            sends[0] += 1
            if sends[0] == 1:
                ready.set()
                await release.wait()
            return None
        adapter.send_slash_confirm.side_effect = present
        first = event('/model first\nFIRST')
        task = asyncio.create_task(r._hm_cmd_model(first, first.source, key))
        try:
            await asyncio.wait_for(ready.wait(), 3)
            old_id = sc.get_pending(key)['confirm_id']
            second = event('/model second\nSECOND')
            await r._hm_cmd_model(second, second.source, key)
            new_id = sc.get_pending(key)['confirm_id']
            assert old_id != new_id
            release.set()
            await asyncio.wait_for(task, 3)
            await r._handle_message(event('!approve'))
            record('late_presentation', old_id=old_id, new_id=new_id, received=routed(r))
            assert routed(r) == ['SECOND']
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            sc.clear(key)
    asyncio.run(scenario())


def test_fast_confirmation_has_payload_before_prompt_returns(tmp_path, monkeypatch):
    async def scenario():
        r, adapter, event, key, sc = setup(tmp_path, monkeypatch)
        ready, release = asyncio.Event(), asyncio.Event()
        async def present(**kwargs):
            ready.set()
            await release.wait()
            return None
        adapter.send_slash_confirm.side_effect = present
        request = event('/model first\nFIRST')
        task = asyncio.create_task(r._hm_cmd_model(request, request.source, key))
        try:
            await asyncio.wait_for(ready.wait(), 3)
            pending = sc.get_pending(key)
            reply = await sc.resolve(key, pending['confirm_id'], 'once')
            duplicate = await sc.resolve(key, pending['confirm_id'], 'once')
            payload = getattr(reply, 'payload', None)
            release.set()
            await asyncio.wait_for(task, 3)
            record('fast_confirmation', payload=payload, duplicate=duplicate,
                   stash_empty=not r._model_inline_payload_stash())
            assert payload == 'FIRST'
            assert duplicate is None
            assert r._model_inline_payload_stash() == {}
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            sc.clear(key)
    asyncio.run(scenario())


@pytest.mark.parametrize('choice', ['once', 'cancel'])
def test_displaced_callback_cannot_consume_successor(tmp_path, monkeypatch, choice):
    async def scenario():
        r, _, event, key, sc = setup(tmp_path, monkeypatch)
        try:
            await r._handle_message(event('/model first\nFIRST'))
            old = sc.get_pending(key)
            await r._handle_message(event('/model second\nSECOND'))
            new = sc.get_pending(key)
            # Directly call a retained callback to isolate the ownership guard.
            # Normal resolve() rejects the old confirm_id before reaching it.
            r._commit_model_switch = AsyncMock(return_value='should not be called')
            reply = await old['handler'](choice)
            record('displaced_'+choice, commits=r._commit_model_switch.await_count,
                   current=str(r._model_inline_payload_stash().get(key)), reply=str(reply))
            r._commit_model_switch.assert_not_awaited()
            assert r._model_inline_payload_stash()[key] == 'SECOND'
            assert sc.get_pending(key)['confirm_id'] == new['confirm_id']
        finally:
            sc.clear(key)
    asyncio.run(scenario())


def test_boundary_revocation_prevents_retained_callback(tmp_path, monkeypatch):
    async def scenario():
        r, _, event, key, sc = setup(tmp_path, monkeypatch)
        try:
            await r._handle_message(event('/model first\nFIRST'))
            pending = sc.get_pending(key)
            # Same binding removal as the conversation-scope funnel, isolated here.
            # The PR's unmodified reset test additionally invokes the actual reset path.
            r._model_inline_payload_stash().pop(key)
            r._commit_model_switch = AsyncMock(return_value='not a current request')
            await pending['handler']('once')
            record('revoked_callback', commits=r._commit_model_switch.await_count)
            r._commit_model_switch.assert_not_awaited()
        finally:
            sc.clear(key)
    asyncio.run(scenario())


def test_expired_confirmation_alone_never_routes(tmp_path, monkeypatch):
    async def scenario():
        r, _, event, key, sc = setup(tmp_path, monkeypatch)
        clock = [1000.0]
        monkeypatch.setattr(sc, 'time', SimpleNamespace(time=lambda: clock[0]))
        try:
            await r._handle_message(event('/model first\nFIRST'))
            pending = sc.get_pending(key)
            clock[0] += sc.DEFAULT_TIMEOUT_SECONDS + 1
            result = await sc.resolve(key, pending['confirm_id'], 'once')
            record('expired', result=result, overrides=dict(r._session_model_overrides))
            assert result is None
            assert r._session_model_overrides == {}
            assert routed(r) == []
        finally:
            sc.clear(key)
    asyncio.run(scenario())


def test_direct_switch_does_not_clear_newer_guard_after_await(tmp_path, monkeypatch):
    async def scenario():
        from gateway.slash_commands_model import ModelSwitchConfirmation
        r, _, event, key, sc = setup(tmp_path, monkeypatch)
        ready, release = asyncio.Event(), asyncio.Event()
        original = r._handle_model_command
        async def handler(evt, **kwargs):
            if evt.get_command_args().strip() == 'direct':
                ready.set()
                await release.wait()
                return ModelSwitchConfirmation('Direct model selected')
            return await original(evt, **kwargs)
        r._handle_model_command = handler
        try:
            await r._handle_message(event('/model old\nOLD'))
            direct = event('/model direct')
            task = asyncio.create_task(r._hm_cmd_model(direct, direct.source, key))
            try:
                await asyncio.wait_for(ready.wait(), 3)
                next_event = event('/model next\nNEXT')
                await r._hm_cmd_model(next_event, next_event.source, key)
                release.set()
                await asyncio.wait_for(task, 3)
                await r._handle_message(event('!approve'))
                record('direct_return', received=routed(r))
                assert routed(r) == ['NEXT']
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        finally:
            sc.clear(key)
    asyncio.run(scenario())
