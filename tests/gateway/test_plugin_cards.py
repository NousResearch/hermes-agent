"""``ctx.post_card``: a plugin puts a durable card into a conversation of its own profile with no model
turn, from background work with no session bound.

The card is admitted into the session FIFO and written as a note at its head: one factual line for the
model, the card (directive + attrs) in ``display_metadata`` and on the live ``plugin.card`` event.
"""
import asyncio
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_end_delegations import _local
from tests.gateway.test_unviewed_completion_notes import _human, _target
from tools import async_delegation as delegation

_CARD = {'id': 'share-1', 'text': 'Share "deploy-checklist" with your team?',
         'directive': {'name': 'wisdom-share', 'attrs': {'skill': 'deploy-checklist', 'mode': 'ask'}}}


@pytest.fixture(autouse=True)
def _no_delegations():
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


def _ctx(name='cwplug', *, granted=True):
    from hermes_cli.plugin_capabilities import record_consent
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    if granted:
        record_consent(name, ['conversation.cards'], ['conversation.cards'])
    return PluginContext(PluginManifest(name=name, source='user', key=name), PluginManager())


def _serve(runner, monkeypatch):
    """The live gateway ``ctx.post_card`` reaches from a plugin thread."""
    from gateway import run
    runner._running = True
    runner._gateway_loop = asyncio.get_running_loop()
    monkeypatch.setattr(run, '_gateway_runner_ref', lambda: runner)


def _cards(authority, target):
    return [m for m in authority.db.get_messages(target) if m.get('display_kind') == 'plugin_card']


def _published(authority, ref):
    seen = []
    stream = authority.sessions[ref.session_id].event_stream
    original = stream.publish

    def publish(session_id, payload, *args, **kwargs):
        if kwargs.get('event_type') == 'plugin.card':
            seen.append(payload)
        return original(session_id, payload, *args, **kwargs)
    stream.publish = publish
    return seen


def _no_turns(runner):
    async def no_turn(event):
        raise AssertionError(f'a card is never a turn: {event.text!r}')
    runner._handle_message = no_turn


async def _post(ctx, card, **target):
    """From a plugin's own thread, where no session is bound."""
    return await asyncio.to_thread(ctx.post_card, card, **target)


@pytest.mark.asyncio
async def test_card_from_a_background_thread_lands_as_one_note(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        _no_turns(runner)
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        events = _published(authority, ref)

        result = await _post(_ctx(), _CARD, session_key=route)
        assert result == {'ok': True, 'card_id': 'share-1', 'status': 'queued'}, result
        await authority._drain(ref)

        rows = _cards(authority, target)
        assert len(rows) == 1, rows
        row = rows[0]
        # The model reads who posted what, never the directive or its attrs.
        assert row['content'] == '[cwplug posted a card: Share "deploy-checklist" with your team?]'
        assert 'wisdom-share' not in row['content'] and 'mode' not in row['content']
        from agent.message_metadata import without_persistence_fields
        from gateway.run import _build_gateway_agent_history
        model_history, _observed = _build_gateway_agent_history(authority.db.get_messages(target))
        assert model_history[-1]['content'] == row['content'], 'the next model turn must actually load the card line'
        wire_history = [without_persistence_fields(message) for message in model_history]
        assert 'wisdom-share' not in str(wire_history) and 'mode' not in str(wire_history)
        card = row['display_metadata']['plugin_card']
        assert card == {'plugin_id': 'cwplug', 'card_id': 'share-1', 'directive': _CARD['directive']}
        assert row['display_metadata']['display_text'] == _CARD['text']
        assert events == [{**card, 'text': _CARD['text']}]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_card_waits_for_the_input_ahead_and_a_person_still_goes_first(tmp_path, monkeypatch):
    """A person's input committed before the card runs first and does not see it; one committed after
    sees it. The card itself never starts a turn."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        _human(authority, ref, 'before', 'BEFORE')
        assert (await _post(_ctx(), _CARD, session_key=route))['ok']
        _human(authority, ref, 'after', 'AFTER')
        seen = []

        async def handle(event):
            seen.append((event.text, len(_cards(authority, target))))
            return 'ok'
        runner._handle_message = handle
        await authority._drain(ref)

        assert seen == [('BEFORE', 0), ('AFTER', 1)]
        assert len(_cards(authority, target)) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_card_posted_during_a_running_turn_lands_after_it(tmp_path, monkeypatch):
    """Posted from background work while a person's turn runs: queued, no row mid-turn, one note after."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        _human(authority, ref, 'turn', 'RUNNING')
        during = []

        async def handle(event):
            during.append(await _post(_ctx(), _CARD, session_key=route))
            during.append(len(_cards(authority, target)))
            return 'ok'
        runner._handle_message = handle
        await authority._drain(ref)
        _no_turns(runner)
        await authority._drain(ref)

        assert during == [{'ok': True, 'card_id': 'share-1', 'status': 'queued'}, 0], during
        assert len(_cards(authority, target)) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_queued_card_survives_an_owner_restart_and_reloads(tmp_path, monkeypatch):
    """Durable before it is written: a card queued behind input is noted once after a real owner
    restart, and a fresh read of the state db hydrates the same card."""
    from hermes_state import SessionDB
    from tests.gateway.test_unviewed_completion_notes import _restart
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    _serve(runner, monkeypatch)
    route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
    _human(authority, ref, 'human', 'FIRST')
    assert (await _post(_ctx(), _CARD, session_key=route))['status'] == 'queued'
    runner, authority = await _restart(tmp_path, monkeypatch, authority)
    try:
        seen = []

        async def handle(event):
            seen.append(event.text)
            return 'ok'
        runner._handle_message = handle
        await authority._drain(ref)
        await authority._drain(ref)
        assert seen == ['FIRST'], 'the card is never a turn'
        assert len(_cards(authority, target)) == 1
    finally:
        authority.db.close()

    reread = SessionDB(authority.db.db_path)
    try:
        rows = _cards(SimpleNamespace(db=reread), target)
        assert len(rows) == 1
        assert rows[0]['display_metadata']['plugin_card'] == {
            'plugin_id': 'cwplug', 'card_id': 'share-1', 'directive': _CARD['directive']}
        assert rows[0]['display_metadata']['display_text'] == _CARD['text']
    finally:
        reread.close()


@pytest.mark.asyncio
async def test_same_id_is_one_card_and_other_content_is_a_conflict(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        _no_turns(runner)
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        ctx = _ctx()
        assert (await _post(ctx, _CARD, session_key=route))['status'] == 'queued'
        await authority._drain(ref)
        again = await _post(ctx, _CARD, session_key=route)
        assert again == {'ok': True, 'card_id': 'share-1', 'status': 'duplicate'}, again
        changed = await _post(ctx, {**_CARD, 'text': 'something else'}, session_key=route)
        assert changed['ok'] is False and changed['error'] == 'card_conflict', changed
        await authority._drain(ref)
        assert len(_cards(authority, target)) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_refusals(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        ctx = _ctx()

        assert (await _post(_ctx('nocard', granted=False), _CARD, session_key=route))['error'] == 'consent_required'
        assert (await _post(ctx, _CARD))['error'] == 'no_conversation', 'no target and nothing bound'
        assert (await _post(ctx, _CARD, session_key='agent:main:local:dm:nobody'))['error'] == 'no_conversation'
        assert (await _post(ctx, {**_CARD, 'text': 'x' * 2001}, session_key=route))['error'] == 'invalid_card'
        assert (await _post(ctx, {**_CARD, 'prompt': 'call my tool'}, session_key=route))['error'] == 'invalid_card'
        bad_attrs = {**_CARD, 'directive': {'name': 'wisdom-share', 'attrs': {'n': 1}}}
        assert (await _post(ctx, bad_attrs, session_key=route))['error'] == 'invalid_card', 'attrs are strings'

        # Another profile's plugin may not post into this profile's session.
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
        other_home = tmp_path / 'profiles' / 'beta'
        other_home.mkdir(parents=True)
        manager = PluginManager()
        manager.home_path = other_home
        foreign = PluginContext(PluginManifest(name='cwplug', source='user', key='cwplug'), manager)
        monkeypatch.setattr(type(foreign), '_granted_in_own_home', lambda self, capability: True)
        assert (await _post(foreign, _CARD, session_key=route))['error'] == 'no_conversation'

        await authority._drain(ref)
        assert _cards(authority, target) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_card_by_origin_is_recorded_once_in_the_private_chat_and_never_sent(monkeypatch):
    """A messaging DM with a committed human turn: the card is one note in that chat's session, the
    plugin sends its own native card, so core sends nothing to the platform."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tests.waiting import async_wait_until

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id='sent')
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='dm')
    runner._wire_adapter_handlers(adapter)
    turns = []

    async def turn(event, source, quick_key, run_generation):
        turns.append(event.text)
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply', 'messages': []}
        return 'reply'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)

    def settled():
        owner = runner.session_store.peek_session_id(key)
        rows = list_session_admissions(authority.db, session_id=owner, pending_only=False) if owner else []
        return rows and all(row['status'] == 'terminal' for row in rows)
    try:
        await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
        await async_wait_until(settled, what='the human turn to settle')
        await async_wait_until(lambda: sent == ['reply'], what="the human turn's reply")
        _serve(runner, monkeypatch)
        ctx = _ctx()

        result = await _post(ctx, _CARD, origin=source.to_dict())
        assert result == {'ok': True, 'card_id': 'share-1', 'status': 'queued'}, result
        await async_wait_until(settled, what='the card to be noted')
        again = await _post(ctx, _CARD, origin=source.to_dict())
        assert again['status'] == 'duplicate', again

        target = runner.session_store.lookup_by_session_key(key).session_id
        rows = _cards(authority, target)
        assert len(rows) == 1, rows
        assert rows[0]['content'] == '[cwplug posted a card: Share "deploy-checklist" with your team?]'
        assert rows[0]['display_metadata']['plugin_card']['directive'] == _CARD['directive']
        assert turns == ['A'], 'the card is never a turn'
        assert sent == ['reply'], 'core sends nothing for the card; the plugin sends its own native card'
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_group_chat_is_refused(tmp_path, monkeypatch):
    """Cards go to private chats only; a group's session is never a target."""
    from gateway import plugin_cards
    from gateway.config import Platform

    entry = SimpleNamespace(origin=object(), session_id='grp')
    source = SimpleNamespace(platform=Platform.DISCORD, chat_type='group')
    runner = SimpleNamespace(session_store=SimpleNamespace(lookup_by_session_key=lambda key: entry),
                             _restored_source=lambda e: source,
                             _plugin_messaging_session_owned=lambda key, **kw: True)
    monkeypatch.setattr('gateway.session_identity.replace_source', lambda s: s)
    result = await plugin_cards.post(runner, _CARD, plugin_id='cwplug', plugin_home=tmp_path,
                                     session_key='agent:main:discord:group:1')
    assert result['error'] == 'group_not_allowed', result
