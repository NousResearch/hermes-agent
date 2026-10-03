"""Group Chat notice buttons on each platform: native buttons whose data is the whole choice.

The gateway resolves a tap from ``hg:<action>:<token>`` alone (gateway.group_chat_actions), so
nothing is kept in the adapter: a tap works after a restart and after other prompts. Each test
drives one adapter's send and tap paths against a runner that records what it was asked.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.test_discord_clarify_buttons import _make_adapter as discord_adapter
from tests.gateway.test_discord_clarify_buttons import _make_interaction
from tests.gateway.test_slack_clarify_buttons import _make_adapter as slack_adapter
from tests.gateway.test_telegram_approval_buttons import _make_adapter as telegram_adapter
from tests.gateway.test_whatsapp_cloud import _make_adapter as whatsapp_adapter
from tests.gateway.test_whatsapp_cloud import _mock_httpx_response

TOKEN = 'AbCdEfGhIjKl'
BUTTONS = [('Keep going', f'hg:go:{TOKEN}'), ('Go back to Mac mini', f'hg:back:{TOKEN}'),
           ('Ask me first', f'hg:ask:{TOKEN}')]
CONFIRM = {'text': 'Go back to Mac mini? Home VPS pauses now…', 'buttons': [
    ('Go back to Mac mini', f'hg:back!:{TOKEN}'), ('Cancel', f'hg:no:{TOKEN}')]}


class Runner:
    """What an adapter hands the gateway on a tap, and the gateway's answer."""

    def __init__(self, answer=CONFIRM):
        self.taps, self.answer = [], answer

    async def _group_chat_action(self, platform, chat_id, user_id, data):
        self.taps.append((platform, chat_id, user_id, data))
        return self.answer


def test_every_choice_fits_telegrams_callback_data():
    from gateway.group_chat_actions import data_for, label
    for action in ('go', 'back', 'back!', 'ask', 'k0', 'k1!', 'cont', 'cont!', 'no'):
        assert len(data_for(action, TOKEN).encode()) <= 64
    assert label('Continue on ', 'Home Server Room') == 'Continue on Home Server Room'
    assert label('Keep going on ', 'The very long name of a computer') == 'Keep going on The very long nam…'
    assert len(label('Go back to ', 'x' * 80)) == 32  # WhatsApp's adapter cuts its own labels to 20


@pytest.mark.asyncio
async def test_telegram_sends_inline_buttons_and_edits_in_place_on_a_tap(monkeypatch):
    from plugins.platforms.telegram import adapter as telegram
    adapter = telegram_adapter()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=7))
    monkeypatch.setattr(telegram, 'InlineKeyboardButton', lambda text, callback_data: (text, callback_data))
    monkeypatch.setattr(telegram, 'InlineKeyboardMarkup', lambda rows: rows)
    sent = await adapter.send_group_actions('12345', '“Research” moved to <Home VPS>', BUTTONS)
    assert sent.success
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert 'HTML' in repr(kwargs['parse_mode']) and kwargs['text'] == '“Research” moved to &lt;Home VPS&gt;'
    assert kwargs['reply_markup'] == [[button] for button in BUTTONS]
    # The tap needs nothing the adapter kept: a fresh adapter (a restart) resolves it through the gateway.
    adapter = telegram_adapter()
    adapter.gateway_runner = runner = Runner()
    # Another prompt opened in the chat since (a /reasoning picker) takes nothing from this notice.
    adapter._choice_picker_state['12345'] = {'msg_id': 8, 'choices': [], 'session_key': 's', 'on_choice_selected': None}
    monkeypatch.setattr(adapter, '_callback_authorized', AsyncMock(return_value=True))
    query = AsyncMock()
    query.data, query.message, query.from_user = f'hg:back:{TOKEN}', MagicMock(), MagicMock()
    query.message.chat_id, query.from_user.id = 12345, 42
    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())
    assert runner.taps == [('telegram', '12345', '42', f'hg:back:{TOKEN}')]
    edit = query.edit_message_text.call_args.kwargs
    assert edit['text'] == CONFIRM['text'] and edit['reply_markup'] == [[button] for button in CONFIRM['buttons']]
    runner.answer = None  # not this person's to choose: told so, the message stays
    query.edit_message_text.reset_mock()
    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())
    assert not query.edit_message_text.called
    assert query.answer.call_args.kwargs['text']


@pytest.mark.asyncio
async def test_discord_sends_buttons_without_a_timeout_and_routes_clicks_by_custom_id():
    adapter = discord_adapter(allowed_users={'42'})
    channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=9)))
    adapter._resolve_channel = AsyncMock(return_value=channel)
    sent = await adapter.send_group_actions('555', 'Notice', BUTTONS)
    assert sent.success
    view = channel.send.call_args.kwargs['view']
    assert view.timeout is None  # survives a restart: nothing waits on it
    assert [(b.label, b.custom_id) for b in view.children] == BUTTONS
    adapter = discord_adapter(allowed_users={'42'})
    adapter.gateway_runner = runner = Runner()
    interaction = _make_interaction(user_id='42')
    interaction.data, interaction.channel_id = {'custom_id': f'hg:back:{TOKEN}'}, 555
    interaction.edit_original_response = AsyncMock()
    interaction.followup = SimpleNamespace(send=AsyncMock())
    await adapter._on_group_action_interaction(interaction)
    assert runner.taps == [('discord', '555', '42', f'hg:back:{TOKEN}')]
    interaction.response.defer.assert_awaited()
    edit = interaction.edit_original_response.call_args.kwargs
    assert edit['content'] == CONFIRM['text']
    assert [(b.label, b.custom_id) for b in edit['view'].children] == CONFIRM['buttons']
    other = _make_interaction(user_id='99')
    other.data, other.channel_id = {'custom_id': f'hg:go:{TOKEN}'}, 555
    await adapter._on_group_action_interaction(other)
    assert other.response.send_message.call_args.kwargs['ephemeral'] is True and len(runner.taps) == 1
    unrelated = _make_interaction(user_id='42')
    unrelated.data = {'custom_id': 'clarify:1:0'}
    await adapter._on_group_action_interaction(unrelated)
    assert len(runner.taps) == 1 and not unrelated.response.defer.called


@pytest.mark.asyncio
async def test_slack_sends_block_kit_buttons_and_updates_the_message_on_a_click():
    adapter = slack_adapter()
    client = adapter._team_clients['T1']
    client.chat_postMessage = AsyncMock(return_value={'ok': True, 'ts': '1.2'})
    adapter._get_client = lambda channel_id, team_id=None: client
    sent = await adapter.send_group_actions('D1', 'Notice', BUTTONS)
    assert sent.success and sent.message_id == '1.2'
    blocks = client.chat_postMessage.call_args.kwargs['blocks']
    assert blocks[0]['text'] == {'type': 'plain_text', 'text': 'Notice'}
    assert [(e['text']['text'], e['action_id'], e['value']) for e in blocks[1]['elements']] == [
        (name, f'hermes_group_{index}', data) for index, (name, data) in enumerate(BUTTONS)]
    adapter.gateway_runner = runner = Runner()
    adapter._begin_interaction = AsyncMock(return_value=(
        'T1', 'hermes_group_1', f'hg:back:{TOKEN}', {}, '1.2', 'D1', 'Alice', 'U42'))
    client.chat_update = AsyncMock()
    await adapter._handle_group_action(AsyncMock(), {}, {})
    assert runner.taps == [('slack', 'D1', 'U42', f'hg:back:{TOKEN}')]
    update = client.chat_update.call_args.kwargs
    assert update['ts'] == '1.2' and update['text'] == CONFIRM['text']
    assert [e['value'] for e in update['blocks'][1]['elements']] == [data for _, data in CONFIRM['buttons']]


@pytest.mark.asyncio
async def test_whatsapp_sends_reply_buttons_and_answers_a_tap_with_a_new_message():
    adapter = whatsapp_adapter()
    adapter._http_client = MagicMock()
    adapter._http_client.post = AsyncMock(return_value=_mock_httpx_response(200, {'messages': [{'id': 'w1'}]}))
    assert (await adapter.send_group_actions('15551234567', 'Notice', BUTTONS)).success
    payload = adapter._http_client.post.call_args.kwargs['json']
    assert [(b['reply']['title'], b['reply']['id']) for b in payload['interactive']['action']['buttons']] == BUTTONS
    adapter.gateway_runner = runner = Runner()
    adapter._is_dm_allowed = lambda sender: sender == '15551234567'
    raw = {'from': '15551234567', 'type': 'interactive',
           'interactive': {'type': 'button_reply', 'button_reply': {'id': f'hg:back:{TOKEN}', 'title': 'Go back'}}}
    assert await adapter._dispatch_interactive_reply(raw, {}) is True
    assert runner.taps == [('whatsapp_cloud', '15551234567', '15551234567', f'hg:back:{TOKEN}')]
    payload = adapter._http_client.post.call_args.kwargs['json']  # WhatsApp can't edit: a new message
    assert payload['interactive']['body']['text'] == CONFIRM['text']
    assert [b['reply']['id'] for b in payload['interactive']['action']['buttons']] == [
        data for _, data in CONFIRM['buttons']]
