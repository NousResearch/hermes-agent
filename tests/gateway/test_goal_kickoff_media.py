"""A ``/goal`` sent as a photo caption kicks off with the photo: the kickoff turn is rebuilt from the
command, and it used to keep only the text, so the agent never saw the file the goal was about."""
import pytest

from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_goal_captioned_photo_kickoff_carries_the_photo(monkeypatch, tmp_path):
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from tests.gateway.test_messaging_slash_command_fifo import _owner

    runner, adapter, sent = _owner(monkeypatch)
    await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key, ran = runner.session_store._generate_session_key(source), []

    async def turn(event, source, quick_key, run_generation):
        ran.append(event)
        if (captured := execution_result.get()) is not None:
            captured['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    async def no_judge(**_):
        return None
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    monkeypatch.setattr(runner, '_post_turn_goal_continuation', no_judge)

    chart = tmp_path / 'chart.png'
    chart.write_bytes(b'\x89PNG\r\n\x1a\n' + b'0' * 64)

    def idle():
        return not adapter._active_sessions and not runner._is_session_running(key)
    await adapter.handle_message(MessageEvent(text='hello', source=source, message_id='H'))
    await async_wait_until(lambda: ran and idle(), what='a session for the goal')
    await adapter.handle_message(MessageEvent(
        text='/goal describe this chart', message_type=MessageType.PHOTO, source=source, message_id='G',
        media_urls=[str(chart)], media_types=['image/png']))
    await async_wait_until(lambda: len(ran) >= 2 and idle(), what='the goal kickoff turn')

    kickoff = ran[-1]
    assert 'describe this chart' in kickoff.text, sent
    assert len(kickoff.media_urls) == 1 and kickoff.media_urls[0].endswith('.png'), kickoff.media_urls
    assert kickoff.media_types == ['image/png']
    assert kickoff.message_type == MessageType.PHOTO
