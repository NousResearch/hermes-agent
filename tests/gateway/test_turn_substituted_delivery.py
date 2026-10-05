"""Gateway decoration must not turn a substituted (pre_turn short-circuit) result into
text/media side effects."""
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
import asyncio
import pytest

from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_runner import TurnRunner


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['https://example.com/', '收到，稍后见。', '  exact  ', '[SILENT]'])
async def test_substituted_gateway_returns_exact_text_without_voice_footer_or_reasoning(text):
    runner = GatewayTurnMixin()
    result = {'final_response': text, 'substituted': True,
              'messages': [{'role': 'assistant', 'content': text}], 'last_reasoning': 'must not appear'}
    response, silent, messages = await runner._hmwa_shape_agent_response(
        result, None, [], None, None, None, None, None, None, None)
    assert response == text and silent is False
    assert runner._hmwa_prepend_reasoning(result, response, None, False) == text
    assert runner._hmwa_runtime_footer_line(result, None, 1) == ''
    runner._adapter_for_source = Mock(side_effect=AssertionError('unexpected voice/media route'))
    delivered = await runner._hmwa_deliver_turn_response(
        SimpleNamespace(), None, None, None, None, result, messages, response, '', False)
    assert delivered == text
    runner._adapter_for_source.assert_not_called()


@pytest.mark.parametrize('substituted', [True, False])
def test_stream_boundary_preserves_substituted_but_keeps_normal_path_repair(monkeypatch, substituted):
    repair = Mock(return_value='repaired')
    monkeypatch.setattr('gateway.run_turn_runner.repair_explicit_computer_use_media_paths', repair)
    owner = SimpleNamespace(_ctx=SimpleNamespace(result_holder=[None]))
    result = {'final_response': 'https://example.com/', 'substituted': substituted, 'messages': []}
    TurnRunner._finish_stream_consumer(owner, result, [], None)
    if substituted:
        repair.assert_not_called()
        assert result['final_response'] == 'https://example.com/'
    else:
        repair.assert_called_once()
        assert result['final_response'] == 'repaired'
    assert owner._ctx.result_holder[0] is result


@pytest.mark.asyncio
@pytest.mark.parametrize('text', [
    'https://example.com/', 'https://example.com/preview.png',
    'MEDIA:/tmp/not-an-attachment.png', '  exact  ', '[SILENT]',
    'first\n\n\nsecond',
])
async def test_generic_delivery_keeps_substituted_text_and_has_no_attachments(text):
    from gateway.platforms.weixin import WeixinAdapter
    from gateway.config import PlatformConfig

    adapter = WeixinAdapter(PlatformConfig(enabled=True, token='test-token',
                                           extra={'account_id': 'test-account'}))
    adapter._send_session = SimpleNamespace()
    adapter._token_store = SimpleNamespace(get=lambda *a: 'test-context')
    event = SimpleNamespace(_substituted=True)
    extracted = await adapter._extract_response_content(text, event, 'test', is_ephemeral_response=False)
    assert extracted.text_content == text
    assert not (extracted.images or extracted.media_files or extracted.local_files)
    assert not adapter._wants_auto_tts(event, 'test', asyncio.Event(), text, [])
    for name in ('extract_media', 'extract_images', 'extract_local_files'):
        setattr(adapter, name, Mock(side_effect=AssertionError('substituted entered media/format route')))


@pytest.mark.asyncio
async def test_weixin_send_skips_media_pipeline_for_substituted_text():
    """The platform send path must honour the generic substituted metadata too: a MEDIA-looking
    literal (and any URL) is delivered verbatim, never rewritten into an attachment."""
    from gateway.platforms.weixin import WeixinAdapter
    from gateway.config import PlatformConfig

    adapter = WeixinAdapter(PlatformConfig(enabled=True, token='test-token',
                                           extra={'account_id': 'test-account'}))
    adapter._send_session = SimpleNamespace()
    adapter._token = 'test-token'
    adapter._token_store = SimpleNamespace(get=lambda *a: 'test-context')
    sent: list[str] = []
    adapter._send_text_chunk = AsyncMock(side_effect=lambda **kw: sent.append(kw['chunk']))
    for name in ('extract_media', 'extract_images', 'extract_local_files', 'format_message'):
        setattr(adapter, name, Mock(side_effect=AssertionError('substituted entered media/format route')))

    literal = 'MEDIA:/tmp/not-an-attachment.png https://example.com/  exact  '
    result = await adapter.send('chat', literal, metadata={'_substituted': True})
    assert result.success
    assert ''.join(sent) == literal