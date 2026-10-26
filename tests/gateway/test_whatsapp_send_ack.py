from unittest.mock import AsyncMock, MagicMock
import pytest
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize('status,data', [
    (200, {'success': True}),
    (200, {'success': True, 'messageId': ''}),
    (200, {'success': False, 'messageId': 'local-only'}),
    (502, {'success': False, 'status': 'PENDING', 'error': 'ACK_TIMEOUT'}),
])
async def test_send_refuses_missing_id_or_bridge_refusal(status, data):
    adapter = _make_adapter()
    resp = MagicMock(status=status)
    resp.json = AsyncMock(return_value=data)
    resp.text = AsyncMock(return_value='ACK_TIMEOUT: PENDING')
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
    result = await adapter.send('120363406889865453@g.us', 'OFFLINE ONLY')
    assert not result.success
    assert result.error


@pytest.mark.asyncio
async def test_send_accepts_acknowledged_id():
    adapter = _make_adapter()
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={'success': True, 'messageId': 'server-acked', 'status': 'SERVER_ACK'})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
    result = await adapter.send('120363406889865453@g.us', 'OFFLINE ONLY')
    assert result.success
    assert result.message_id == 'server-acked'
