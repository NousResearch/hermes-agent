"""Direct Slack/Matrix fallback markers share the parent's task-local guard."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.native_document_guard import NativeDocumentFallback, require_native_document
from gateway.platforms.base import SendResult


@pytest.mark.asyncio
@pytest.mark.parametrize('platform', ['slack', 'matrix'])
@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('native_success', [False, True])
async def test_native_upload_and_text_fallback_remain_distinct(tmp_path, platform, strict, native_success):
    path = tmp_path / 'brief.txt'
    native = SendResult(success=True, message_id='native-1')
    notice = SendResult(success=True, message_id='notice-1')
    if platform == 'slack':
        from plugins.platforms.slack.adapter import SlackAdapter
        adapter = SlackAdapter(PlatformConfig(enabled=True, token='test-token'))
        adapter._app = object()
        adapter._dm_target = AsyncMock(return_value='D1')
        adapter._resolve_thread_ts = lambda *args: None
        adapter._upload_with_retry = AsyncMock(return_value=native)
        if not native_success:
            adapter._upload_with_retry.side_effect = RuntimeError('controlled upload failure')
        path.write_text('brief', encoding='utf-8')
    else:
        from plugins.platforms.matrix.adapter import MatrixAdapter
        adapter = MatrixAdapter(PlatformConfig(enabled=True, token='test-token'))
        adapter._client = object()
        adapter._upload_and_send = AsyncMock(return_value=native)
        if native_success:
            path.write_text('brief', encoding='utf-8')
    adapter.send = AsyncMock(return_value=notice)

    async def send():
        return await adapter.send_document('D1', str(path), caption='Brief')

    if strict:
        with require_native_document():
            if native_success:
                assert await send() is native
            else:
                with pytest.raises(NativeDocumentFallback):
                    await send()
    else:
        assert await send() is (native if native_success else notice)
    if native_success or strict:
        adapter.send.assert_not_awaited()
    else:
        adapter.send.assert_awaited_once()
    # Leaving the opt-in context restores ordinary adapter notices.
    if not native_success:
        assert await send() is notice
