import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import i18n
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.config import Platform
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_concurrent_profile_languages_restore_and_propagate(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_LANGUAGE', raising=False)
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    (tmp_path / 'config.yaml').write_text('display:\n  language: ar\n')
    i18n.reset_language_cache()
    assert i18n.get_language() == 'ar'
    homes = []
    for name, config in [('zh', 'display:\n  language: zh\n'), ('en', '{}\n')]:
        home = tmp_path / name
        home.mkdir()
        (home / 'config.yaml').write_text(config)
        homes.append(home)
    async def render(home, expected):
        with _profile_runtime_scope(home):
            await asyncio.sleep(0)
            assert i18n.get_language() == expected
            assert await asyncio.to_thread(i18n.get_language) == expected
            with _profile_runtime_scope(homes[1]):
                assert i18n.get_language() == 'en'
            assert i18n.get_language() == expected
    try:
        await asyncio.gather(render(homes[0], 'zh'), render(homes[1], 'en'))
        assert i18n.get_language() == 'ar'
        monkeypatch.setenv('HERMES_LANGUAGE', 'en')
        with _profile_runtime_scope(homes[0]):
            assert i18n.get_language() == 'en'
    finally:
        i18n.reset_language_cache()


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['suspended', 'daily', 'idle', 'resume_pending_expired'])
async def test_localized_reset_retains_current_notification_policy(tmp_path, monkeypatch, reason):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_LANGUAGE', raising=False)
    (tmp_path / 'config.yaml').write_text('display:\n  language: zh\n')
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(send=AsyncMock())
    runner._adapter_for_source = lambda source: adapter
    runner._profile_scope_for_source = lambda source: _profile_runtime_scope(tmp_path)
    monkeypatch.setattr('gateway.run._resolve_gateway_model_context', lambda: SimpleNamespace(
        model='model-id', provider='custom', context_length=8192, context_source='config', base_url=''))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='123', user_id='u')
    entry = SimpleNamespace(auto_reset_reason=reason)
    await runner._hmwa_deliver_auto_reset_notice(entry, source, [])
    if reason == 'suspended':
        content = adapter.send.call_args.args[1]
        assert '停止后的会话已重置' in content
        assert '◆ 模型：`model-id`' in content
        assert '/resume' in content
    else:
        adapter.send.assert_not_awaited()
