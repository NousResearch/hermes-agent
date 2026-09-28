from unittest.mock import AsyncMock, patch

import pytest

from agent import i18n
from cron.scheduler_delivery import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig


@pytest.mark.parametrize('language, env, expected', [
    ('zh', None, 'zh'), ('ar', None, 'ar'), ('zh', 'en', 'en'),
    (None, None, 'en'), ('invalid', None, 'en'),
])
def test_wrapper_uses_job_profile_not_process_cache(tmp_path, monkeypatch, language, env, expected):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_LANGUAGE', raising=False)
    (tmp_path / 'config.yaml').write_text('display:\n  language: zh\n')
    i18n.reset_language_cache()
    assert i18n.get_language() == 'zh'
    (tmp_path / 'config.yaml').write_text('display:\n  language: ' + (language or 'null') + '\n')
    if env:
        monkeypatch.setenv('HERMES_LANGUAGE', env)
    cfg = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    send = AsyncMock(return_value={'success': True})
    try:
        with patch('gateway.config.load_gateway_config', return_value=cfg), patch('tools.send_message_tool._send_to_platform', send):
            _deliver_result({'id': 'job-1', 'name': 'report', 'deliver': 'telegram:123'}, 'BODY')
        content = send.call_args.args[3]
        assert content == i18n.t('cron.delivery.wrapper', lang=expected, task_name='report', job_id='job-1', content='BODY')
        if expected == 'en':
            assert content == ('Cronjob Response: report\n(job_id: job-1)\n-------------\n\nBODY\n\n'
                               'To stop or manage this job, send me a new message (e.g. "stop reminder report").')
    finally:
        i18n.reset_language_cache()


@pytest.mark.parametrize('platform, wrap', [('yuanbao', True), ('telegram', False)])
def test_clean_delivery_does_not_parse_localized_wrapper(tmp_path, monkeypatch, platform, wrap):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_LANGUAGE', 'zh')
    (tmp_path / 'config.yaml').write_text(f'cron:\n  wrap_response: {str(wrap).lower()}\n')
    cfg = GatewayConfig(platforms={Platform(platform): PlatformConfig(enabled=True)})
    send = AsyncMock(return_value={'success': True})
    with patch('gateway.config.load_gateway_config', return_value=cfg), patch('tools.send_message_tool._send_to_platform', send):
        _deliver_result({'id': 'job-1', 'deliver': f'{platform}:123'}, 'BODY')
    assert send.call_args.args[3] == 'BODY'
