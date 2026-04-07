import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv('BLUEBUBBLES_SERVER_URL', 'http://localhost:1234')
    monkeypatch.setenv('BLUEBUBBLES_PASSWORD', 'secret')
    from gateway.platforms.bluebubbles import BlueBubblesAdapter
    cfg = PlatformConfig(enabled=True, extra={
        'server_url': 'http://localhost:1234',
        'password': 'secret',
        **extra,
    })
    return BlueBubblesAdapter(cfg)


class TestBlueBubblesPlatformEnum:
    def test_bluebubbles_enum_exists(self):
        assert Platform.BLUEBUBBLES.value == 'bluebubbles'


class TestBlueBubblesConfigLoading:
    def test_apply_env_overrides_bluebubbles(self, monkeypatch):
        monkeypatch.setenv('BLUEBUBBLES_SERVER_URL', 'http://localhost:1234')
        monkeypatch.setenv('BLUEBUBBLES_PASSWORD', 'secret')
        monkeypatch.setenv('BLUEBUBBLES_WEBHOOK_PORT', '9999')
        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.BLUEBUBBLES in config.platforms
        bc = config.platforms[Platform.BLUEBUBBLES]
        assert bc.enabled is True
        assert bc.extra['server_url'] == 'http://localhost:1234'
        assert bc.extra['password'] == 'secret'
        assert bc.extra['webhook_port'] == 9999

    def test_connected_platforms_includes_bluebubbles(self, monkeypatch):
        monkeypatch.setenv('BLUEBUBBLES_SERVER_URL', 'http://localhost:1234')
        monkeypatch.setenv('BLUEBUBBLES_PASSWORD', 'secret')
        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.BLUEBUBBLES in config.get_connected_platforms()


class TestBlueBubblesHelpers:
    def test_check_requirements(self, monkeypatch):
        monkeypatch.setenv('BLUEBUBBLES_SERVER_URL', 'http://localhost:1234')
        monkeypatch.setenv('BLUEBUBBLES_PASSWORD', 'secret')
        from gateway.platforms.bluebubbles import check_bluebubbles_requirements
        assert check_bluebubbles_requirements() is True

    def test_format_message_strips_markdown(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message('**Hello** `world`') == 'Hello world'

    def test_init_normalizes_webhook_path(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_path='bluebubbles-webhook')
        assert adapter.webhook_path == '/bluebubbles-webhook'

    def test_webhook_prefers_chat_guid_over_message_guid(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            'guid': 'MESSAGE-GUID',
            'chatGuid': 'iMessage;-;user@example.com',
            'chatIdentifier': 'user@example.com',
        }
        record = adapter._extract_payload_record(payload) or {}
        chat_guid = adapter._value(
            record.get('chatGuid'),
            payload.get('chatGuid'),
            record.get('chat_guid'),
            payload.get('chat_guid'),
            payload.get('guid'),
        )
        assert chat_guid == 'iMessage;-;user@example.com'

    def test_webhook_can_fall_back_to_sender_when_chat_fields_missing(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            'data': {
                'guid': 'MESSAGE-GUID',
                'text': 'hello',
                'handle': {'address': 'user@example.com'},
                'isFromMe': False,
            }
        }
        record = adapter._extract_payload_record(payload) or {}
        chat_guid = adapter._value(
            record.get('chatGuid'),
            payload.get('chatGuid'),
            record.get('chat_guid'),
            payload.get('chat_guid'),
            payload.get('guid'),
        )
        chat_identifier = adapter._value(
            record.get('chatIdentifier'),
            record.get('identifier'),
            payload.get('chatIdentifier'),
            payload.get('identifier'),
        )
        sender = adapter._value(
            record.get('handle', {}).get('address') if isinstance(record.get('handle'), dict) else None,
            record.get('sender'),
            record.get('from'),
            record.get('address'),
        ) or chat_identifier or chat_guid
        if not (chat_guid or chat_identifier) and sender:
            chat_identifier = sender
        assert chat_identifier == 'user@example.com'

    def test_extract_payload_record_accepts_list_data(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            'type': 'new-message',
            'data': [
                {
                    'text': 'hello',
                    'chatGuid': 'iMessage;-;user@example.com',
                    'chatIdentifier': 'user@example.com',
                }
            ],
        }
        record = adapter._extract_payload_record(payload)
        assert record == payload['data'][0]
