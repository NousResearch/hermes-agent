"""Malformed quoted secret values must not bypass display masking."""
import pytest

from agent.streaming_redact import StreamingSecretSanitizer, sanitize_terminal_secret_text


@pytest.mark.parametrize("key", ["password", "api_key", "token"])
@pytest.mark.parametrize("escape", ['\\q', '\\x20', '\\uZZZZ', '\\\n', '\\\r\n'])
def test_malformed_escaped_secret_keeps_the_recognized_value_opaque(key, escape):
    payload = '{"' + key + '":"head' + escape + r'\"opaqueSecretTail123456"}'
    expected = '{"' + key + '": "***"}'
    assert sanitize_terminal_secret_text(payload) == expected
    state = StreamingSecretSanitizer()
    assert "".join(state.feed(char) for char in payload) + state.flush() == expected


@pytest.mark.parametrize("payload", [
    '{"token":"cpu\\q"}',
    '{"password":"os.getenv(\'X\')\\q"}',
    '{"token":"CPU"}',
    '{"password":"os.getenv(\'X\')"}',
])
def test_malformed_json_reuses_existing_assignment_ambiguity_gate(payload):
    assert sanitize_terminal_secret_text(payload) == payload
