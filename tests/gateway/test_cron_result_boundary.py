"""Additional controller checks of the canonical cron-result redaction boundary."""
import sys

import pytest

from gateway.session_results import _redacted
from cron.scheduler_prompt import _archive_answer


def document(prompt, answer):
    return (f'# Cron Job: fixture\n**Prompt Characters:** {len(prompt)}\n## Prompt\n\n{prompt}\n\n'
            f'**Response Characters:** {len(answer)}\n## Response\n\n{answer}\n')


def enable_native_redaction(monkeypatch):
    import agent.redact
    monkeypatch.setenv('HERMES_REDACT_SECRETS', '1')
    monkeypatch.setattr(agent.redact, '_REDACT_ENABLED', True)
    return agent.redact.redact_sensitive_text


def test_real_redactor_roundtrip_with_nested_marker(monkeypatch):
    redact = enable_native_redaction(monkeypatch)
    prompt = 'curl -H "Authorization: Bearer $POSTPROXY_API_KEY"\n## Response\nquoted prompt\n'
    answer = 'Authorization: Bearer $POSTPROXY_API_KEY\nFinal response.'
    original = document(prompt, answer)
    assert _archive_answer(redact(original)) is None
    result = _redacted({'cron_result': [True, original, answer, None]})['cron_result']
    assert _archive_answer(result[1]) == redact(answer)
    assert result[2] == redact(answer)
    assert prompt not in result[1]


def test_noncanonical_first_element_is_not_exempt_from_redaction(monkeypatch):
    redact = enable_native_redaction(monkeypatch)
    text = 'Authorization: Bearer $POSTPROXY_API_KEY'
    assert redact(text) != text
    result = _redacted({'cron_result': [text, document('prompt', 'answer'), 'answer', None]})['cron_result']
    assert result[0] == redact(text)


@pytest.mark.parametrize('status_kind', ['string', 'integer', 'mapping', 'list', 'none'])
def test_malformed_status_uses_ordinary_recursive_redaction(monkeypatch, status_kind):
    redact = enable_native_redaction(monkeypatch)
    text = 'Authorization: Bearer ' + '$' + 'POSTPROXY_API_KEY'
    status = {'string': text, 'integer': 1, 'mapping': {'label': text},
              'list': [text], 'none': None}[status_kind]
    value = [status, document(text, 'answer'), text, None]

    def ordinary(item):
        if isinstance(item, str):
            return redact(item)
        if isinstance(item, list):
            return [ordinary(part) for part in item]
        if isinstance(item, dict):
            return {key: ordinary(part) for key, part in item.items()}
        return item

    actual = _redacted({'cron_result': value})
    assert isinstance(actual, dict)
    assert actual['cron_result'] == ordinary(value)


@pytest.mark.parametrize('length', [0, 1, 2, 3, 5])
def test_noncanonical_lengths_do_not_receive_frame_treatment(monkeypatch, length):
    redact = enable_native_redaction(monkeypatch)
    text = 'Authorization: Bearer ' + '$' + 'POSTPROXY_API_KEY'
    value = [True, document(text, text), text, None, text][:length]
    expected = [redact(part) if isinstance(part, str) else part for part in value]
    actual = _redacted({'cron_result': value})
    assert isinstance(actual, dict)
    assert actual['cron_result'] == expected


@pytest.mark.parametrize('boundary', ['prompt_response', 'header_prompt'])
def test_cross_component_secret_patterns_keep_whole_document_masking(monkeypatch, boundary):
    redact = enable_native_redaction(monkeypatch)
    marker = 'SYNTHETIC-MATERIAL-ONLY-NOT-A-REAL-KEY'
    begin = '-----BEGIN ' + 'PRIVATE KEY-----'
    end = '-----END ' + 'PRIVATE KEY-----'
    if boundary == 'prompt_response':
        prompt, answer = begin + '\n' + marker + '\n', end
        original = document(prompt, answer)
    else:
        prompt, answer = end + '\nnormal prompt', 'safe response'
        original = begin + '\n' + marker + '\n' + document(prompt, answer)
    assert _archive_answer(original) == answer
    ordinary = redact(original)
    assert marker not in ordinary
    actual = _redacted({'cron_result': [True, original, answer, None]})
    assert isinstance(actual, dict)
    result = actual['cron_result']
    assert isinstance(result, list)
    assert isinstance(result[1], str)
    assert result[1] == ordinary
    assert marker not in result[1]
    # Whole-document masking can leave a legacy-readable safe response.
    # Preserve that existing reader behavior instead of inventing a new policy.
    assert _archive_answer(result[1]) == _archive_answer(ordinary)


@pytest.mark.parametrize('field', ['Prompt', 'Response'])
def test_unconvertible_counts_preserve_ordinary_recursive_redaction(monkeypatch, field):
    redact = enable_native_redaction(monkeypatch)
    sensitive = 'Authorization: ' + 'Bearer ' + '$' + 'SYNTHETIC_KEY_REFERENCE'
    assert redact(sensitive) != sensitive
    original = document(sensitive, sensitive)
    previous_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(sys.int_info.default_max_str_digits)
        digits = '9' * (sys.get_int_max_str_digits() + 1)
        malformed = original.replace(f'**{field} Characters:** {len(sensitive)}',
                                     f'**{field} Characters:** {digits}', 1)
        actual = _redacted({'cron_result': [True, malformed, sensitive, sensitive]})
        assert actual == {'cron_result': [True, redact(malformed),
                                          redact(sensitive), redact(sensitive)]}
    finally:
        sys.set_int_max_str_digits(previous_limit)
