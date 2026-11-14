import json

import pytest

from hermes_cli.session_transcript import redact_transcript


def test_escaped_json_url_is_redacted_without_losing_siblings():
    credential = 'opaque-' + 'x' * 23
    source = json.dumps({'query': 'https://example.invalid/query?opaque=' + credential, 'limit': 5}).replace('/', r'\/')
    actual = redact_transcript(source)
    removed = credential not in actual
    assert removed, 'URL credential leaked'
    assert json.loads(actual)['limit'] == 5


def test_markdown_link_boundary_preserves_immediate_suffix():
    credential = 'opaque-' + 'x' * 23
    url = 'https://example.invalid/a_(b)?opaque=' + credential
    actual = redact_transcript('[docs](' + url + '),next')
    expected = '[docs](https://example.invalid/a_(b)?opaque=***),next'
    correct = actual == expected
    assert correct, 'link boundary or suffix was lost'


@pytest.mark.parametrize('escape', ['slash', 'unicode_scheme', 'unicode_all', 'plain'])
def test_json_string_escapes_and_duplicates(escape):
    credential = 'opaque-' + 'x' * 23
    url = 'HTTPS://user:pass@example.invalid/path?opaque=' + credential
    encoded = json.dumps(url)
    if escape == 'slash':
        encoded = encoded.replace('/', r'\/')
    elif escape == 'unicode_scheme':
        encoded = encoded.replace('H', r'\u0048').replace(':', r'\u003a')
    elif escape == 'unicode_all':
        encoded = '"' + ''.join(f'\\u{ord(c):04x}' for c in url) + '"'
    source = '{ "same": 1, "same": 2, "items": [' + encoded + ', "a\\\\b\\\"c"], "limit": 5 }'
    actual = redact_transcript(source)
    safe = credential not in actual and 'user:pass' not in actual
    assert safe
    assert actual.count('"same"') == 2
    parsed = json.loads(actual)
    assert parsed['items'][1] == 'a\\b"c'
    assert parsed['limit'] == 5


def test_json_unchanged_layout_and_key_policy():
    plain = ' { "same" : 1, "same":2, "path" : "a\\\\b" } '
    assert redact_transcript(plain) == plain
    secret = 'opaque-' + 'x' * 23
    actual = redact_transcript(json.dumps({'password': secret, 'normal': 'unchanged'}))
    parsed = json.loads(actual)
    assert parsed['password'] != secret
    assert parsed['normal'] == 'unchanged'


@pytest.mark.parametrize('suffix', [',next', '.next', ')next', '](after)', ' more'])
def test_markdown_suffixes_and_multiple_links(suffix):
    secret = 'opaque-' + 'x' * 23
    text = '[x](https://example.invalid/a_(b_(c))?x=' + secret + ')' + suffix
    result = redact_transcript(text)
    correct = result == '[x](https://example.invalid/a_(b_(c))?x=***)' + suffix
    assert correct


@pytest.mark.parametrize('prefix', ['{', '[' * 65, '{' * 65])
def test_non_json_text_remains_literal_when_no_secret(prefix):
    text = prefix + 'unfinished: literal text and backslash \\ here'
    assert redact_transcript(text) == text


def test_deep_valid_json_still_fails_closed():
    text = '[' * 65 + '"private"' + ']' * 65
    result = redact_transcript(text)
    assert isinstance(result, str)
    assert json.loads(result) == '[redaction unavailable]'


def test_json_query_escaped_quote_does_not_leak_credential_tail():
    tail = 'fragment' + 'z' * 20
    source = json.dumps({'url': 'https://example.invalid/?x=abc"' + tail})
    result = json.loads(redact_transcript(source))
    removed = tail not in result['url']
    assert removed, 'decoded quote allowed query tail to escape redaction'


@pytest.mark.parametrize('separator', ['"', "'", '<', '>', '`'])
@pytest.mark.parametrize('component', ['query', 'userinfo'])
def test_decoded_json_uri_credentials_are_atomic(component, separator):
    tail = 'fragment' + 'z' * 20
    credential = 'abc' + separator + tail
    if component == 'query':
        url = 'HTTPS://example.invalid/path?opaque=' + credential
    else:
        url = 'HTTPS://person:' + credential + '@example.invalid/path'
    redacted = redact_transcript(json.dumps({'url': url, 'limit': 5}))
    assert isinstance(redacted, str)
    actual = json.loads(redacted)
    removed = tail not in actual['url']
    assert removed, 'decoded URI credential tail escaped redaction'
    assert actual['limit'] == 5


@pytest.mark.parametrize('punctuation', ['!', ')', ']', '}', '.', ';', '?'])
def test_atomic_json_query_does_not_restore_credential_punctuation(punctuation):
    source = json.dumps({'url': 'https://example.invalid/?opaque=' + punctuation, 'normal': 'kept'})
    redacted = redact_transcript(source)
    assert isinstance(redacted, str)
    parsed = json.loads(redacted)
    # Semicolon is a query separator in this export policy, not part of its value.
    expected = 'https://example.invalid/?opaque=***' + (';' if punctuation == ';' else '')
    complete = parsed['url'] == expected
    assert complete, 'standalone URI query value was mistaken for prose punctuation'
    assert parsed['normal'] == 'kept'


@pytest.mark.parametrize('separator', ['"', '\\', '\n', '\t'])
def test_sensitive_json_escaped_value_retains_siblings_and_layout(separator):
    credential = 'abc' + separator + 'tail'
    value = json.dumps(credential)
    source = ' { "password" : ' + value + ', "same":1, "same":2, "limit" : 5 } '
    actual = redact_transcript(source)
    assert isinstance(actual, str)
    parsed = json.loads(actual)
    assert isinstance(parsed, dict), 'sensitive value caused whole-document content loss'
    assert parsed['password'] == '***'
    assert parsed['limit'] == 5
    assert actual == source.replace(value, '"***"', 1)


def test_deep_json_fails_closed():
    secret = 'opaque-' + 'x' * 23
    text = '[' * 1100 + json.dumps(secret) + ']' * 1100
    result = redact_transcript(text)
    removed = secret not in result
    assert removed
    assert json.loads(result) == '[redaction unavailable]'
