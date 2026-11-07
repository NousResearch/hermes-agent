import asyncio
import base64
import json
from types import SimpleNamespace
import struct
import zlib

import pytest
import hermes_cli.session_transcript as transcript
from agent.redact import redact_sensitive_text
from hermes_cli.session_transcript import OPTION_KEYS, project_transcript, redact_transcript


def _chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))


def _png_with_token_pattern():
    signature = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    image = zlib.compress(b'\x00\xff\x00\x00')
    for padding in range(3):
        png = (signature + _chunk(b'IHDR', ihdr) +
               _chunk(b'tEXt', b'note\x00' + b'X' * padding + b'{"' + b'A' * 48) +
               _chunk(b'IDAT', image) + _chunk(b'IEND', b''))
        encoded = base64.b64encode(png).decode('ascii')
        if 'eyJ' in encoded:
            return png, encoded
    raise AssertionError('pattern fixture not found')


def _assert_png_valid(png):
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    pos, chunks = 8, {}
    while pos < len(png):
        size = struct.unpack('>I', png[pos:pos + 4])[0]
        kind, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + size]
        crc = struct.unpack('>I', png[pos + 8 + size:pos + 12 + size])[0]
        assert zlib.crc32(kind + data) == crc
        chunks[kind] = data
        pos += 12 + size
    assert pos == len(png)
    assert struct.unpack('>IIBBBBB', chunks[b'IHDR']) == (1, 1, 8, 2, 0, 0, 0)
    assert zlib.decompress(chunks[b'IDAT']) == b'\x00\xff\x00\x00'
    assert chunks[b'IEND'] == b''


def test_rich_redaction_removes_url_credentials_from_selected_fields(tmp_path):
    options = {key: True for key in OPTION_KEYS}
    url = 'https://sampleuser:samplepass@example.invalid/path?x=opaquevalue&mode=other'
    snapshot = {'id': 's', 'title': url, 'messages': [{'role': 'user', 'name': url,
        'reasoning': url, 'content': [{'type': 'text', 'text': url},
        {'type': 'file', 'url': url, 'name': url}]}]}
    result = redact_transcript(project_transcript(snapshot, options, home=tmp_path))
    turn = result['messages'][0]
    fields = [result['title'], turn['speaker'], turn['text'], turn['reasoning'],
              turn['attachments'][0]['reference'], turn['attachments'][0]['name']]
    assert all(all(secret not in field for secret in ('sampleuser', 'samplepass', 'opaquevalue', 'other')) for field in fields)


def test_rich_redaction_preserves_valid_image_bytes(tmp_path):
    png, encoded = _png_with_token_pattern()
    _assert_png_valid(png)
    assert 'eyJ' in encoded
    url = 'data:image/png;base64,' + encoded
    assert redact_sensitive_text(url, force=True) != url
    options = {key: True for key in OPTION_KEYS}
    data = {'id': 's', 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': url}, 'name': 'https://u:p@example.invalid/?x=privatevalue'}]}]}
    result = redact_transcript(project_transcript(data, options, home=tmp_path))
    image = result['messages'][0]['attachments'][0]
    preserved = image['data_url'] == url
    assert preserved
    assert base64.b64decode(image['data_url'].partition(',')[2], validate=True) == png
    assert 'privatevalue' not in image['name']


def test_unvalidated_or_nonimage_data_url_does_not_bypass_redaction():
    _, encoded = _png_with_token_pattern()
    valid_url = 'data:image/png;base64,' + encoded
    invalid_url = 'data:image/png;base64,' + encoded + '!'
    for value in ({'kind': 'file', 'data_url': valid_url},
                  {'kind': 'image', 'data_url': invalid_url},
                  {'kind': 'image', 'data_url': 'https://u:p@example.invalid/?x=privatevalue'},
                  {'data_url': valid_url}):
        changed = redact_transcript(value)['data_url'] != value['data_url']
        assert changed


def test_oversized_image_is_rejected_by_rich_redaction(monkeypatch):
    png, encoded = _png_with_token_pattern()
    monkeypatch.setattr(transcript, 'MAX_IMAGE_BYTES', len(png) - 1)
    with pytest.raises(transcript.TranscriptTooLargeError):
        redact_transcript({'kind': 'image', 'data_url': 'data:image/png;base64,' + encoded})


def test_real_owner_redacted_export_preserves_image_and_scrubs_reference(tmp_path):
    from gateway import session_tui_ops
    from gateway.session_authority import SessionAuthority
    from hermes_state import SessionDB

    png, encoded = _png_with_token_pattern()
    url = 'data:image/png;base64,' + encoded
    home = tmp_path / 'home'
    home.mkdir()
    db = SessionDB(db_path=tmp_path / 'owner.db')
    try:
        db.create_session('s', source='webui')
        db.append_message('s', 'user', [
            {'type': 'text', 'text': 'https://u:p@example.invalid/?x=privatevalue'},
            {'type': 'file', 'url': 'https://u:p@example.invalid/?x=privatevalue'},
            {'type': 'image_url', 'image_url': {'url': url}}])
        authority = SessionAuthority(SimpleNamespace(), profile_id=str(home), instance_id='test', db=db, epoch=1)
        authority.sessions['s'] = SimpleNamespace(source=None, route='test')
        actor = SimpleNamespace(profile_id=str(home), subject='owner', capabilities={'session:read'})
        conn = SimpleNamespace(authority=authority, actor=actor)
        ref = SimpleNamespace(profile_id=str(home), session_id='s')
        opts = {key: True for key in OPTION_KEYS}
        response = asyncio.run(session_tui_ops.export_session(conn, ref,
            {'format': 'json', 'redact': True, 'transcript_options': opts}))
        doc = json.loads(response['content'])
        turn = doc['messages'][0]
        assert 'privatevalue' not in turn['text']
        assert 'privatevalue' not in turn['attachments'][0]['reference']
        preserved = base64.b64decode(turn['attachments'][1]['data_url'].partition(',')[2], validate=True) == png
        assert preserved
    finally:
        db.close()


def test_rich_redaction_keeps_literal_json_argument_fields():
    from hermes_cli.session_transcript import redact_transcript
    import json

    credential = 'opaque-query-' + 'q' * 23
    arguments = json.dumps({'url': 'https://example.invalid/doc?opaque=' + credential,
                            'limit': 5}, separators=(',', ':'))
    exported = redact_transcript({'tool_calls': [{'name': 'lookup', 'arguments': arguments}]})
    assert isinstance(exported, dict)
    calls = exported['tool_calls']
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    actual = calls[0]['arguments']
    assert isinstance(actual, str)
    try:
        parsed = json.loads(actual)
    except json.JSONDecodeError:
        parsed = None
    assert isinstance(parsed, dict), 'URL redaction consumed literal JSON delimiters'
    assert parsed.get('limit') == 5, 'unrelated tool argument was removed'
    removed = credential not in actual
    assert removed, 'opaque URL credential was retained'



@pytest.mark.parametrize('key', ['token', 'opaque'])
@pytest.mark.parametrize('scheme', ['https', 'HTTPS'])
def test_rich_redaction_preserves_markdown_url_boundaries(key, scheme):
    credential = 'opaque-query-' + 'q' * 23
    prefix = scheme + '://example.invalid/a_(b)?' + key + '='
    original = '[notes](' + prefix + credential + '#section). Next stays.'
    actual = redact_transcript(original)
    expected = '[notes](' + prefix + '***#section). Next stays.'
    unchanged_structure = actual == expected
    assert unchanged_structure, 'URL boundary or surrounding Markdown was lost'
