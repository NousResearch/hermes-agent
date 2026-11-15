"""Portable, synthetic SQLite-to-canonical projection option matrix."""
import base64
import itertools
import json
import struct
import zlib
from pathlib import Path
import tempfile

import pytest

from hermes_cli.session_transcript import OPTION_KEYS, project_transcript, redact_transcript
from hermes_state import SessionDB

BASE_PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=='
QUERY_CREDENTIAL = 'opaque-query-' + 'q' * 23
USER_CREDENTIAL = 'opaque-user-' + 'u' * 23
QUERY_URL = 'https://example.invalid/source?token=' + QUERY_CREDENTIAL
USERINFO_URL = 'https://sample:' + USER_CREDENTIAL + '@example.invalid/private'


def redaction_trap_png():
    raw = base64.b64decode(BASE_PNG.partition(',')[2], validate=True)
    end = 8 + 12 + int.from_bytes(raw[8:12], 'big')
    # A private ancillary PNG chunk can hold arbitrary bytes without changing pixels.
    marker = 'eyJ' + 'A' * 13
    payload = b'X' * (-(end + 8) % 3) + base64.b64decode(marker, validate=True)
    kind = b'raNd'
    chunk = struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    encoded = base64.b64encode(raw[:end] + chunk + raw[end:]).decode('ascii')
    has_pattern = marker in encoded
    assert has_pattern, 'fixture must contain the redactor-sensitive base64 pattern'
    return 'data:image/png;base64,' + encoded


PNG = redaction_trap_png()

SECRET = 'sk-' + 'a' * 48
LINK = 'https://example.invalid/synthetic-link'
KEYS = sorted(OPTION_KEYS)


def synthetic_snapshot(home):
    home.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=home / 'state.db')
    try:
        db.create_session('synthetic-matrix', source='webui')
        db.append_message('synthetic-matrix', 'system', 'SYSTEM_PRIVATE')
        db.append_message('synthetic-matrix', 'user', [
            {'type': 'text', 'text': 'USER_雪\nsecond line ' + LINK + ' ' + SECRET + '\nQuery: ' + QUERY_URL + '\nUserinfo: ' + USERINFO_URL},
            {'type': 'image_url', 'name': 'IMAGE_VALID', 'image_url': {'url': PNG}},
            {'type': 'image_url', 'name': 'IMAGE_UNAVAILABLE', 'image_url': {'url': QUERY_URL}},
            {'type': 'video_url', 'name': 'VIDEO_MARKER', 'video_url': {'url': 'https://example.invalid/clip.mp4?token=' + QUERY_CREDENTIAL}},
            {'type': 'input_file', 'name': 'FILE_MARKER', 'file': {'filename': 'synthetic.txt', 'url': QUERY_URL}},
        ], timestamp=1700000001)
        db.append_message('synthetic-matrix', 'assistant', 'ASSISTANT_MARKER ' + LINK + ' [a](https://one.example),[b](https://two.example) [boundary](' + QUERY_URL + '),next', reasoning='REASONING_MARKER',
                          tool_calls=[{'id': 'call1', 'type': 'function', 'function': {'name': 'TOOL_CALL_MARKER', 'arguments': json.dumps({'query': QUERY_URL, 'limit': 5, 'literal': '[brackets](stay literal)', 'quoted_url': QUERY_URL + '"' + QUERY_CREDENTIAL, 'userinfo_url': 'https://sample:' + QUERY_CREDENTIAL + '"tail@example.invalid/path', 'punctuation_url': 'https://example.invalid/?opaque=!', 'password': 'abc"tail', 'sensitive_sibling': 'SIBLING_PRESERVED'}, separators=(',', ':')).replace('/', r'\/')}}], timestamp=1700000002)
        db.append_message('synthetic-matrix', 'tool', 'TOOL_RESULT_MARKER', tool_name='TOOL_NAME_MARKER', tool_call_id='call1', timestamp=1700000003)
        db.append_message('synthetic-matrix', 'assistant', 'ASIDE_MARKER', display_kind='aside', timestamp=1700000004)
        db.append_message('synthetic-matrix', 'assistant', 'NOTICE_MARKER', display_kind='notice', timestamp=1700000005)
        snapshot = db.export_session('synthetic-matrix', include_compacted=True)
        snapshot['messages'] = db.get_messages('synthetic-matrix', include_compacted=True, include_ancestors=True)
        return snapshot
    finally:
        db.close()


def expected_markers(options):
    return {'USER_雪': True, 'ASSISTANT_MARKER': True, 'NOTICE_MARKER': True,
            'SYSTEM_PRIVATE': False, 'TOOL_CALL_MARKER': options['include_tools'],
            'TOOL_RESULT_MARKER': options['include_tools'], 'REASONING_MARKER': options['include_reasoning'],
            'IMAGE_VALID': options['include_images'], 'IMAGE_UNAVAILABLE': options['include_images'],
            'VIDEO_MARKER': options['include_videos'], 'FILE_MARKER': options['include_files'],
            'ASIDE_MARKER': options['include_asides']}


def projection(snapshot, home, options, redact):
    doc = project_transcript(snapshot, options, home=home)
    return redact_transcript(doc) if redact else doc


def verify(doc, options, redact):
    serialized = json.dumps(doc, ensure_ascii=False)
    for marker, expected in expected_markers(options).items():
        assert (marker in serialized) == expected, marker
    assert (SECRET in serialized) != redact
    for credential in (QUERY_CREDENTIAL, USER_CREDENTIAL):
        retained = credential in serialized
        assert retained is not redact, 'URL credential redaction mismatch'
    assert LINK in serialized
    if options['include_tools']:
        call = next(row for row in doc['messages'] if row.get('tool_calls'))['tool_calls'][0]
        parsed = json.loads(call['arguments'])
        assert parsed['limit'] == 5 and parsed['literal'] == '[brackets](stay literal)'
        sensitive_preserved = parsed['password'] == ('***' if redact else 'abc"tail')
        assert sensitive_preserved, 'sensitive JSON value was not safely preserved/masked'
        assert parsed['sensitive_sibling'] == 'SIBLING_PRESERVED'
        retained = QUERY_CREDENTIAL in call['arguments']
        assert retained is not redact, 'literal argument URL redaction mismatch'
    assert doc['options'] == options
    assert len(doc['messages']) == 3 + int(options['include_tools']) + int(options['include_asides'])
    assert next(row for row in doc['messages'] if row['text'] == 'NOTICE_MARKER')['kind'] == 'notice'
    if options['include_asides']:
        assert next(row for row in doc['messages'] if row['text'] == 'ASIDE_MARKER')['kind'] == 'aside'
    assert all('timestamp' in row for row in doc['messages'])
    attachments = [a for row in doc['messages'] for a in row.get('attachments', [])]
    assert len(attachments) == 2 * int(options['include_images']) + int(options['include_videos']) + int(options['include_files'])
    if options['include_images']:
        assert any(a.get('data_url') == PNG for a in attachments)
        assert any(a.get('name') == 'IMAGE_UNAVAILABLE' and a.get('unavailable') for a in attachments)


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    home = tmp_path_factory.mktemp('synthetic-matrix')
    return synthetic_snapshot(home), home


@pytest.mark.parametrize('bits', list(itertools.product((False, True), repeat=8)))
@pytest.mark.parametrize('redact', [False, True])
def test_sqlite_projection_matrix(source, bits, redact):
    snapshot, home = source
    options = dict(zip(KEYS, bits))
    verify(projection(snapshot, home, options, redact), options, redact)


def test_excluding_saved_aside(source):
    snapshot, home = source
    options = dict.fromkeys(KEYS, False)
    assert 'ASIDE_MARKER' not in json.dumps(projection(snapshot, home, options, False))


def test_absent_timestamps_are_not_invented(tmp_path):
    options = dict.fromkeys(KEYS, False)
    doc = project_transcript({'id': 'synthetic', 'messages': [{'role': 'user', 'content': 'hello'}]}, options, home=tmp_path)
    assert 'created_at' not in doc and 'timestamp' not in doc['messages'][0]


def test_z_timestamps_are_normalized(tmp_path):
    options = dict.fromkeys(KEYS, False)
    doc = project_transcript({'id': 'synthetic', 'started_at': '2023-11-14T22:13:20Z',
                              'messages': [{'role': 'user', 'content': 'hello', 'timestamp': '2023-11-14T22:13:21Z'}]},
                             options, home=tmp_path)
    assert doc['created_at'] == '2023-11-14T22:13:20+00:00'
    assert doc['messages'][0]['timestamp'] == '2023-11-14T22:13:21+00:00'


def write_matrix(path):
    with tempfile.TemporaryDirectory() as temporary:
        home = Path(temporary) / 'home'
        snapshot = synthetic_snapshot(home)
        with Path(path).open('w', encoding='utf-8', newline='\n') as output:
            for bits, redact in itertools.product(itertools.product((False, True), repeat=8), (False, True)):
                options = dict(zip(KEYS, bits))
                doc = projection(snapshot, home, options, redact)
                verify(doc, options, redact)
                expected = expected_markers(options)
                serialized = json.dumps(doc, ensure_ascii=False)
                observed = {marker: marker in serialized for marker in expected}
                assert expected == observed
                output.write(json.dumps({'options': options, 'redact': redact,
                                         'expected_source_markers': expected,
                                         'observed_source_markers': observed,
                                         'dto': doc}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    import sys
    write_matrix(sys.argv[1])
