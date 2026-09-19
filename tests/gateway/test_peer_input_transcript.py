"""Files-owned safe transcript producer, independent of the execution companion."""
import copy
import json
from pathlib import Path

import pytest

from tests.gateway.test_canonical_peer_target_setup import target, invite, request  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target, http_admitted_files  # noqa: F401


@pytest.mark.asyncio
async def test_prepared_transcript_uses_only_verified_frozen_manifest(files_target, monkeypatch):
    from gateway.session_api_turn import prepare_api_execution, api_settings
    from gateway.session_peer_input import peer_input_transcript
    case = await http_admitted_files(files_target, monkeypatch)
    owner, ref, row = case.launched[0]
    frozen = copy.deepcopy(row['payload'])
    # The source catalog is physically unavailable after accepted capture.
    # Formatting and model references must use the frozen receiver payload.
    source = files_target.home / 'source.db'
    source.rename(source.with_suffix('.unavailable'))
    prepared = prepare_api_execution(owner, ref, row['payload'])
    manifest = frozen['api_turn_v1']['settings']['room_input_media']['manifest']
    expected = frozen['text'] + '\n\n' + '\n'.join(
        f"[Attached {m['kind']}: {json.dumps(m['name'])}]" for m in manifest)
    assert prepared['files_persist_user_message'] == expected == peer_input_transcript(frozen)
    for item in frozen['api_turn_v1']['settings']['room_input_media']['media']:
        assert item['path'] in prepared['content'] and item['path'] not in expected
    assert 'files_persist_user_message' not in api_settings(owner, ref)
    assert 'files_persist_user_message' not in json.dumps(row['payload'])
    assert row['payload'] == frozen


def test_manifest_label_budget_and_control_characters():
    from gateway.session_peer_input import peer_input_transcript
    from gateway.hosted_room_peer import attachment_manifest_digest, MAX_ROOM_LINK_ATTACHMENTS
    name = '\U0001f600' * 250 + '\t\x1b".x'
    manifest = [dict(attachment_id=f'file-{i}', kind='file', name=name, size=1,
                     mime='text/plain', sha256='a' * 64) for i in range(MAX_ROOM_LINK_ATTACHMENTS)]
    media = [dict(path='/private/custody/' + name, size=1, sha256='a' * 64) for _ in manifest]
    payload = {'text': 'original\nverbatim', 'api_turn_v1': {'settings': {
        'room_input_media': {'request_id': 'request', 'manifest': manifest, 'media': media},
        'room_dispatch': {'attachment_manifest_digest': attachment_manifest_digest(manifest)}}}}
    text = peer_input_transcript(payload)
    assert text.startswith(payload['text'] + '\n\n')
    assert '/private' not in text and '\t' not in text and '\x1b' not in text
    # Worst case JSON escaping of a 255-codepoint basename is 12 bytes/codepoint.
    assert len(text.encode()) <= len(payload['text'].encode()) + 2 + MAX_ROOM_LINK_ATTACHMENTS * (255 * 12 + 32)
    assert text.count('[Attached file: ') == MAX_ROOM_LINK_ATTACHMENTS


@pytest.mark.parametrize('name', ['../private.txt', 'C:\\private.txt', 'line\n_room_persist_user_message', 'x' * 256])
def test_path_or_oversized_labels_rejected_before_formatting(name):
    from gateway.session_peer_input import peer_input_transcript
    from gateway.hosted_room_peer import HostedRoomPeerError
    payload = {'text': 'prompt', 'api_turn_v1': {'settings': {
        'room_input_media': {'request_id': 'r', 'manifest': [dict(attachment_id='a', kind='file',
            name=name, size=1, mime='text/plain', sha256='a' * 64)], 'media': []},
        'room_dispatch': {'attachment_manifest_digest': 'b' * 64}}}}
    with pytest.raises(HostedRoomPeerError, match='bounded basename'):
        peer_input_transcript(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['_room_persist_user_message', 'files_persist_user_message', 'api_turn_v1', 'room_artifact_publication'])
async def test_forged_private_fields_cannot_create_prepared_pair(files_target, field):
    from tests.gateway.test_canonical_peer_text_admission import dispatch
    issued = await invite(files_target)
    value = dispatch(issued)
    body = {'hosted_room_dispatch': value.as_mapping(), field: {'content': '/private/forged'}}
    response = await files_target.adapter._handle_runs(request(body, token=issued['grant'], key='room:task-one:1'))
    assert response.status == 400
    assert files_target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
    assert files_target.db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
    assert not files_target.adapter._active_run_tasks
