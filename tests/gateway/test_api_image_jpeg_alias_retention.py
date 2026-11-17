"""JPEG aliases retain pixels across restart without changing legacy admission identity."""
import base64
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agent.session_persistence import _durable_content
from agent.turn_context import build_api_messages
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _normalize_multimodal_content
from gateway.session import SessionStore
from gateway.session_api import bind_api_session, restore_api_session
from gateway.session_api_media import commit_api_images, compact_settled_api_payloads, rehydrate_api_images
from gateway.session_api_turn import (
    _SETTING_KEYS, admit_api_turn, prepare_api_execution, recover_authority_api_turns,
)
from gateway.session_authority import SessionAuthority
from gateway.session_ingress_media import native_inputs_root, release_admission_media
from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError, admit_session_input, begin_runtime_epoch, claim_session_input,
    get_session_admission, settle_session_input,
)

def _jpeg_bytes():
    with BytesIO() as buffer:
        Image.new('RGB', (1, 1), 'red').save(buffer, format='JPEG')
        return buffer.getvalue()


JPEG = _jpeg_bytes()
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=')


class _Agent:
    api_mode = 'chat_completions'
    provider, model = 'custom', 'test-vision'
    ephemeral_system_prompt = None
    _compression_warning = None
    _current_turn_timestamp = 10_000.0

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


@contextmanager
def _runtime(home, instance):
    db = SessionDB(home / 'state.db')
    store = SessionStore(config=GatewayConfig(), sessions_dir=home / 'sessions')
    store._db = db
    runner = SimpleNamespace(_draining=False, session_store=store)
    owner = SessionAuthority(runner, profile_id='default', instance_id=instance, db=db,
                             epoch=begin_runtime_epoch(db, instance_id=instance))
    runner.session_authority = owner
    api = APIServerAdapter(PlatformConfig(enabled=True))
    api.gateway_runner, api._session_db = runner, db
    runner._adapter_for_source = lambda source: api
    try:
        yield owner, api
    finally:
        api._response_store.close()
        api._run_idempotency_store.close()
        db.close()


def _image(mime, data, detail='high'):
    return {'type': 'image_url', 'image_url': {
        'url': f'data:{mime};base64,' + base64.b64encode(data).decode(), 'detail': detail}}


def _settle(owner, row):
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id=row['target_session_id'])
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
                         generation=started['generation'], outcome='completed')


def _digest(owner, admission_id):
    from hermes_state_terminal import terminal_admission
    with owner.db._read_ctx() as conn:
        row = conn.execute('SELECT payload_digest FROM session_admissions WHERE admission_id=?',
                           (admission_id,)).fetchone()
        return row[0] if row else terminal_admission(conn, admission_id)['payload_digest']


@pytest.mark.parametrize('mime', ['image/jpg', 'IMAGE/JPG', 'image/jpeg'])
@pytest.mark.parametrize('startup_compaction', [False, True], ids=['settlement', 'startup'])
def test_inline_jpeg_retention_and_restart_replay(tmp_path, monkeypatch, mime, startup_compaction):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(json.dumps({'agent': {'image_input_mode': 'native'}}))
    parts = _normalize_multimodal_content([
        {'type': 'text', 'text': 'before'}, _image(mime, JPEG),
        {'type': 'text', 'text': 'between'}, _image('image/png', PNG, 'auto'),
        _image(mime, JPEG, 'low'), {'type': 'text', 'text': 'after'},
    ])
    original = deepcopy(parts)
    kwargs = dict(session_id='images', request_id='responses:images:one',
                  user_message=parts, conversation_history=[])
    with _runtime(tmp_path, 'first') as (owner, api):
        _, ref, row = admit_api_turn(api, **kwargs)
        media = row['payload']['api_turn_v1']['media']
        assert len(media) == 3
        assert [r['sha256'] for r in media] == [hashlib.sha256(b).hexdigest() for b in (JPEG, PNG, JPEG)]
        for reference, data in zip(media, (JPEG, PNG, JPEG)):
            path = Path(reference['path'])
            assert path.read_bytes() == data and path.is_relative_to(tmp_path)
        assert Path(media[0]['path']).suffix == '.jpg'
        live = prepare_api_execution(owner, ref, row['payload'])['content']
        owner.db.append_message('images', 'user', _durable_content(live))
        owner.db.append_message('images', 'assistant', 'seen')
        # Startup must leave an executable row's full request intact.
        assert compact_settled_api_payloads(owner.db) == 0
        _settle(owner, row)
        if not startup_compaction:
            release_admission_media(owner.db, row['admission_id'])
        digest = _digest(owner, row['admission_id'])
    # All original SQLite handles and adapter stores are closed before reopening.
    with _runtime(tmp_path, 'second') as (owner, api):
        recover_authority_api_turns(api, owner)
        saved = get_session_admission(owner.db, admission_id=row['admission_id'])
        assert _digest(owner, saved['admission_id']) == digest
        compacted = saved['payload']['text']
        assert [p['image_url']['url'] for p in compacted if p['type'] == 'image_url'] == [
            'hermes-retained:' + r['sha256'] for r in media]
        restored = rehydrate_api_images(compacted, media)
        assert restored[1] == _image('image/jpeg', JPEG)
        assert restored[3] == _image('image/png', PNG, 'auto')
        assert restored[4] == _image('image/jpeg', JPEG, 'low')
        retry = admit_api_turn(api, **kwargs)[2]
        assert retry['admission_id'] == row['admission_id'] and _digest(owner, retry['admission_id']) == digest
        history = owner.db.get_messages_as_conversation('images')
        history.append({'role': 'user', 'content': 'follow up'})
        frozen = deepcopy(history)
        request, _ = build_api_messages(_Agent(), history, current_turn_user_idx=len(history) - 1,
            ext_prefetch_cache='', plugin_user_context='', moa_config=None, active_system_prompt='')
        replay = next(m['content'] for m in request if m['role'] == 'user')
        assert [p['type'] for p in replay] == [p['type'] for p in original]
        assert replay[1] == restored[1] and replay[3:5] == restored[3:5]
        assert replay[0]['text'].startswith('before') and replay[2]['text'] == 'between'
        assert replay[-1]['text'] == 'after'
        assert history == frozen and parts == original
        assert owner.db.get_messages_as_conversation('images') == frozen[:-1]
        # Previously admitted legacy batches may replay, but fresh batches still obey both caps.
        for count, limit in ((11, 1_000_000), (7, 4096)):
            with monkeypatch.context() as bounded:
                bounded.setattr('gateway.platforms.base.get_inbound_media_max_bytes', lambda limit=limit: limit)
                with pytest.raises(RuntimeStoreError, match='invalid_params'):
                    admit_api_turn(api, session_id='rejected', request_id=f'new-{count}',
                        user_message=[_image(mime, JPEG)] * count, conversation_history=[])
        assert sorted(native_inputs_root().glob('*/*')) == sorted({Path(r['path']) for r in media})


@pytest.mark.parametrize('with_png', [False, True], ids=['alias-only', 'alias-before-png'])
@pytest.mark.parametrize('status', ['queued', 'terminal', 'retired'])
@pytest.mark.parametrize('batch', ['single', 'over-count', 'over-bytes'])
def test_legacy_alias_execution_and_exact_retry_survive_restart(tmp_path, monkeypatch, with_png, status, batch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    count, limit = {'single': (1, 1_000_000), 'over-count': (11, 1_000_000), 'over-bytes': (7, 4096)}[batch]
    monkeypatch.setattr('gateway.platforms.base.get_inbound_media_max_bytes', lambda: limit)
    parts = [{'type': 'text', 'text': 'look'}, *[_image('image/jpg', JPEG)] * count]
    if with_png:
        parts.append(_image('image/png', PNG, 'low'))
    kwargs = dict(session_id='legacy', request_id='legacy-one', user_message=parts, conversation_history=[])
    with _runtime(tmp_path, 'first') as (owner, _api):
        bind_api_session(owner, 'legacy')
        # Exact historical capture: image/jpg was skipped, but canonical siblings were retained.
        media = commit_api_images(parts[count + 1:])
        payload = {'text': parts, 'api_turn_v1': {
            'history': [], 'settings': dict.fromkeys(_SETTING_KEYS), 'media': media}}
        row = admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id='legacy',
                                  request_id='legacy-one', payload=payload)
        digest = _digest(owner, row['admission_id'])
        if status != 'queued':
            _settle(owner, row)
        if status == 'retired':
            owner.db.delete_session('legacy')
    with _runtime(tmp_path, 'second') as (owner, api):
        saved = get_session_admission(owner.db, admission_id=row['admission_id'])
        if status != 'retired':
            ref = restore_api_session(owner, 'legacy')
            live = prepare_api_execution(owner, ref, saved['payload'])['content']
            text = live[0]['text']
            assert f"[Image attached: {parts[1]['image_url']['url']}]" in text
            assert text.count('[Image attached at:') == int(with_png)
            if with_png:
                assert text.endswith(f"[Image attached at: {media[0]['path']}]")
            compact_settled_api_payloads(owner.db)
            after = get_session_admission(owner.db, admission_id=row['admission_id'])
            assert after['payload']['text'][1] == parts[1]
            if with_png and status == 'terminal':
                assert after['payload']['text'][count + 1]['image_url']['url'] == 'hermes-retained:' + media[0]['sha256']
        retry = admit_api_turn(api, **kwargs)[2]
        assert retry['admission_id'] == row['admission_id']
        assert _digest(owner, retry['admission_id']) == digest and retry['status'] == saved['status']
        variants = [
            {'user_message': [{'type': 'text', 'text': 'changed'}, *parts[1:]]},
            {'user_message': [parts[0], _image('image/jpg', JPEG + b'changed'), *parts[2:]]},
            {'requested_model': 'changed'}, {'conversation_history': [{'role': 'user', 'content': 'changed'}]},
            {'session_id': 'other'},
        ]
        for change in variants:
            with pytest.raises(RuntimeStoreError, match='admission_conflict'):
                admit_api_turn(api, **{**kwargs, **change})
        _assert_retry_fences(owner, api, kwargs, monkeypatch)
        # Legacy retries leave no newly retained alias bytes, including retired receipts.
        held = [] if status == 'retired' else [Path(r['path']) for r in media]
        assert sorted(native_inputs_root().glob('*/*')) == sorted(held)
        if status == 'retired':
            assert owner.db.get_session('legacy') is None


def _assert_retry_fences(owner, api, kwargs, monkeypatch):
    with monkeypatch.context() as draining:
        draining.setattr(owner.runner, '_draining', True)
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            admit_api_turn(api, **kwargs)
    begin_runtime_epoch(owner.db, instance_id='replacement')
    with pytest.raises(RuntimeStoreError, match='stale_epoch'):
        admit_api_turn(api, **kwargs)
