"""Retained native bytes always have an owner: no orphan from a rejected batch, no leftover
when equal bytes were admitted under different basenames."""
from pathlib import Path

import pytest

from gateway.session_ingress_media import _media_root, capture_native_media, release_admission_media
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, cancel_session_input

PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 8


def _staged(name, size):
    from gateway.platforms.base import get_image_cache_dir
    path = get_image_cache_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG + b'\x00' * max(0, size - len(PNG)))
    return path


def test_rejected_batch_publishes_nothing(tmp_path, monkeypatch):
    """(a) File B trips the aggregate budget after A was published: A must not stay retained
    with no admission to release it — while a file an EARLIER capture published (and so an
    earlier admission may own) is not rolled back with the rejected batch."""
    from gateway.platforms import base
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: 3 * 1024)
    from hermes_state_runtime import RuntimeStoreError
    owned = Path(capture_native_media([_staged('owned.png', 512)])[0]['path'])
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        capture_native_media([_staged('owned.png', 512), _staged('a.png', 2 * 1024), _staged('b.png', 2 * 1024)])
    assert owned.exists()
    assert [p.name for p in _media_root().iterdir()] == [owned.parent.name]


def test_equal_bytes_under_different_basenames_are_fully_released(tmp_path, monkeypatch):
    """(b) Two admissions retain the same digest as ``<digest>/a.png`` and ``<digest>/b.png``.
    Cancelling A while B is live keeps A's bytes (shared digest); after B is terminal too,
    nothing under the digest directory may remain."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='current')
    rows = []
    for name in ('a.png', 'b.png'):
        media = capture_native_media([_staged(name, 64)])
        payload = {'text': name, 'attachments_v1': {'media': media, 'media_types': ['image/png']}}
        rows.append(admit_session_input(db, epoch=epoch, principal_id='p', session_id='s',
                                        request_id=name, payload=payload))
    first, second = rows
    digest_dir = _media_root() / first['payload']['attachments_v1']['media'][0]['sha256']
    assert sorted(p.name for p in digest_dir.iterdir()) == ['a.png', 'b.png']
    cancel_session_input(db, epoch=epoch, admission_id=first['admission_id'])
    release_admission_media(db, first['admission_id'])
    assert (digest_dir / 'b.png').exists(), 'bytes a live admission still needs must survive'
    cancel_session_input(db, epoch=epoch, admission_id=second['admission_id'])
    release_admission_media(db, second['admission_id'])
    assert not digest_dir.exists(), list(digest_dir.iterdir())
    assert list(_media_root().iterdir()) == []
