"""API data: images are validated as a batch before any byte is staged or committed."""
import base64
import os
from pathlib import Path

import pytest

from gateway.platforms import base
from gateway.session_api_turn import admit_api_turn
from gateway.session_ingress_media import _media_root
from hermes_state_runtime import RuntimeStoreError

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=')


def _part(data, mime='image/png'):
    return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,' + base64.b64encode(data).decode()}}


def _files(root):
    return [p for p in Path(root).rglob('*') if p.is_file()] if Path(root).exists() else []


@pytest.mark.parametrize('case', ['foreign-target', 'over-total', 'too-many', 'mime-mismatch'])
def test_refused_api_image_admission_leaves_no_bytes(api, owner, monkeypatch, case):
    owner.db.create_session('tg', source='telegram')
    monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: 4096)
    sid, parts = 'api-images', [_part(PNG)]
    if case == 'foreign-target':
        sid = 'tg'
    elif case == 'over-total':
        parts = [_part(PNG + os.urandom(3000)), _part(PNG + os.urandom(3000))]
    elif case == 'too-many':
        parts = [_part(PNG + bytes([i])) for i in range(11)]
    else:
        parts = [_part(b'<html><script>alert(1)</script></html>')]
    with pytest.raises(RuntimeStoreError):
        admit_api_turn(api, session_id=sid, user_message=parts, conversation_history=[])
    assert _files(_media_root()) == []
    assert _files(base.get_image_cache_dir()) == []
