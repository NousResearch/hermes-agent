"""Exercise managed-image read boundaries with synthetic files, never live profiles."""
import base64
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import session_transcript as transcript

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNQLdrxHwAEXgJPUgh8kgAAAABJRU5ErkJggg==')


def test_normal_profile_cache_image_is_read(tmp_path):
    home = tmp_path / 'owner'
    image = home / 'cache' / 'images' / 'ok.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(PNG)
    options = dict.fromkeys(transcript.OPTION_KEYS, False)
    options['include_images'] = True
    doc = transcript.project_transcript({'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': str(image)}}]}]}, options, home=home)
    assert base64.b64decode(doc['messages'][0]['attachments'][0]['data_url'].partition(',')[2]) == PNG


def test_workspace_image_requires_the_explicit_trusted_root(tmp_path):
    home = tmp_path / 'owner'
    home.mkdir()
    root = tmp_path / 'work' / '.hermes' / 'attachments'
    root.mkdir(parents=True)
    image = root / 'ok.png'
    image.write_bytes(PNG)
    options = dict.fromkeys(transcript.OPTION_KEYS, False)
    options['include_images'] = True
    data = {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': str(image)}}]}]}
    assert 'data_url' not in transcript.project_transcript(data, options, home=home)['messages'][0]['attachments'][0]
    doc = transcript.project_transcript(data, options, home=home, trusted_roots=[root])
    assert base64.b64decode(doc['messages'][0]['attachments'][0]['data_url'].partition(',')[2]) == PNG


def test_unavailable_opened_handle_identity_fails_closed(tmp_path, monkeypatch):
    image = tmp_path / 'ok.png'
    image.write_bytes(PNG)
    monkeypatch.setattr(transcript, '_opened_path', lambda _handle: None)
    assert transcript._managed_image(str(image), [(tmp_path, tmp_path)]) is None


@pytest.mark.platforms('windows')
def test_cache_redirect_between_validation_and_open_is_rejected_before_read(tmp_path, monkeypatch):
    home = tmp_path / 'owner'
    cache = home / 'cache'
    image = cache / 'images' / 'ok.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(PNG)
    foreign = tmp_path / 'foreign'
    (foreign / 'images').mkdir(parents=True)
    (foreign / 'images' / 'ok.png').write_bytes(PNG)
    saved = home / 'cache-original'
    real_open = Path.open
    state = {'redirected': False, 'reads': 0}

    class GuardedFile:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def fileno(self):
            return self.handle.fileno()
        def read(self, size):
            state['reads'] += 1
            return self.handle.read(size)

    def open_after_redirect(self, *args, **kwargs):
        if self == image and not state['redirected']:
            cache.rename(saved)
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(cache), str(foreign)], check=True, capture_output=True)
            state['redirected'] = True
            return GuardedFile(real_open(self, *args, **kwargs))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_after_redirect)
    try:
        assert transcript._managed_image(str(image), [(cache / 'images', home)]) is None
        assert state == {'redirected': True, 'reads': 0}
    finally:
        if state['redirected']:
            os.rmdir(cache)
            saved.rename(cache)


def test_managed_file_over_limit_reports_a_size_error(tmp_path):
    image = tmp_path / 'oversize.png'
    with image.open('wb') as handle:
        handle.write(PNG)
        handle.truncate(transcript.MAX_IMAGE_BYTES + 1)
    with pytest.raises(transcript.TranscriptTooLargeError, match='image budget'):
        transcript._managed_image(str(image), [(tmp_path, tmp_path)])
