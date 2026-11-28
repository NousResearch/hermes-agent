"""Synthetic managed image boundary and image size regressions."""
import base64
import random
import struct
import subprocess
import zlib

import pytest

from hermes_cli.session_transcript import _managed_image, _image_data_url, project_transcript, OPTION_KEYS, TranscriptTooLargeError


@pytest.mark.platforms('windows')
def test_junction_in_managed_root_ancestor_is_refused(tmp_path):
    owner = tmp_path / 'owner'
    owner.mkdir()
    foreign = tmp_path / 'foreign'
    (foreign / 'images').mkdir(parents=True)
    (foreign / 'images' / 'x.png').write_bytes(b'\x89PNG\r\n\x1a\nFOREIGN')
    subprocess.run(['cmd', '/c', 'mklink', '/J', str(owner / 'cache'), str(foreign)], check=True, capture_output=True)
    options = dict.fromkeys(OPTION_KEYS, False)
    options['include_images'] = True
    ref = str(owner / 'cache' / 'images' / 'x.png')
    doc = project_transcript({'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': ref}}]}]}, options, home=owner)
    assert 'data_url' not in doc['messages'][0]['attachments'][0]


def test_large_image_data_within_budget(tmp_path):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    width = height = 1100
    rng = random.Random(42)
    pixels = b''.join(b'\0' + rng.randbytes(width * 3) for _ in range(height))
    raw = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))
    assert len(raw) > 3_600_000
    value = 'data:image/png;base64,' + base64.b64encode(raw).decode()
    assert _image_data_url(value) == value
    root = tmp_path / 'images'
    root.mkdir()
    image = root / 'large.png'
    image.write_bytes(raw)
    assert _managed_image(str(image), [(root, tmp_path)]) == value


def test_over_budget_data_fails_explicitly():
    raw = b'\x89PNG\r\n\x1a\n' + b'x' * (16 * 1024 * 1024)
    value = 'data:image/png;base64,' + base64.b64encode(raw).decode()
    with pytest.raises(TranscriptTooLargeError):
        _image_data_url(value)
