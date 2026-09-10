"""Real authenticated multipart storage for browser draft attachments."""
import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def uploads(monkeypatch, tmp_path):
    home = tmp_path / '.hermes'
    profile = home / 'profiles' / 'files-check'
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    from hermes_cli import web_server
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    yield client, home, profile, web_server
    client.close()


def test_multipart_upload_is_private_unique_profile_scoped_and_generic(uploads):
    client, home, profile, ws = uploads
    paths = []
    for query, root in [('', home), ('?profile=files-check', profile), ('?profile=files-check', profile)]:
        response = client.post('/api/chat/file-upload' + query,
                               files={'file': ('../../notes.txt', b'generic bytes', 'text/plain')})
        assert response.status_code == 200, response.text
        data = response.json()
        path = Path(data['path'])
        assert path.is_relative_to(root / 'attachments')
        assert path.name == data['name'] == 'notes.txt'
        assert data['bytes'] == len(b'generic bytes')
        assert data['mime_type'] == 'text/plain'
        assert path.read_bytes() == b'generic bytes'
        paths.append(path)
    assert len(set(paths)) == len(paths)
    assert client.post('/api/chat/file-upload', files={'file': ('a', b'x')},
                       headers={ws._SESSION_HEADER_NAME: 'invalid'}).status_code == 401
    assert client.post('/api/chat/file-upload?profile=missing',
                       files={'file': ('a', b'x')}).status_code == 404


@pytest.mark.linux_only
def test_multipart_upload_uses_private_directory_and_file(uploads):
    client, _, _, _ = uploads
    response = client.post('/api/chat/file-upload', files={'file': ('a.txt', b'x')})
    assert response.status_code == 200
    path = Path(response.json()['path'])
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600


def test_multipart_cap_and_cancel_clean_up(uploads, monkeypatch):
    client, home, _, ws = uploads
    monkeypatch.setattr(ws, '_MANAGED_FILE_MAX_BYTES', 4)
    response = client.post('/api/chat/file-upload', files={'file': ('a.txt', b'too large')})
    assert response.status_code == 413
    assert list((home / 'attachments').iterdir()) == []

    from hermes_cli.web_routers import files
    from fastapi import UploadFile

    class CancelledUpload(UploadFile):
        async def read(self, size=-1):
            raise asyncio.CancelledError()

    upload = CancelledUpload(filename='a.txt', file=BytesIO(b'x'))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(files.upload_chat_file(file=upload))
    assert upload.file.closed
    assert list((home / 'attachments').iterdir()) == []
