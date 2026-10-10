"""A no-overwrite upload must not replace a file created after its preflight."""

import asyncio
import base64
import tempfile
from typing import BinaryIO, cast

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.web_routers import files


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.asyncio
async def test_competing_stream_uploads_honor_overwrite(
    tmp_path, monkeypatch, overwrite
):
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    target = root / "report.bin"
    arrived = 0
    ready = asyncio.Event()

    class SynchronizedUpload(UploadFile):
        async def read(self, size=-1):
            nonlocal arrived
            if self.file.tell() == 0:
                arrived += 1
                if arrived == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), timeout=10)
            return await super().read(size)

    bodies = [b"first upload" * 1000, b"second upload" * 1000]
    uploads = []
    for body in bodies:
        spool = tempfile.SpooledTemporaryFile(max_size=64)
        spool.write(body)
        spool.seek(0)
        uploads.append(
            SynchronizedUpload(file=cast(BinaryIO, spool), filename=target.name)
        )
    try:
        results = await asyncio.gather(
            *(
                files.upload_managed_file_stream(
                    request=Request({"type": "http"}),
                    file=upload,
                    path=str(target),
                    overwrite=overwrite,
                )
                for upload in uploads
            ),
            return_exceptions=True,
        )
        winners = [i for i, result in enumerate(results) if isinstance(result, dict)]
        if overwrite:
            assert len(winners) == 2
            assert target.read_bytes() in bodies
        else:
            assert len(winners) == 1, results
            loser = results[1 - winners[0]]
            assert isinstance(loser, HTTPException) and loser.status_code == 409
            assert target.read_bytes() == bodies[winners[0]]
        assert sorted(p.name for p in root.iterdir()) == [target.name]
        assert all(upload.file.closed for upload in uploads)
    finally:
        for upload in uploads:
            await upload.close()


@pytest.mark.parametrize("overwrite", [False, True])
def test_json_upload_honors_file_created_during_decode(
    tmp_path, monkeypatch, overwrite
):
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    target = root / "report.txt"
    competing_content = b"saved by another client"
    uploaded_content = b"new upload"
    decode = files._decode_data_url

    def decode_with_competing_save(data_url):
        result = decode(data_url)
        target.write_bytes(competing_content)
        return result

    monkeypatch.setattr(files, "_decode_data_url", decode_with_competing_save)
    with TestClient(web_server.app) as client:
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        response = client.post(
            "/api/files/upload",
            json={
                "path": str(target),
                "overwrite": overwrite,
                "data_url": "data:text/plain;base64,"
                + base64.b64encode(uploaded_content).decode("ascii"),
            },
        )
    assert response.status_code == (200 if overwrite else 409), response.text
    assert target.read_bytes() == (uploaded_content if overwrite else competing_content)
    assert sorted(p.name for p in root.iterdir()) == [target.name]
