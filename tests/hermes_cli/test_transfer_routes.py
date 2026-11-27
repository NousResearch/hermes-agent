"""``/api/ops/transfer/*``: a backup carried to a cloud agent as JSON chunks, then restored there.

The real routes on a FastAPI app; only the spawned ``hermes import`` is recorded, not run.
"""

import base64
import io
import zipfile

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from hermes_cli.web_routers import ops
    from hermes_cli.web_routers import transfer

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    spawned: list[list[str]] = []
    monkeypatch.setattr(ops, "_spawn_action", lambda argv, name, **kw: spawned.append(argv) or {"ok": True, "name": name})
    app = FastAPI()
    app.include_router(transfer.router)
    return TestClient(app), spawned


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("config.yaml", "display:\n  skin: ares\n")
    return buf.getvalue()


def _send(http, upload_id, offset, data: bytes):
    return http.post("/api/ops/transfer/chunk",
                     json={"upload_id": upload_id, "offset": offset, "data": base64.b64encode(data).decode()})


def test_chunks_assemble_the_archive_and_restore_keeps_the_host(client):
    http, spawned = client
    archive = _zip()
    first = _send(http, None, 0, archive[:10]).json()
    assert _send(http, first["upload_id"], 10, archive[10:]).json()["received"] == len(archive)
    # A retried chunk rewrites from its own offset; a gap names what the upload holds.
    assert _send(http, first["upload_id"], 10, archive[10:]).json()["received"] == len(archive)
    gap = _send(http, first["upload_id"], len(archive) + 5, b"x")
    assert gap.status_code == 409 and gap.json()["detail"]["received"] == len(archive)

    started = http.post("/api/ops/transfer/restore", json={"upload_id": first["upload_id"], "size": len(archive)})

    assert started.status_code == 200 and started.json()["name"] == "import"
    assert spawned[0][-3:] == ["--force", "--keep-host", "--messaging-off"]


def test_a_body_past_one_chunk_is_refused_before_it_is_read(client):
    http, _ = client
    from hermes_cli.web_routers.transfer import _BODY_MAX
    response = http.post("/api/ops/transfer/chunk", content=b"x" * (_BODY_MAX + 1),
                         headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_starting_an_upload_reclaims_abandoned_ones(client):
    """A canceled or failed transfer never reaches restore; the next upload drops its leftovers."""
    import os
    import time

    from hermes_cli.web_routers.transfer import _backups
    http, _ = client
    backups = _backups(None)
    backups.mkdir(parents=True, exist_ok=True)
    stale, live = backups / f"transfer-upload-{'a' * 24}.zip", backups / f"transfer-upload-{'b' * 24}.zip"
    stale.write_bytes(b"partial")
    live.write_bytes(b"in progress")
    two_hours_ago = time.time() - 7200
    os.utime(stale, (two_hours_ago, two_hours_ago))

    assert _send(http, None, 0, b"PK").status_code == 200

    assert not stale.exists() and live.exists()


def test_a_late_duplicate_chunk_never_cuts_off_the_chunks_after_it(client):
    """A dropped request can land after its retry and the next chunk: it must not truncate them."""
    http, _ = client
    archive = _zip()
    first = _send(http, None, 0, archive[:10]).json()
    assert _send(http, first["upload_id"], 10, archive[10:20]).json()["received"] == 20
    assert _send(http, first["upload_id"], 20, archive[20:]).json()["received"] == len(archive)

    late = _send(http, first["upload_id"], 10, archive[10:20])
    differing = _send(http, first["upload_id"], 10, b"x" * 10)

    assert late.status_code == 200 and late.json()["received"] == len(archive)
    assert differing.status_code == 409
    restored = http.post("/api/ops/transfer/restore", json={"upload_id": first["upload_id"], "size": len(archive)})
    assert restored.status_code == 200
