"""Unavailable model files must not break discovery or the Local Models status endpoint."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.local_runtime import bootstrap
from hermes_cli.web_routers import local_models


pytestmark = pytest.mark.platforms("any")


@pytest.fixture
def model_status(tmp_path, monkeypatch):
    mdir = tmp_path / "models"
    mdir.mkdir()
    monkeypatch.setattr(bootstrap, "models_dir", lambda: mdir)
    monkeypatch.setattr(local_models, "_state_endpoint", lambda: None)
    monkeypatch.setattr(local_models.binaries, "installed_engine", lambda backend: None)
    app = FastAPI()
    app.include_router(local_models.router)
    with TestClient(app) as client:
        yield mdir, client


@pytest.mark.parametrize(
    "kind", ["single", "first-shard", "later-shard", "missing-shard", "directory"]
)
def test_unavailable_files_are_not_staged_and_recover(model_status, tmp_path, kind):
    mdir, client = model_status
    healthy = mdir / "healthy.gguf"
    healthy.write_bytes(b"healthy")
    target = tmp_path / "external.gguf"
    if kind in {"first-shard", "later-shard", "missing-shard"}:
        first = mdir / "split-00001-of-00002.gguf"
        later = mdir / "split-00002-of-00002.gguf"
        missing = first if kind == "first-shard" else later
        (later if missing == first else first).write_bytes(b"part")
        model_id = "split"
    else:
        missing = mdir / "single.gguf"
        model_id = "single"
    if kind == "directory":
        missing.mkdir()
    elif kind != "missing-shard":
        missing.symlink_to(target)

    assert bootstrap.staged_models() == [healthy]
    response = client.get("/api/local-models/status")
    assert response.status_code == 200
    assert response.json()["models"] == [
        {"id": "healthy", "size_bytes": 7, "size_label": "0.0 GB"},
    ]
    # Reconnecting a volume or completing a download restores the row on the next poll.
    if kind == "directory":
        missing.rmdir()
        missing.write_bytes(b"restored")
    elif kind == "missing-shard":
        missing.write_bytes(b"restored")
    else:
        assert missing.is_symlink()  # Discovery never deletes the user's link.
        target.write_bytes(b"restored")
    rows = {
        row["id"]: row
        for row in client.get("/api/local-models/status").json()["models"]
    }
    assert set(rows) == {"healthy", model_id}
    assert rows[model_id]["size_bytes"] == (12 if model_id == "split" else 8)


@pytest.mark.parametrize("kind", ["single", "first-shard", "later-shard"])
@pytest.mark.parametrize("phase", ["discovery", "sizing", "permission"])
def test_status_isolates_files_lost_during_a_poll(
    model_status, monkeypatch, caplog, kind, phase
):
    mdir, client = model_status
    (mdir / "healthy.gguf").write_bytes(b"healthy")
    if kind == "single":
        victim = mdir / "vanishing.gguf"
    else:
        first = mdir / "split-00001-of-00002.gguf"
        later = mdir / "split-00002-of-00002.gguf"
        first.write_bytes(b"part")
        later.write_bytes(b"part")
        victim = first if kind == "first-shard" else later
    victim.write_bytes(b"part")
    discover = bootstrap.staged_models
    split_parts = local_models.split_parts
    stat = Path.stat

    def lose_after_discovery():
        models = discover()
        victim.unlink()
        return models

    def lose_before_sizing(path):
        parts = split_parts(path)
        if path.name != "healthy.gguf":
            victim.unlink()
        return parts

    def denied(path, *args, **kwargs):
        if path == victim:
            raise PermissionError("model volume is inaccessible")
        return stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        if phase == "discovery":
            patch.setattr(bootstrap, "staged_models", lose_after_discovery)
        elif phase == "sizing":
            patch.setattr(local_models, "split_parts", lose_before_sizing)
        else:
            # Fail after discovery, so the endpoint must handle a later filesystem error.
            models = discover()
            patch.setattr(bootstrap, "staged_models", lambda: models)
            patch.setattr(Path, "stat", denied)
        response = client.get("/api/local-models/status")
        assert response.status_code == 200
        assert [row["id"] for row in response.json()["models"]] == ["healthy"]
        assert "unavailable" in caplog.text

    victim.write_bytes(b"part")
    rows = client.get("/api/local-models/status").json()["models"]
    assert len(rows) == 2
    assert next(row for row in rows if row["id"] != "healthy")["size_bytes"] == (
        4 if kind == "single" else 8
    )
