"""Open-tab file: shared across devices, revision-checked, never a partial write."""

import json
from pathlib import Path

import pytest

from hermes_cli.desktop_open_tabs import RevisionConflict, empty_document, load, normalize_tiles, save


def test_missing_file_is_an_empty_revision(tmp_path: Path):
    assert load(tmp_path) == empty_document()


def test_round_trip_drops_runtime_fields_and_bumps_revision(tmp_path: Path):
    saved = save(
        tmp_path,
        [
            {
                "storedSessionId": "sess-a",
                "dir": "right",
                "anchor": "workspace",
                "before": None,
                "runtimeId": "must-not-persist",
                "error": "nope",
            },
            {"storedSessionId": "sess-b", "dir": "center"},
        ],
        0,
    )
    assert saved["revision"] == 1
    assert saved["updated_at"]
    assert saved["tiles"] == [
        {"storedSessionId": "sess-a", "dir": "right", "anchor": "workspace", "before": None},
        {"storedSessionId": "sess-b", "dir": "center"},
    ]
    assert load(tmp_path)["tiles"] == saved["tiles"]
    assert "runtimeId" not in (tmp_path / "desktop-open-tabs.json").read_text()


def test_stale_revision_does_not_clobber(tmp_path: Path):
    save(tmp_path, [{"storedSessionId": "kept"}], 0)
    with pytest.raises(RevisionConflict) as caught:
        save(tmp_path, [{"storedSessionId": "lost"}], 0)
    assert caught.value.current["tiles"] == [{"storedSessionId": "kept"}]
    assert load(tmp_path)["tiles"] == [{"storedSessionId": "kept"}]
    updated = save(tmp_path, [], 1)
    assert updated["revision"] == 2
    assert updated["tiles"] == []


def test_homes_are_isolated(tmp_path: Path):
    macbook = tmp_path / "macbook"
    mini = tmp_path / "mini"
    save(macbook, [{"storedSessionId": "on-macbook"}], 0)
    save(mini, [{"storedSessionId": "on-mini"}], 0)
    assert load(macbook)["tiles"][0]["storedSessionId"] == "on-macbook"
    assert load(mini)["tiles"][0]["storedSessionId"] == "on-mini"


def test_corrupt_file_reads_as_empty_and_can_be_replaced(tmp_path: Path):
    path = tmp_path / "desktop-open-tabs.json"
    path.write_text("{not json", encoding="utf-8")
    assert load(tmp_path) == empty_document()
    save(tmp_path, [{"storedSessionId": "recovered"}], 0)
    assert load(tmp_path)["tiles"] == [{"storedSessionId": "recovered"}]


def test_normalize_caps_and_dedupes():
    raw = [{"storedSessionId": "dup"}, {"storedSessionId": "dup"}, {"storedSessionId": "  "}]
    raw.extend({"storedSessionId": f"s{i}"} for i in range(50))
    tiles = normalize_tiles(raw)
    assert tiles[0]["storedSessionId"] == "dup"
    assert len(tiles) == 40
    assert json.dumps(tiles).count("runtimeId") == 0


def test_router_get_put_and_conflict(tmp_path: Path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from hermes_cli.web_routers import desktop_open_tabs as routes

    async def _inline(_profile, fn):
        return fn()

    monkeypatch.setattr(routes, "scoped_to_thread", _inline)
    monkeypatch.setattr(routes, "destructive_profile", lambda profile, _route: profile)
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home",
        lambda: tmp_path,
    )
    # The handler imports get_hermes_home inside the call. Patch both seams.
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    empty = client.get("/api/desktop/open-tabs")
    assert empty.status_code == 200
    assert empty.json()["revision"] == 0
    assert empty.json()["tiles"] == []

    saved = client.put(
        "/api/desktop/open-tabs",
        json={"base_revision": 0, "tiles": [{"storedSessionId": "from-macbook", "dir": "right"}]},
    )
    assert saved.status_code == 200
    assert saved.json()["revision"] == 1

    conflict = client.put(
        "/api/desktop/open-tabs",
        json={"base_revision": 0, "tiles": [{"storedSessionId": "stale"}]},
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["tiles"][0]["storedSessionId"] == "from-macbook"

    followed = client.get("/api/desktop/open-tabs")
    assert followed.json()["tiles"][0]["storedSessionId"] == "from-macbook"
