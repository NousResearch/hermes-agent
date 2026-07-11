from pathlib import Path

import pytest

from capabilities.video_library.store import VideoLibraryStore


def _raw_config(root: Path, source: Path) -> dict:
    return {
        "video_libraries": [
            {
                "id": "beef-noodle",
                "name": "牛肉面资产库",
                "root": str(root),
                "source_roots": [str(source)],
                "mode": "linked",
                "taxonomy": "beef-noodle-v1",
            }
        ]
    }


def test_add_source_root_persists_real_directory_once(tmp_path, monkeypatch):
    from capabilities.video_library import management

    original = tmp_path / "original"
    original.mkdir()
    added = tmp_path / "merchant-footage"
    added.mkdir()
    raw = _raw_config(tmp_path / "vault", original)
    writes = []
    monkeypatch.setattr(management, "read_raw_config", lambda: raw)
    monkeypatch.setattr(management, "save_config", lambda value, **kwargs: writes.append((value, kwargs)))

    first = management.add_library_source_root("beef-noodle", added)
    second = management.add_library_source_root("beef-noodle", added)

    assert first["source_roots"] == [str(original.resolve()), str(added.resolve())]
    assert second["source_roots"] == first["source_roots"]
    assert len(writes) == 1
    assert writes[0][1]["preserve_keys"] == {("video_libraries",)}


def test_add_source_root_rejects_file(tmp_path):
    from capabilities.video_library.management import add_library_source_root

    path = tmp_path / "not-a-directory.mp4"
    path.write_bytes(b"video")

    with pytest.raises(ValueError, match="directory"):
        add_library_source_root("beef-noodle", path)


def test_migrate_legacy_library_is_idempotent_and_preserves_legacy(tmp_path, monkeypatch):
    from capabilities.video_library import management

    source = tmp_path / "legacy.mp4"
    source.write_bytes(b"same-content")
    legacy = VideoLibraryStore(root=tmp_path / "legacy")
    old = legacy.import_asset(source)
    target = VideoLibraryStore(root=tmp_path / "target")
    monkeypatch.setattr(management, "get_legacy_store", lambda: legacy)
    monkeypatch.setattr(management, "get_named_store", lambda _library_id: target)

    first = management.migrate_legacy_library("beef-noodle")
    second = management.migrate_legacy_library("beef-noodle")

    assert first["imported"] == 1
    assert second["skipped"] == 1
    assert legacy.get_asset(old["id"])["id"] == old["id"]
    assert len(target.list_assets()) == 1
    assert first["records"][0]["source_asset_id"] == old["id"]
    assert first["records"][0]["target_asset_id"]


def test_migrate_legacy_library_records_partial_failures(tmp_path, monkeypatch):
    from capabilities.video_library import management

    missing = tmp_path / "missing.mp4"

    class LegacyStore:
        def list_assets(self):
            return [{"id": "asset-missing", "managed_path": str(missing), "sha256": "missing"}]

    monkeypatch.setattr(management, "get_legacy_store", LegacyStore)
    monkeypatch.setattr(
        management,
        "get_named_store",
        lambda _library_id: VideoLibraryStore(root=tmp_path / "target"),
    )

    result = management.migrate_legacy_library("beef-noodle")

    assert result["failed"] == 1
    assert result["records"][0]["source_asset_id"] == "asset-missing"
    assert result["records"][0]["target_asset_id"] == ""
    assert result["records"][0]["state"] == "failed"
    assert "missing.mp4" in result["records"][0]["error"]
