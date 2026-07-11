from pathlib import Path

from capabilities.video_library import batch
from capabilities.video_library.batch import VideoLibraryBatchRunner, build_library_service
from capabilities.video_library.config import VideoLibraryConfig


def _library(tmp_path: Path, names: list[str]) -> VideoLibraryConfig:
    root = tmp_path / "牛肉面资产库"
    source = root / "01_原始素材"
    source.mkdir(parents=True)
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
    return VideoLibraryConfig(
        id="beef-noodle",
        mode="linked",
        name="牛肉面资产库",
        root=root,
        source_roots=(source,),
        taxonomy="beef-noodle-v1",
    )


def test_scan_continues_after_one_file_fails(tmp_path, monkeypatch):
    library = _library(tmp_path, ["good.mp4", "bad.mp4", "nested/good2.mov", "ignore.txt"])
    runner = VideoLibraryBatchRunner(library)

    def process(path):
        if path.name == "bad.mp4":
            raise RuntimeError("broken")
        return {"status": "complete"}

    monkeypatch.setattr(runner, "process_file", process)

    result = runner.scan()

    assert result.total == 3
    assert result.complete == 2
    assert result.failed == 1
    assert result.errors[0]["file"].endswith("bad.mp4")


def test_scan_counts_skipped_unchanged_content(tmp_path, monkeypatch):
    library = _library(tmp_path, ["one.mp4"])
    runner = VideoLibraryBatchRunner(library)
    monkeypatch.setattr(runner, "process_file", lambda _path: {"status": "skipped"})

    result = runner.scan()

    assert result.total == 1
    assert result.skipped == 1
    assert result.complete == 0


def test_dry_run_does_not_create_asset_database(tmp_path):
    library = _library(tmp_path, ["one.mp4", "two.mov"])
    runner = VideoLibraryBatchRunner(library)

    result = runner.scan(dry_run=True)

    assert result.dry_run is True
    assert result.total == 2
    assert result.writes_planned
    assert not library.database_path.exists()


def test_scan_rejects_source_symlink_escape(tmp_path):
    library = _library(tmp_path, [])
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    link = library.source_roots[0] / "linked.mp4"
    try:
        link.symlink_to(outside)
    except OSError:
        return

    result = VideoLibraryBatchRunner(library).scan(dry_run=True)

    assert result.total == 0
    assert result.failed == 1
    assert result.errors[0]["stage"] == "authorization"


def test_scan_excludes_generated_video_directories_when_library_root_is_a_source(tmp_path):
    root = tmp_path / "牛肉面资产库"
    (root / "01_原始素材").mkdir(parents=True)
    (root / "02_精选镜头" / "asset-1").mkdir(parents=True)
    (root / ".hermes-assets" / "managed-assets").mkdir(parents=True)
    (root / "01_原始素材" / "raw.mp4").write_bytes(b"raw")
    (root / "02_精选镜头" / "asset-1" / "clip.mp4").write_bytes(b"derived")
    (root / ".hermes-assets" / "managed-assets" / "copy.mp4").write_bytes(b"derived")
    library = VideoLibraryConfig(
        id="beef-noodle",
        mode="linked",
        name="牛肉面资产库",
        root=root,
        source_roots=(root,),
        taxonomy="beef-noodle-v1",
    )

    result = VideoLibraryBatchRunner(library).scan(dry_run=True)

    assert result.total == 1


def test_prune_derived_assets_is_dry_run_by_default(tmp_path, monkeypatch):
    library = _library(tmp_path, [])
    derived = library.root / "02_精选镜头" / "old.mp4"
    derived.parent.mkdir(parents=True)
    derived.write_bytes(b"derived")
    service = build_library_service(library)
    asset = service.store.import_asset(derived, source_mode="linked", library_id=library.id)
    monkeypatch.setattr(batch, "resolve_library_config", lambda _library_id: library)

    preview = batch.prune_derived_assets(library.id)

    assert preview["matched"] == 1
    assert preview["deleted"] == 0
    assert service.store.get_asset(asset["id"]) is not None

    executed = batch.prune_derived_assets(library.id, execute=True)

    assert executed["deleted"] == 1
    assert service.store.get_asset(asset["id"]) is None
    assert derived.is_file()


def test_library_status_reports_failed_assets_separately_from_semantic_failures(tmp_path, monkeypatch):
    library = _library(tmp_path, [])
    source = library.source_roots[0] / "raw.mp4"
    source.write_bytes(b"raw")
    service = build_library_service(library)
    asset = service.store.import_asset(source, source_mode="linked", library_id=library.id)
    service.store.update_asset_metadata(asset["id"], {}, status="failed")
    clip = service.store.replace_clips(
        asset["id"],
        [{"end_seconds": 1.0, "source_file_path": str(source), "start_seconds": 0.0}],
    )[0]
    service.store.update_clip_status(clip["id"], "semantic_failed")
    monkeypatch.setattr(batch, "resolve_library_config", lambda _library_id: library)

    status = batch.library_status(library.id)

    assert status["failed_assets"] == 1
    assert status["semantic_failed"] == 1
    assert status["failed"] == 1
