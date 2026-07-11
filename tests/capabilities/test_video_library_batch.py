from pathlib import Path

from capabilities.video_library.batch import VideoLibraryBatchRunner
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
