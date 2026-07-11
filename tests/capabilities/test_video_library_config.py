from pathlib import Path

import pytest

from capabilities.video_library import config as video_library_config
from capabilities.video_library.config import (
    VideoLibraryConfig,
    load_library_configs,
    resolve_library_config,
    resolve_source_path,
)


def _config(root: Path, source: Path) -> dict:
    return {
        "video_libraries": [
            {
                "id": "beef-noodle",
                "mode": "linked",
                "name": "牛肉面资产库",
                "root": str(root),
                "source_roots": [str(source)],
                "taxonomy": "beef-noodle-v1",
            }
        ]
    }


def test_loads_linked_obsidian_library(tmp_path: Path):
    root = tmp_path / "牛肉面资产库"
    source = root / "01_原始素材"
    source.mkdir(parents=True)

    libraries = load_library_configs(_config(root, source))

    library = libraries["beef-noodle"]
    assert library.database_path == root / ".hermes-assets" / "index.sqlite"
    assert library.keyframes_dir == root / "03_关键帧"
    assert library.selected_clips_dir == root / "02_精选镜头"
    assert library.source_roots == (source.resolve(),)


def test_resolve_library_config_rejects_unknown_id(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(KeyError, match="unknown video library"):
        resolve_library_config("other", config=_config(tmp_path / "vault", source))


def test_rejects_source_outside_allowlist(tmp_path: Path):
    source = tmp_path / "allowed"
    source.mkdir()
    library = load_library_configs(_config(tmp_path / "vault", source))["beef-noodle"]
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"video")

    with pytest.raises(ValueError, match="outside configured source roots"):
        resolve_source_path(library, outside)


def test_rejects_symlink_escape_from_allowlist(tmp_path: Path):
    source = tmp_path / "allowed"
    source.mkdir()
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"video")
    link = source / "linked.mp4"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    library = load_library_configs(_config(tmp_path / "vault", source))["beef-noodle"]

    with pytest.raises(ValueError, match="outside configured source roots"):
        resolve_source_path(library, link)


def test_rejects_duplicate_library_ids(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    entry = _config(tmp_path / "vault", source)["video_libraries"][0]

    with pytest.raises(ValueError, match="unique"):
        load_library_configs({"video_libraries": [entry, dict(entry)]})


def test_generated_library_path_only_matches_managed_output_directories(tmp_path: Path):
    root = tmp_path / "牛肉面资产库"
    library = VideoLibraryConfig(
        id="beef-noodle",
        mode="linked",
        name="牛肉面资产库",
        root=root,
        source_roots=(root,),
        taxonomy="beef-noodle-v1",
    )

    classify = video_library_config.is_generated_library_path
    assert classify(library, root / "02_精选镜头" / "clip.mp4") is True
    assert classify(library, root / "03_关键帧" / "frame.jpg") is True
    assert classify(library, root / "04_素材分析" / "report.md") is True
    assert classify(library, root / "timelines" / "timeline.json") is True
    assert classify(library, root / ".hermes-assets" / "managed-assets" / "copy.mp4") is True
    assert classify(library, root / "01_原始素材" / "raw.mp4") is False
    assert classify(library, tmp_path / "outside.mp4") is False
