import sqlite3

from capabilities.video_library.store import VideoLibraryStore


def test_store_creates_material_library_schema(tmp_path):
    store = VideoLibraryStore(root=tmp_path / "library")

    with sqlite3.connect(store.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {"assets", "clips", "tags", "clip_tags", "analysis_jobs"} <= tables

    with sqlite3.connect(store.db_path) as conn:
        asset_columns = {row[1] for row in conn.execute("PRAGMA table_info(assets)")}
        clip_columns = {row[1] for row in conn.execute("PRAGMA table_info(clips)")}
        job_columns = {row[1] for row in conn.execute("PRAGMA table_info(analysis_jobs)")}

    assert {"library_id", "source_mode", "source_mtime_ns"} <= asset_columns
    assert {"source_file_path", "semantic_json", "quality_score", "confidence", "materialized"} <= clip_columns
    assert {"stage", "attempts"} <= job_columns


def test_import_asset_is_content_idempotent_and_managed(tmp_path):
    source = tmp_path / "merchant source.mp4"
    source.write_bytes(b"same-video-content")
    store = VideoLibraryStore(root=tmp_path / "library")

    first = store.import_asset(source)
    second = store.import_asset(source)

    assert first["id"] == second["id"]
    assert first["sha256"] == second["sha256"]
    assert first["managed_path"].startswith(str(store.assets_dir))
    assert store.list_assets() == [second]


def test_linked_import_does_not_copy_source(tmp_path):
    root = tmp_path / "牛肉面资产库"
    source_root = root / "01_原始素材"
    source_root.mkdir(parents=True)
    source = source_root / "拉面.mp4"
    source.write_bytes(b"same-video")
    store = VideoLibraryStore(root=root, db_path=root / ".hermes-assets" / "index.sqlite")

    asset = store.import_asset(source, source_mode="linked", library_id="beef-noodle")

    assert asset["managed_path"] == str(source.resolve())
    assert asset["source_mode"] == "linked"
    assert asset["library_id"] == "beef-noodle"
    assert not any(store.assets_dir.iterdir())


def test_linked_import_is_idempotent_after_source_rename(tmp_path):
    root = tmp_path / "牛肉面资产库"
    source_root = root / "01_原始素材"
    source_root.mkdir(parents=True)
    original = source_root / "原名.mp4"
    original.write_bytes(b"same-video")
    store = VideoLibraryStore(root=root)
    first = store.import_asset(original, source_mode="linked", library_id="beef-noodle")
    renamed = original.with_name("新名字.mp4")
    original.rename(renamed)

    second = store.import_asset(renamed, source_mode="linked", library_id="beef-noodle")

    assert second["id"] == first["id"]
    assert second["source_path"] == str(renamed.resolve())
    assert second["managed_path"] == str(renamed.resolve())


def test_replace_clips_and_tags_is_transactional(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(root=tmp_path / "library")
    asset = store.import_asset(source)

    clips = store.replace_clips(
        asset["id"],
        [
            {
                "end_seconds": 3.2,
                "file_path": str(store.clips_dir / "clip-1.mp4"),
                "keyframe_path": str(store.keyframes_dir / "clip-1.jpg"),
                "start_seconds": 0.0,
            },
            {
                "end_seconds": 6.0,
                "file_path": str(store.clips_dir / "clip-2.mp4"),
                "keyframe_path": str(store.keyframes_dir / "clip-2.jpg"),
                "start_seconds": 3.2,
            },
        ],
    )
    tagged = store.replace_clip_tags(
        clips[0]["id"],
        [
            {"confidence": 0.95, "name": "门店", "source": "manual"},
            {"confidence": 0.8, "name": "竖屏", "source": "technical"},
        ],
    )

    assert [clip["clip_index"] for clip in clips] == [0, 1]
    assert clips[0]["duration_seconds"] == 3.2
    assert [tag["name"] for tag in tagged] == ["竖屏", "门店"]
    assert store.list_clips(tag="门店")[0]["id"] == clips[0]["id"]

    replacement = store.replace_clips(
        asset["id"],
        [
            {
                "end_seconds": 2.0,
                "file_path": str(store.clips_dir / "replacement.mp4"),
                "keyframe_path": "",
                "start_seconds": 0.0,
            }
        ],
    )

    assert len(replacement) == 1
    assert store.list_clips(tag="门店") == []


def test_analysis_job_lifecycle(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(root=tmp_path / "library")
    asset = store.import_asset(source)

    job = store.create_analysis_job(asset["id"], analyzer_version="scene-v1")
    completed = store.update_analysis_job(job["id"], state="complete", progress=100)

    assert job["state"] == "queued"
    assert completed["state"] == "complete"
    assert completed["progress"] == 100


def test_semantic_clip_fields_round_trip(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(root=tmp_path / "library")
    asset = store.import_asset(source)
    job = store.create_analysis_job(asset["id"], analyzer_version="semantic-v1")

    result = store.commit_analysis(
        asset["id"],
        job["id"],
        {},
        [
            {
                "confidence": 0.88,
                "description": "厨师拉面",
                "end_seconds": 4,
                "file_path": "",
                "keyframe_path": str(tmp_path / "key.jpg"),
                "materialized": False,
                "quality_score": 0.91,
                "semantic_json": {"content": {"actions": ["拉面"]}},
                "source_file_path": asset["managed_path"],
                "start_seconds": 0,
                "tags": [
                    {
                        "confidence": 0.95,
                        "name": "动作/拉面",
                        "source": "semantic-controlled",
                    }
                ],
            }
        ],
    )

    clip = result["clips"][0]
    assert clip["description"] == "厨师拉面"
    assert clip["semantic_json"]["content"]["actions"] == ["拉面"]
    assert clip["quality_score"] == 0.91
    assert clip["confidence"] == 0.88
    assert clip["materialized"] is False

    matches = store.search_clips("厨师拉面")
    assert matches[0]["id"] == clip["id"]
    assert matches[0]["score"] > 0


def test_search_clips_matches_relevant_phrases_inside_a_full_chinese_script_sentence(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(root=tmp_path / "library")
    asset = store.import_asset(source)
    job = store.create_analysis_job(asset["id"], analyzer_version="semantic-v1")
    result = store.commit_analysis(
        asset["id"],
        job["id"],
        {},
        [
            {
                "confidence": 0.9,
                "description": "后厨大型汤锅热气升腾，大块牛肉清晰可见",
                "end_seconds": 5,
                "file_path": "",
                "keyframe_path": str(tmp_path / "key.jpg"),
                "materialized": False,
                "quality_score": 0.88,
                "semantic_json": {"content": {"subjects": ["汤锅", "牛肉"]}},
                "source_file_path": asset["managed_path"],
                "start_seconds": 0,
                "tags": [{"name": "内容/热汤", "source": "semantic-controlled"}],
            }
        ],
    )

    matches = store.search_clips("热气腾腾的牛肉汤刚刚出锅，大块牛肉铺满整碗。")

    assert matches[0]["id"] == result["clips"][0]["id"]
    assert matches[0]["score"] > 0
