import json
from pathlib import Path
from types import SimpleNamespace

from capabilities.video_library.semantic import (
    SemanticClipResult,
    analyze_keyframes,
    normalize_semantic_result,
)
from capabilities.video_library.service import VideoLibraryService
from capabilities.video_library.store import VideoLibraryStore


def test_normalizes_controlled_and_free_tags():
    result = normalize_semantic_result(
        {
            "analysis": {"confidence": -0.2},
            "content": {
                "actions": ["抻面", "拉面"],
                "free_tags": ["热气升腾"],
                "subjects": ["厨师"],
                "summary": "厨师手工拉面",
            },
            "creative": {"commercial_functions": ["品质证明"]},
            "quality": {"overall_score": 1.4},
        },
        taxonomy="beef-noodle-v1",
    )

    assert "动作/拉面" in result.controlled_tags
    assert "主体/厨师" in result.controlled_tags
    assert result.controlled_tags.count("动作/拉面") == 1
    assert result.free_tags == ["热气升腾"]
    assert result.quality_score == 1.0
    assert result.confidence == 0.0


def test_vision_analyzer_sends_keyframe_not_video(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    calls = []
    payload = {
        "analysis": {"confidence": 0.92},
        "content": {"actions": ["拉面"], "subjects": ["厨师"], "summary": "厨师手工拉面"},
        "creative": {"commercial_functions": ["制作过程"]},
        "quality": {"overall_score": 0.88},
    }

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))]
        )

    monkeypatch.setattr("capabilities.video_library.semantic.call_llm", fake_call_llm)

    result = analyze_keyframes([frame], taxonomy="beef-noodle-v1")

    content = calls[0]["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in content)
    assert not any(".mp4" in str(part) for part in content)
    assert result.summary == "厨师手工拉面"
    assert result.model == "configured-vision"


def test_service_persists_semantic_result_after_technical_analysis(tmp_path, monkeypatch):
    source = tmp_path / "merchant.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(root=tmp_path / "library")
    asset = store.import_asset(source)

    def fake_semantic(_frames, *, taxonomy):
        assert taxonomy == "beef-noodle-v1"
        return SemanticClipResult(
            confidence=0.9,
            controlled_tags=["主体/厨师", "动作/拉面"],
            free_tags=["热气升腾"],
            model="fake-vision",
            quality_score=0.85,
            raw={"content": {"actions": ["拉面"]}},
            search_text="后厨厨师手工拉面",
            summary="厨师手工拉面",
        )

    service = VideoLibraryService(store, semantic_analyzer=fake_semantic, taxonomy="beef-noodle-v1")
    monkeypatch.setattr(
        "capabilities.video_library.service.media.probe_media",
        lambda _path: {"duration_seconds": 4.0, "fps": 30.0, "height": 1920, "width": 1080},
    )
    monkeypatch.setattr(
        "capabilities.video_library.service.media.detect_scene_boundaries",
        lambda *_args, **_kwargs: [(0.0, 4.0)],
    )

    def write_file(_source, output, **_kwargs):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
        return path

    monkeypatch.setattr("capabilities.video_library.service.media.extract_clip", write_file)
    monkeypatch.setattr("capabilities.video_library.service.media.extract_keyframe", write_file)

    result = service.analyze_asset(asset["id"])

    clip = result["clips"][0]
    assert clip["description"] == "厨师手工拉面"
    assert clip["semantic_json"]["content"]["actions"] == ["拉面"]
    assert clip["quality_score"] == 0.85
    assert {tag["name"] for tag in clip["tags"]} >= {"动作/拉面", "主体/厨师", "热气升腾"}
    assert result["job"]["stage"] == "indexing"
