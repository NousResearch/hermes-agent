import importlib
from pathlib import Path
import subprocess
import sys


class FakeClient:
    def search(self, library_id: str, query: str, limit: int = 5):
        suffix = query.replace(" ", "-")
        return [
            {
                "id": f"clip-{suffix}",
                "asset_id": f"asset-{suffix}",
                "description": query,
                "quality_score": 0.9,
                "confidence": 0.9,
                "score": 1.0,
                "source_file_path": f"/source/{suffix}.mp4",
            }
        ]


def test_acceptance_plan_selects_one_distinct_shot_per_required_intent():
    acceptance = importlib.import_module("scripts.video_library_acceptance")

    plan = acceptance.build_acceptance_plan(FakeClient(), "beef-noodle")

    assert [row["query"] for row in plan["selections"]] == [
        "前厅顾客吃面",
        "员工端碗上餐",
        "成品牛肉面特写",
    ]
    assert len({row["asset_id"] for row in plan["selections"]}) == 3


def test_acceptance_script_runs_directly_from_repository_root():
    repository = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "scripts/video_library_acceptance.py", "--help"],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--library" in completed.stdout


class FakeRenderClient(FakeClient):
    def __init__(self, final_path: Path):
        self.calls = []
        self.final_path = final_path

    def create_timeline(self, library_id, clip_ids, script):
        self.calls.append(("timeline", library_id, clip_ids, script))
        video = [
            {"clipId": clip_id, "file": f"/materialized/{clip_id}.mp4"}
            for clip_id in clip_ids
        ]
        shots = [
            {
                "assetId": f"asset-{index}",
                "clipId": clip_id,
                "libraryId": library_id,
                "sourcePath": f"/source/{clip_id}.mp4",
                "sourceSha256": f"sha-{index}",
            }
            for index, clip_id in enumerate(clip_ids)
        ]
        return {
            "path": "/library/timelines/test.json",
            "timeline": {"tracks": {"video": video}, "shotPlan": shots},
        }

    def cache_material(self, source_path, filename):
        self.calls.append(("cache", source_path, filename))
        return filename

    def cache_audio(self, source_path):
        self.calls.append(("audio", source_path))
        return "acceptance.mp3"

    def create_video(self, body):
        self.calls.append(("render", body))
        return "task-local"

    def get_task(self, task_id):
        self.calls.append(("poll", task_id))
        return {"id": task_id, "state": "complete", "final_video_path": str(self.final_path)}


def test_execute_render_preserves_named_library_order_and_provenance(tmp_path):
    acceptance = importlib.import_module("scripts.video_library_acceptance")
    final_path = tmp_path / "final-1.mp4"
    final_path.write_bytes(b"mp4")
    audio_path = tmp_path / "acceptance.mp3"
    audio_path.write_bytes(b"mp3")
    client = FakeRenderClient(final_path)
    plan = acceptance.build_acceptance_plan(client, "beef-noodle")

    result = acceptance.execute_render(
        client,
        plan,
        audio_path=audio_path,
        timeout_seconds=5,
    )

    timeline_call = next(call for call in client.calls if call[0] == "timeline")
    render_call = next(call for call in client.calls if call[0] == "render")
    assert timeline_call[1] == "beef-noodle"
    assert render_call[1]["video_source"] == "local"
    assert render_call[1]["match_materials_to_script"] is True
    assert [item["url"] for item in render_call[1]["video_materials"]] == [
        call[2] for call in client.calls if call[0] == "cache"
    ]
    assert result["task_id"] == "task-local"
    assert result["timeline_path"] == "/library/timelines/test.json"
    assert result["final_video_path"] == str(final_path)
    assert len(result["selected_sources"]) == 3
