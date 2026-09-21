"""Strict TDD contract for the bounded foreground SahilBlog runner."""
import json
from pathlib import Path

import pytest


def test_run_once_writes_atomic_state_and_terminal_record(tmp_path):
    from blog.scheduler_runner import run_once

    seen = {}

    def child(*, deadline, max_new_drafts, max_images):
        seen.update(deadline=deadline, max_new_drafts=max_new_drafts, max_images=max_images)
        return {"drafts_created": 1, "images_created": 3}

    result = run_once(
        state_root=tmp_path,
        child_runner=child,
        max_new_drafts=1,
        max_images=3,
        timeout_seconds=10,
    )

    assert result["status"] == "completed"
    assert result["run_id"]
    assert seen["deadline"] > 0
    assert seen["max_new_drafts"] == 1
    assert seen["max_images"] == 3

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["active_run_id"] is None
    assert state["last_run_id"] == result["run_id"]
    run_record = json.loads((tmp_path / "runs" / f"{result['run_id']}.json").read_text())
    assert run_record["status"] == "completed"
    assert run_record["result"] == {"drafts_created": 1, "images_created": 3}
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize(("max_new_drafts", "max_images"), [(2, 3), (1, 4), (-1, 1), (1, -1)])
def test_run_once_rejects_unbounded_work_before_child_runs(tmp_path, max_new_drafts, max_images):
    from blog.scheduler_runner import run_once

    with pytest.raises(ValueError):
        run_once(
            state_root=tmp_path,
            child_runner=lambda **_: pytest.fail("child must not run"),
            max_new_drafts=max_new_drafts,
            max_images=max_images,
        )


def test_run_once_returns_locked_without_starting_another_child(tmp_path):
    from blog.scheduler_runner import run_once

    def nested_child(**_):
        return run_once(
            state_root=tmp_path,
            child_runner=lambda **__: pytest.fail("second child must not run"),
        )

    result = run_once(state_root=tmp_path, child_runner=nested_child)

    assert result["status"] == "completed"
    assert result["result"] == {"status": "locked", "run_id": None}


def test_run_once_marks_timeout_and_clears_active_run_in_finally(tmp_path):
    from blog.scheduler_runner import run_once

    def timed_out_child(**_):
        raise TimeoutError("deadline exceeded")

    result = run_once(state_root=tmp_path, child_runner=timed_out_child)

    assert result["status"] == "timed_out"
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["active_run_id"] is None
    run_record = json.loads((tmp_path / "runs" / f"{result['run_id']}.json").read_text())
    assert run_record["status"] == "timed_out"


def test_terminal_json_is_stable_compact_and_json_parseable(tmp_path):
    from blog.scheduler_runner import terminal_json

    text = terminal_json({"run_id": "r", "status": "completed", "result": {"b": 2, "a": 1}})

    assert text == '{"result":{"a":1,"b":2},"run_id":"r","status":"completed"}'
    assert json.loads(text)["status"] == "completed"


def test_production_child_uses_the_single_stage_only_pipeline_seam(monkeypatch, tmp_path):
    import blog.blog_pipeline as pipeline
    from blog.scheduler_runner import run_production_child

    seen = {}

    def stage_only(**kwargs):
        seen.update(kwargs)
        return {"status": "staged", "slug": "safe-draft", "mdx_path": "/tmp/safe.mdx"}

    monkeypatch.setattr(pipeline, "run_stage_draft_only", stage_only, raising=False)

    result = run_production_child(
        stream="ai",
        repo=tmp_path,
        deadline=9999999999,
        max_new_drafts=1,
        max_images=3,
    )

    assert seen == {
        "stream": "ai",
        "repo": str(tmp_path),
        "max_new_drafts": 1,
        "max_images": 3,
        "dry_run": False,
    }
    assert result == {
        "operation": "stage_draft_only",
        "stage_only": True,
        "max_new_drafts": 1,
        "max_images": 3,
        "pipeline": {"status": "staged", "slug": "safe-draft", "mdx_path": "/tmp/safe.mdx"},
    }


@pytest.mark.parametrize("dangerous_mode", ["publish", "approval", "commit", "push", "delivery"])
def test_production_child_rejects_dangerous_modes_before_pipeline_runs(tmp_path, dangerous_mode):
    from blog.scheduler_runner import run_production_child

    with pytest.raises(ValueError, match="not permitted"):
        run_production_child(
            stream="ai",
            repo=tmp_path,
            deadline=9999999999,
            max_new_drafts=1,
            max_images=3,
            **{dangerous_mode: True},
        )


def test_controlled_dry_run_cli_verifies_paths_without_provider_execution(monkeypatch, tmp_path, capsys):
    import blog.scheduler_runner as runner

    repo = tmp_path / "repo"
    repo.mkdir()
    seen = {}

    def child(**kwargs):
        seen.update(kwargs)
        return {"operation": "stage_draft_only", "stage_only": True, "dry_run": True}

    monkeypatch.setattr(runner, "run_production_child", child, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "scheduler_runner",
            "--state-root", str(tmp_path / "state"),
            "--repo", str(repo),
            "--stream", "ai",
            "--controlled-dry-run",
        ],
    )

    assert runner._cli() == 0
    terminal = json.loads(capsys.readouterr().out)
    assert terminal["status"] == "completed"
    assert seen["dry_run"] is True
    assert seen["max_new_drafts"] == 1
    assert seen["max_images"] == 3


@pytest.mark.parametrize("dangerous_arg", ["--publish", "--approval", "--commit", "--push", "--delivery"])
def test_scheduler_cli_rejects_dangerous_mode_arguments(monkeypatch, tmp_path, dangerous_arg):
    import blog.scheduler_runner as runner

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "scheduler_runner", "--state-root", str(tmp_path / "state"),
            "--repo", str(repo), "--controlled-dry-run", dangerous_arg,
        ],
    )

    with pytest.raises(SystemExit) as exc:
        runner._cli()
    assert exc.value.code == 2
