"""Tests for the durable `/loop` V1 engine and global `/view` helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command
from hermes_cli.loops import (
    block_story,
    build_story_execution_prompt,
    close_loop,
    complete_story,
    create_loop,
    generate_plan_scaffolds,
    handle_loop_command,
    handle_view_command,
    index_docs,
    list_loops,
    render_global_view,
    render_loop_status,
    render_review_packet,
    resolve_loop,
    run_next_story,
    slugify,
    validate_story_manifest,
)


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _plan(tmp_path: Path):
    home = tmp_path / "home"
    result = create_loop("Engine Loop", cwd=tmp_path, hermes_home=home)
    generate_plan_scaffolds(cwd=tmp_path, hermes_home=home)
    return home, result["path"]


def test_slugify_normalizes_names():
    assert slugify("Memory / Routing Work!") == "memory-routing-work"
    assert slugify("!!!") == "loop"


def test_create_loop_uses_profile_home_outside_git(tmp_path):
    hermes_home = tmp_path / "hermes-home"
    work = tmp_path / "not-git"
    work.mkdir()

    result = create_loop("Profile Loop", cwd=work, hermes_home=hermes_home)

    assert result["created"] is True
    assert result["path"] == hermes_home / "loops" / "profile-loop"
    assert (result["path"] / "loop.json").exists()
    assert (hermes_home / "loops" / ".active").read_text().strip() == "profile-loop"
    assert result["state"]["scope"] == "profile"


def test_create_loop_uses_repo_local_state_inside_git(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    result = create_loop("Repo Bound Loop", cwd=repo, hermes_home=tmp_path / "home")

    assert result["path"] == repo / ".hermes" / "loops" / "repo-bound-loop"
    assert result["state"]["scope"] == "repo"
    assert (repo / ".hermes" / "loops" / ".gitignore").exists()
    assert "loop.json" in (repo / ".hermes" / "loops" / ".gitignore").read_text()


def test_create_loop_is_idempotent_for_same_slug(tmp_path):
    home = tmp_path / "home"
    first = create_loop("Same Loop", cwd=tmp_path, hermes_home=home)
    second = create_loop("Same Loop", cwd=tmp_path, hermes_home=home)

    assert first["created"] is True
    assert second["created"] is False
    assert first["path"] == second["path"]
    assert len(list_loops(tmp_path, hermes_home=home)) == 1


def test_docs_add_skips_outside_and_secret_like_explicit_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)
    home = tmp_path / "home"
    loop_path = create_loop("Docs Safety", cwd=repo, hermes_home=home)["path"]
    docs = repo / "docs"
    docs.mkdir()
    good = docs / "guide.md"
    good.write_text("# Safe Guide\nUseful project docs.", encoding="utf-8")
    ssh_dir = repo / ".ssh"
    ssh_dir.mkdir()
    secretish = ssh_dir / "id_rsa.txt"
    secretish.write_text("-----BEGIN OPENSSH PRIVATE KEY-----", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("# External Secret Context", encoding="utf-8")

    message = index_docs([str(good), str(secretish), str(outside)], cwd=repo, hermes_home=home)
    docs_md = (loop_path / "docs.md").read_text(encoding="utf-8")

    assert "Indexed 1 docs" in message
    assert "Safe Guide" in docs_md
    assert "OPENSSH PRIVATE KEY" not in docs_md
    assert "External Secret Context" not in docs_md


def test_loop_status_is_read_only_and_reports_counts(tmp_path):
    home = tmp_path / "home"
    result = create_loop("Story Loop", cwd=tmp_path, hermes_home=home)
    stories = [
        {"id": "S1", "title": "Done story", "status": "done"},
        {"id": "S2", "title": "Blocked story", "status": "blocked", "blocked_reason": "Needs API choice"},
        {"id": "S3", "title": "Todo story", "status": "todo"},
    ]
    (result["path"] / "stories.json").write_text(json.dumps({"stories": stories}), encoding="utf-8")
    before = (result["path"] / "stories.json").read_text(encoding="utf-8")

    status = render_loop_status(cwd=tmp_path, hermes_home=home)

    assert "Loop: Story Loop" in status
    assert "Stories: todo:1, blocked:1, done:1" in status
    assert "Next:" in status
    # status must not mutate the manifest
    assert (result["path"] / "stories.json").read_text(encoding="utf-8") == before


def test_plan_seeds_prd_and_story_manifest(tmp_path):
    home, loop_path = _plan(tmp_path)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))
    state = json.loads((loop_path / "loop.json").read_text(encoding="utf-8"))

    assert (loop_path / "prd.md").exists()
    assert manifest["version"] == 1
    assert [s["id"] for s in manifest["stories"]] == ["S1", "S2", "S3"]
    assert validate_story_manifest(manifest) == []
    assert state["phase"] == "planning"


def test_run_next_selects_one_story_and_writes_run_artifacts(tmp_path):
    home, loop_path = _plan(tmp_path)

    message = run_next_story(cwd=tmp_path, hermes_home=home)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))

    assert "Story execution prompt — S1" in message
    assert manifest["stories"][0]["status"] == "running"
    assert manifest["stories"][0]["started_at"]
    # durable run artifacts
    run_dir = loop_path / "runs" / "s1"
    assert (run_dir / "prompt.md").exists()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["story_id"] == "S1"
    assert run["status"] == "running"
    assert run["prompt_path"].endswith("prompt.md")


def test_run_next_refuses_when_a_story_is_already_running(tmp_path):
    home, loop_path = _plan(tmp_path)
    run_next_story(cwd=tmp_path, hermes_home=home)

    message = run_next_story(cwd=tmp_path, hermes_home=home)

    assert "already running" in message
    assert "S1" in message


def test_run_next_respects_dependencies(tmp_path):
    home, loop_path = _plan(tmp_path)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))
    manifest["stories"][0]["status"] = "done"
    (loop_path / "stories.json").write_text(json.dumps(manifest), encoding="utf-8")

    message = run_next_story(cwd=tmp_path, hermes_home=home)

    assert "Story execution prompt — S2" in message


def test_complete_refuses_missing_evidence(tmp_path):
    home, loop_path = _plan(tmp_path)
    run_next_story(cwd=tmp_path, hermes_home=home)

    message = complete_story("S1", "does/not/exist.txt", cwd=tmp_path, hermes_home=home)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))

    assert "Refusing to complete S1" in message
    assert "not found" in message
    assert manifest["stories"][0]["status"] == "running"


def test_complete_refuses_story_that_was_not_started(tmp_path):
    home, loop_path = _plan(tmp_path)
    evidence = loop_path / "evidence.txt"
    evidence.write_text("verified", encoding="utf-8")

    message = complete_story("S2", str(evidence), cwd=tmp_path, hermes_home=home)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))

    assert "Refusing to complete S2" in message
    assert "status is todo" in message
    assert manifest["stories"][1]["status"] == "todo"


def test_complete_refuses_evidence_outside_boundary(tmp_path):
    home, loop_path = _plan(tmp_path)
    run_next_story(cwd=tmp_path, hermes_home=home)
    outside = tmp_path.parent / "outside-evidence.txt"
    outside.write_text("external", encoding="utf-8")

    message = complete_story("S1", str(outside), cwd=tmp_path, hermes_home=home)

    assert "Refusing to complete S1" in message
    assert "outside" in message


def test_complete_refuses_loop_scaffold_as_evidence(tmp_path):
    home, loop_path = _plan(tmp_path)
    run_next_story(cwd=tmp_path, hermes_home=home)

    for evidence in (loop_path / "prd.md", loop_path / "runs" / "s1" / "prompt.md"):
        message = complete_story("S1", str(evidence), cwd=tmp_path, hermes_home=home)
        manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))
        assert "Refusing to complete S1" in message
        assert "bookkeeping" in message
        assert manifest["stories"][0]["status"] == "running"


def test_complete_refuses_directory_evidence(tmp_path):
    home, loop_path = _plan(tmp_path)
    run_next_story(cwd=tmp_path, hermes_home=home)

    message = complete_story("S1", str(loop_path), cwd=tmp_path, hermes_home=home)

    assert "Refusing to complete S1" in message
    assert "not a file" in message


def test_complete_succeeds_with_existing_evidence(tmp_path):
    home, loop_path = _plan(tmp_path)
    run_next_story(cwd=tmp_path, hermes_home=home)
    evidence = loop_path / "runs" / "s1" / "test-output.txt"
    evidence.write_text("3 passed", encoding="utf-8")

    message = complete_story("S1", str(evidence), cwd=tmp_path, hermes_home=home)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))
    run = json.loads((loop_path / "runs" / "s1" / "run.json").read_text(encoding="utf-8"))

    assert "Story S1 marked done" in message
    assert manifest["stories"][0]["status"] == "done"
    assert manifest["stories"][0]["evidence"]
    assert run["status"] == "done"
    assert run["verdict"] == "pass"


def test_complete_with_review_required_moves_to_needs_review(tmp_path):
    home, loop_path = _plan(tmp_path)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))
    manifest["stories"][0]["review_required"] = True
    (loop_path / "stories.json").write_text(json.dumps(manifest), encoding="utf-8")
    run_next_story(cwd=tmp_path, hermes_home=home)
    evidence = loop_path / "evidence.txt"
    evidence.write_text("diff summary", encoding="utf-8")

    message = complete_story("S1", str(evidence), cwd=tmp_path, hermes_home=home)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))

    assert "needs_review" in message
    assert manifest["stories"][0]["status"] == "needs_review"


def test_block_records_explicit_blocker(tmp_path):
    home, loop_path = _plan(tmp_path)
    run_next_story(cwd=tmp_path, hermes_home=home)

    message = block_story("S1", "Waiting on API key", cwd=tmp_path, hermes_home=home)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))
    state = json.loads((loop_path / "loop.json").read_text(encoding="utf-8"))

    assert "Blocked S1" in message
    assert manifest["stories"][0]["status"] == "blocked"
    assert manifest["stories"][0]["blocked_reason"] == "Waiting on API key"
    assert any("S1" in b for b in state["blockers"])


def test_review_renders_packet_without_mutating_stories(tmp_path):
    home, loop_path = _plan(tmp_path)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))
    manifest["stories"][0]["review_required"] = True
    (loop_path / "stories.json").write_text(json.dumps(manifest), encoding="utf-8")
    run_next_story(cwd=tmp_path, hermes_home=home)
    evidence = loop_path / "evidence.txt"
    evidence.write_text("packet", encoding="utf-8")
    complete_story("S1", str(evidence), cwd=tmp_path, hermes_home=home)
    before = (loop_path / "stories.json").read_text(encoding="utf-8")

    packet = render_review_packet(cwd=tmp_path, hermes_home=home)

    assert "Review packet" in packet
    assert "S1" in packet
    assert "read-only" in packet
    assert (loop_path / "stories.json").read_text(encoding="utf-8") == before


def test_close_refuses_when_runnable_stories_remain(tmp_path):
    home, loop_path = _plan(tmp_path)

    message = close_loop(cwd=tmp_path, hermes_home=home)

    assert "Refusing clean close" in message
    assert "S1" in message


def test_close_writes_closeout_when_complete(tmp_path):
    home, loop_path = _plan(tmp_path)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))
    for story in manifest["stories"]:
        story["status"] = "done"
    (loop_path / "stories.json").write_text(json.dumps(manifest), encoding="utf-8")

    message = handle_loop_command("/loop close", cwd=tmp_path, hermes_home=home)
    closeout = (loop_path / "closeout.md").read_text(encoding="utf-8")
    state = json.loads((loop_path / "loop.json").read_text(encoding="utf-8"))

    assert "Closed loop: Engine Loop" in message
    assert "Acceptance" in closeout
    assert "Blocked / deferred" in closeout
    assert state["status"] == "closed"


def test_closed_loop_refuses_run_complete_block(tmp_path):
    home, loop_path = _plan(tmp_path)
    close_loop(cwd=tmp_path, hermes_home=home, force=True)

    assert "closed" in run_next_story(cwd=tmp_path, hermes_home=home).lower()
    assert "closed" in complete_story("S1", "x", cwd=tmp_path, hermes_home=home).lower()
    assert "closed" in block_story("S1", "reason", cwd=tmp_path, hermes_home=home).lower()


def test_start_reopens_closed_loop(tmp_path):
    home, loop_path = _plan(tmp_path)
    close_loop(cwd=tmp_path, hermes_home=home, force=True)

    reopened = create_loop("Engine Loop", cwd=tmp_path, hermes_home=home)
    state = json.loads((loop_path / "loop.json").read_text(encoding="utf-8"))

    assert reopened["created"] is False
    assert state["status"] == "planning"
    assert state["phase"] == "reopened"
    assert "closed_at" not in state
    assert state["last_closed_at"]
    message = run_next_story(cwd=tmp_path, hermes_home=home)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))
    assert "This loop is closed" not in message
    assert manifest["stories"][0]["status"] == "running"


def test_no_active_loop_refusals_are_human_readable(tmp_path):
    home = tmp_path / "home"
    work = tmp_path / "nowhere"
    work.mkdir()
    assert "No active loop" in run_next_story(cwd=work, hermes_home=home)
    assert "No active loop" in complete_story("S1", "x", cwd=work, hermes_home=home)
    assert "No active loop" in render_review_packet(cwd=work, hermes_home=home)


def test_build_story_execution_prompt_requires_evidence_not_assertion():
    prompt = build_story_execution_prompt(
        Path("/tmp/loop"),
        {"name": "Prompt Loop", "goal": "Ship it"},
        {"id": "S9", "title": "Check", "objective": "Verify", "acceptance": ["Done"], "verification": ["pytest"]},
        prd="PRD body",
        docs="Docs body",
        decisions="Decision body",
    )

    assert "S9 — Check" in prompt
    assert "pytest" in prompt
    assert "model assertion alone is not evidence" in prompt


def test_review_required_story_can_be_approved_after_packet_review(tmp_path):
    home, loop_path = _plan(tmp_path)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))
    manifest["stories"][0]["review_required"] = True
    (loop_path / "stories.json").write_text(json.dumps(manifest), encoding="utf-8")
    run_next_story(cwd=tmp_path, hermes_home=home)
    evidence = loop_path / "evidence.txt"
    evidence.write_text("diff summary", encoding="utf-8")
    complete_story("S1", str(evidence), cwd=tmp_path, hermes_home=home)
    approval = loop_path / "review-approval.txt"
    approval.write_text("review approved", encoding="utf-8")

    message = complete_story("S1", str(approval), cwd=tmp_path, hermes_home=home)
    manifest = json.loads((loop_path / "stories.json").read_text(encoding="utf-8"))

    assert "after review approval" in message
    assert manifest["stories"][0]["status"] == "done"
    assert manifest["stories"][0]["review_approved_at"]


# ---------------------------------------------------------------------------
# Command-registry / view integration
# ---------------------------------------------------------------------------

def test_registry_recognizes_loop_and_subcommands():
    cmd = resolve_command("loop")
    assert cmd is not None
    assert cmd.name == "loop"
    for sub in ("start", "status", "plan", "run", "complete", "block", "review", "close"):
        assert sub in cmd.subcommands
    # gateway-dispatchable (not cli_only)
    assert "loop" in GATEWAY_KNOWN_COMMANDS


def test_handle_loop_command_usage_and_routing(tmp_path):
    home = tmp_path / "home"
    assert "Usage:" in handle_loop_command("/loop", cwd=tmp_path, hermes_home=home)
    started = handle_loop_command("/loop start Ship the thing", cwd=tmp_path, hermes_home=home)
    assert "loop: ship the thing" in started.lower()


def test_global_view_falls_back_without_active_loop(tmp_path):
    repo = tmp_path / "repo"
    plans = repo / "docs" / "plans"
    plans.mkdir(parents=True)
    _git_init(repo)
    (plans / "2026-05-20-plan.md").write_text("# Plan", encoding="utf-8")

    view = handle_view_command("/view", cwd=repo, hermes_home=tmp_path / "home")

    assert "## Aim" in view
    assert "Repository: repo" in view
    assert "No active loop state found" in view


def test_no_post_turn_loop_continuation_hook_in_core_surfaces():
    """V1 /loop must not add an unattended post-turn continuation like /goal."""
    root = Path(__file__).resolve().parents[2]
    for rel in ("cli.py", "gateway/run.py", "tui_gateway/server.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "_maybe_continue_loop" not in text
        assert "continue_loop_after_turn" not in text
