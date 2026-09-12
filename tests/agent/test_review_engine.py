"""Tests for the /review command engine — agent/review_engine.py.

Covers conversation snapshotting, reviewer-task composition,
auxiliary.review credential resolution, background dispatch through
delegate_task (including the internal per-call ``credentials_cfg``
override), and the shared dispatch-note formatter.
"""

import json
import subprocess
import threading
import time
from unittest.mock import MagicMock

import pytest

from agent import review_engine as re_mod
from agent.review_engine import (
    build_review_task,
    format_dispatch_note,
    snapshot_recent_messages,
    start_review,
)
from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


# ---------------------------------------------------------------------------
# git review targets
# ---------------------------------------------------------------------------

def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout


@pytest.fixture
def git_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "review@example.com")
    _git(tmp_path, "config", "user.name", "Review Test")
    (tmp_path / "tracked.txt").write_text("one\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


@pytest.mark.parametrize(
    ("text", "kind", "value", "instructions"),
    [
        ("uncommitted", "uncommitted", "", ""),
        ("uncommitted focus on tests", "uncommitted", "", "focus on tests"),
        ("base main focus on tests", "base", "main", "focus on tests"),
        ("commit deadbeef", "commit", "deadbeef", ""),
        ("uncommittedness", None, "", "uncommittedness"),
        ("review base behavior", None, "", "review base behavior"),
        ("Base main", None, "", "Base main"),
    ],
)
def test_parse_review_request_detects_only_exact_leading_selectors(
    text, kind, value, instructions,
):
    target, parsed_instructions = re_mod.parse_review_request(text)
    assert (target.kind if target else None) == kind
    assert (target.value if target else "") == value
    assert parsed_instructions == instructions


@pytest.mark.parametrize("text", ["base", "base   ", "commit", "commit   "])
def test_parse_review_request_rejects_incomplete_selector(text):
    with pytest.raises(ValueError, match="Usage: /review"):
        re_mod.parse_review_request(text)


def test_uncommitted_context_contains_staged_and_unstaged_diff(git_repo):
    (git_repo / "tracked.txt").write_text("one\nstaged\n")
    _git(git_repo, "add", "tracked.txt")
    (git_repo / "tracked.txt").write_text("one\nstaged\nunstaged\n")

    target, _ = re_mod.parse_review_request("uncommitted")
    context = re_mod.collect_git_review_context(target, git_repo)

    assert "Git diff target: uncommitted" in context
    assert "Diff stat:" in context
    assert "+staged" in context
    assert "+unstaged" in context


def test_git_diff_context_is_bounded_with_explicit_marker(monkeypatch, git_repo):
    monkeypatch.setattr(re_mod, "GIT_DIFF_CHAR_CAP", 120)
    (git_repo / "tracked.txt").write_text("one\n" + ("large line\n" * 80))

    target, _ = re_mod.parse_review_request("uncommitted")
    context = re_mod.collect_git_review_context(target, git_repo)

    assert "[... git diff truncated at 120 characters ...]" in context
    full_diff = context.split("Full diff (bounded to 120 characters):\n", 1)[1]
    assert len(full_diff) <= 120


def test_base_and_commit_targets_have_expected_scope(git_repo):
    base_sha = _git(git_repo, "rev-parse", "HEAD").strip()
    _git(git_repo, "branch", "baseline", base_sha)
    (git_repo / "tracked.txt").write_text("one\ntwo\n")
    _git(git_repo, "commit", "-qam", "second")
    second_sha = _git(git_repo, "rev-parse", "HEAD").strip()
    (git_repo / "tracked.txt").write_text("one\ntwo\nthree\n")
    _git(git_repo, "commit", "-qam", "third")

    base_target, _ = re_mod.parse_review_request("base baseline")
    base_context = re_mod.collect_git_review_context(base_target, git_repo)
    assert "+two" in base_context and "+three" in base_context

    commit_target, _ = re_mod.parse_review_request(f"commit {second_sha}")
    commit_context = re_mod.collect_git_review_context(commit_target, git_repo)
    assert "+two" in commit_context
    assert "+three" not in commit_context


@pytest.mark.parametrize("review_request", ["base missing-branch", "commit not-a-sha"])
def test_invalid_git_target_fails_open(review_request, git_repo):
    target, _ = re_mod.parse_review_request(review_request)
    with pytest.raises(ValueError, match="Unable to prepare git review"):
        re_mod.collect_git_review_context(target, git_repo)


def test_non_repository_fails_open(tmp_path):
    target, _ = re_mod.parse_review_request("uncommitted")
    with pytest.raises(ValueError, match="Unable to prepare git review"):
        re_mod.collect_git_review_context(target, tmp_path)


def test_missing_git_executable_fails_open(monkeypatch, git_repo):
    monkeypatch.setattr(
        re_mod.subprocess, "Popen", MagicMock(side_effect=FileNotFoundError),
    )
    target, _ = re_mod.parse_review_request("uncommitted")
    with pytest.raises(ValueError, match="git executable was not found"):
        re_mod.collect_git_review_context(target, git_repo)


def test_undecodable_git_output_fails_open(monkeypatch, git_repo):
    class BadOutput:
        def read(self, _size):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    process = MagicMock()
    process.stdout = BadOutput()
    process.poll.return_value = 0
    process.wait.return_value = 0
    monkeypatch.setattr(re_mod.subprocess, "Popen", MagicMock(return_value=process))
    target, _ = re_mod.parse_review_request("uncommitted")
    with pytest.raises(ValueError, match="undecodable output"):
        re_mod.collect_git_review_context(target, git_repo)


def test_clean_target_spawns_no_reviewer(monkeypatch, git_repo):
    parent = _fake_parent()
    parent.cwd = str(git_repo)
    called = False

    def should_not_dispatch(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("reviewer should not be spawned")

    monkeypatch.setattr("tools.delegate_tool.delegate_task", should_not_dispatch)
    with pytest.raises(ValueError, match="Nothing to review"):
        start_review(parent, [{"role": "user", "content": "review it"}], "uncommitted")
    assert called is False


def test_targeted_review_keeps_conversation_and_adds_diff(monkeypatch, git_repo):
    (git_repo / "tracked.txt").write_text("one\nchanged\n")
    parent = _fake_parent()
    parent.cwd = str(git_repo)
    captured = {}

    def fake_dispatch(**kwargs):
        captured.update(kwargs)
        return json.dumps({"status": "dispatched"})

    monkeypatch.setattr("tools.delegate_tool.delegate_task", fake_dispatch)
    monkeypatch.setattr(re_mod, "_load_review_credentials_cfg", lambda: None)
    result = start_review(
        parent,
        [{"role": "user", "content": "please fix the parser"}],
        "uncommitted focus on correctness",
    )

    assert result["status"] == "dispatched"
    assert "please fix the parser" in captured["context"]
    assert "+changed" in captured["context"]
    assert "focus on correctness" in captured["context"]


# ---------------------------------------------------------------------------
# snapshot_recent_messages
# ---------------------------------------------------------------------------

def test_snapshot_takes_last_ten_chat_messages_only():
    msgs = (
        [{"role": "system", "content": "sys"}]
        + [{"role": "user", "content": f"m{i}"} for i in range(15)]
        + [{"role": "tool", "content": "tool out"}]
    )
    snap = snapshot_recent_messages(msgs)
    assert len(snap) == 10
    assert snap[0]["text"] == "m5"
    assert snap[-1]["text"] == "m14"
    assert all(m["role"] == "user" for m in snap)


def test_snapshot_skips_toolcall_stub_assistant_messages():
    msgs = [
        {"role": "user", "content": "make a PR"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "created"},
        {"role": "assistant", "content": "PR #123: https://example.com/pr/123"},
    ]
    snap = snapshot_recent_messages(msgs)
    assert [m["text"] for m in snap] == [
        "make a PR",
        "PR #123: https://example.com/pr/123",
    ]


def test_snapshot_handles_multimodal_content_lists():
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ],
    }]
    snap = snapshot_recent_messages(msgs)
    assert snap[0]["text"] == "look at this\n[image_url]"


def test_snapshot_caps_oversized_messages():
    msgs = [{"role": "user", "content": "x" * 50_000}]
    snap = snapshot_recent_messages(msgs)
    assert len(snap[0]["text"]) < 13_000
    assert snap[0]["text"].endswith("[... truncated ...]")


# ---------------------------------------------------------------------------
# build_review_task
# ---------------------------------------------------------------------------

def test_build_review_task_includes_excerpt_and_prompt():
    snap = [
        {"role": "user", "text": "review my PR"},
        {"role": "assistant", "text": "PR #99 opened"},
    ]
    prompt = "focus on security\n" + "keep these instructions intact " * 20
    goal, context = build_review_task(snap, prompt)
    assert goal.startswith("Review: focus on security ")
    assert len(goal) <= 80 and "\n" not in goal
    assert goal.endswith("…")
    assert re_mod._REVIEW_GOAL in context
    assert "[USER]" in context and "[PRIMARY AGENT]" in context
    assert "PR #99 opened" in context
    assert prompt.strip() in context


def test_build_review_task_without_prompt_has_no_instruction_block():
    goal, context = build_review_task([{"role": "user", "text": "hi"}])
    assert goal == "Review recent work"
    assert re_mod._REVIEW_GOAL in context
    assert "Additional review instructions" not in context


# ---------------------------------------------------------------------------
# auxiliary.review credential resolution
# ---------------------------------------------------------------------------

def test_load_review_credentials_cfg_reads_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"auxiliary": {"review": {
            "provider": "openrouter",
            "model": "anthropic/claude-opus-4.6",
        }}},
    )
    cfg = re_mod._load_review_credentials_cfg()
    assert cfg == {
        "provider": "openrouter",
        "model": "anthropic/claude-opus-4.6",
        "base_url": "",
        "api_key": "",
        "api_mode": "",
    }


def test_load_review_credentials_cfg_auto_means_inherit(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"auxiliary": {"review": {"provider": "auto", "model": ""}}},
    )
    assert re_mod._load_review_credentials_cfg() is None


def test_load_review_credentials_cfg_missing_section(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: {"auxiliary": {}}
    )
    assert re_mod._load_review_credentials_cfg() is None


# ---------------------------------------------------------------------------
# delegate_task credentials_cfg override (the internal /review routing hook)
# ---------------------------------------------------------------------------

def _fake_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "review-parent-sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    return parent


def test_delegate_task_credentials_cfg_overrides_delegation_config(monkeypatch):
    """The per-call credentials_cfg dict must reach the credential resolver
    instead of the global delegation config section."""
    import tools.delegate_tool as dt

    seen = {}

    def fake_resolve(cfg, parent_agent):
        seen["cfg"] = cfg
        return {
            "model": cfg.get("model"), "provider": None, "base_url": None,
            "api_key": None, "api_mode": None, "command": None, "args": None,
        }

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", fake_resolve)
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(
        dt, "_run_single_child",
        lambda *a, **k: {
            "task_index": 0, "status": "completed", "summary": "ok",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        },
    )

    override = {"provider": "openrouter", "model": "review-model-x"}
    out = dt.delegate_task(
        goal="review this",
        background=True,
        parent_agent=_fake_parent(),
        credentials_cfg=override,
    )
    parsed = json.loads(out)
    assert parsed["status"] == "dispatched"
    assert seen["cfg"] == override


# ---------------------------------------------------------------------------
# start_review end-to-end through the async delegation rail
# ---------------------------------------------------------------------------

def test_start_review_dispatches_background_and_completes(monkeypatch):
    import tools.delegate_tool as dt

    captured = {}

    def fake_run_single_child(task_index, goal, child=None, parent_agent=None, **kw):
        captured["goal"] = goal
        return {
            "task_index": 0, "status": "completed",
            "summary": "REVIEW: looks good", "api_calls": 2,
            "duration_seconds": 0.1, "model": "m", "exit_reason": "completed",
        }

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    built = {}

    def fake_build(**kw):
        built.update(kw)
        return fake_child

    monkeypatch.setattr(dt, "_build_child_agent", fake_build)
    monkeypatch.setattr(dt, "_run_single_child", fake_run_single_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(re_mod, "_load_review_credentials_cfg", lambda: None)

    msgs = [
        {"role": "user", "content": "open a PR for the fix"},
        {"role": "assistant", "content": "PR #77 opened: https://x/pull/77"},
    ]
    result = start_review(_fake_parent(), msgs, "check the tests")
    assert result["status"] == "dispatched"

    # The reviewer briefing carries the conversation excerpt + user prompt.
    assert "PR #77 opened" in built["context"]
    assert "check the tests" in built["context"]
    assert built["goal"].startswith("Review: ")
    assert re_mod._REVIEW_GOAL in built["context"]

    # The completion re-enters via the shared queue like any subagent.
    deadline = time.monotonic() + 5.0
    evt = None
    while time.monotonic() < deadline:
        try:
            evt = process_registry.completion_queue.get(timeout=0.2)
            break
        except Exception:
            continue
    assert evt is not None and evt["type"] == "async_delegation"
    assert evt["results"][0]["summary"] == "REVIEW: looks good"


def test_start_review_rejects_empty_conversation():
    with pytest.raises(ValueError, match="empty"):
        start_review(_fake_parent(), [], "")


def test_start_review_requires_agent():
    with pytest.raises(ValueError, match="No active agent"):
        start_review(None, [{"role": "user", "content": "x"}], "")


# ---------------------------------------------------------------------------
# collect_parent_loaded_skills — reviewer inherits the parent's working skills
# ---------------------------------------------------------------------------

def test_collect_skills_from_preloaded_prompt_and_history():
    from agent.review_engine import collect_parent_loaded_skills

    parent = MagicMock()
    parent.ephemeral_system_prompt = (
        '[IMPORTANT: The user launched this CLI session with the '
        '"hermes-agent-dev" skill preloaded. Treat its instructions as '
        'active guidance for the duration of this session unless the user '
        'overrides them.]'
    )
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "skill_view",
                          "arguments": '{"name": "github-pr-workflow"}'}},
            # reference-file read of an already-counted skill: skipped
            {"function": {"name": "skill_view",
                          "arguments": '{"name": "hermes-agent-dev", '
                                       '"file_path": "references/x.md"}'}},
            {"function": {"name": "read_file",
                          "arguments": '{"path": "/tmp/x"}'}},
        ]},
        # duplicate load coalesces
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "skill_view",
                          "arguments": '{"name": "github-pr-workflow"}'}},
        ]},
    ]
    names = collect_parent_loaded_skills(parent, msgs)
    assert names == ["hermes-agent-dev", "github-pr-workflow"]


def test_collect_skills_empty_when_none_loaded():
    from agent.review_engine import collect_parent_loaded_skills

    parent = MagicMock()
    parent.ephemeral_system_prompt = None
    assert collect_parent_loaded_skills(parent, [
        {"role": "user", "content": "hi"},
    ]) == []


def test_collect_skills_caps_at_limit():
    from agent.review_engine import collect_parent_loaded_skills

    parent = MagicMock()
    parent.ephemeral_system_prompt = ""
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "skill_view",
                          "arguments": json.dumps({"name": f"skill-{i}"})}}
        ]}
        for i in range(15)
    ]
    assert len(collect_parent_loaded_skills(parent, msgs)) == 8


def test_briefing_includes_loaded_skills_instruction():
    snap = [{"role": "user", "text": "review my PR"}]
    _, context = build_review_task(snap, "", ["hermes-agent-dev", "xitter"])
    assert "hermes-agent-dev, xitter" in context
    assert "skill_view" in context
    assert "binding" in context


def test_briefing_omits_skills_block_when_none():
    _, context = build_review_task([{"role": "user", "text": "hi"}], "")
    assert "operating under these loaded skills" not in context


def test_start_review_threads_loaded_skills_into_context(monkeypatch):
    import tools.delegate_tool as dt

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    built = {}

    def fake_build(**kw):
        built.update(kw)
        return fake_child

    monkeypatch.setattr(dt, "_build_child_agent", fake_build)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(
        dt, "_run_single_child",
        lambda *a, **k: {
            "task_index": 0, "status": "completed", "summary": "ok",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        },
    )
    monkeypatch.setattr(re_mod, "_load_review_credentials_cfg", lambda: None)

    parent = _fake_parent()
    parent.ephemeral_system_prompt = (
        'session with the "hermes-agent-dev" skill preloaded.'
    )
    msgs = [
        {"role": "user", "content": "open a PR"},
        {"role": "assistant", "content": "PR #9 opened"},
    ]
    result = start_review(parent, msgs, "")
    assert result["status"] == "dispatched"
    assert "hermes-agent-dev" in built["context"]
    assert "skill_view" in built["context"]


# ---------------------------------------------------------------------------
# Workspace context files — ALL subagents (reviewer included) get AGENTS.md
# et al. in their child system prompt (tools/delegate_tool.py)
# ---------------------------------------------------------------------------

def test_child_system_prompt_embeds_workspace_context(tmp_path):
    """Real file I/O through the same loader the main system prompt uses."""
    from tools.delegate_tool import _build_child_system_prompt

    workspace = tmp_path / "proj"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text(
        "# Project Rules\nAll fixes must cover sibling call sites.\n"
    )
    prompt = _build_child_system_prompt(
        "do the thing", None, workspace_path=str(workspace)
    )
    assert "sibling call sites" in prompt
    assert "binding for your work" in prompt


def test_child_system_prompt_no_context_block_without_files(tmp_path):
    from tools.delegate_tool import _build_child_system_prompt

    workspace = tmp_path / "empty"
    workspace.mkdir()
    prompt = _build_child_system_prompt(
        "do the thing", None, workspace_path=str(workspace)
    )
    assert "project context files" not in prompt


def test_child_system_prompt_no_workspace_no_block():
    from tools.delegate_tool import _build_child_system_prompt

    prompt = _build_child_system_prompt("do the thing", None, workspace_path=None)
    assert "project context files" not in prompt


def test_review_child_gets_workspace_context_via_dispatch(monkeypatch, tmp_path):
    """E2E through start_review: the reviewer child's *system prompt* carries
    the workspace AGENTS.md (inherited from the generalized subagent path)."""
    import tools.delegate_tool as dt

    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("# Rules\nNever break prompt caching.\n")
    monkeypatch.setattr(dt, "_resolve_workspace_hint", lambda parent: str(workspace))

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    built = {}

    real_build_prompt = dt._build_child_system_prompt

    def fake_build(**kw):
        built.update(kw)
        # Reproduce what _build_child_agent does with the real prompt builder
        built["child_prompt"] = real_build_prompt(
            kw.get("goal") or "", kw.get("context"),
            workspace_path=dt._resolve_workspace_hint(kw.get("parent_agent")),
        )
        return fake_child

    monkeypatch.setattr(dt, "_build_child_agent", fake_build)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(
        dt, "_run_single_child",
        lambda *a, **k: {
            "task_index": 0, "status": "completed", "summary": "ok",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        },
    )
    monkeypatch.setattr(re_mod, "_load_review_credentials_cfg", lambda: None)

    result = start_review(_fake_parent(), [
        {"role": "user", "content": "open a PR"},
        {"role": "assistant", "content": "PR #4 opened"},
    ], "")
    assert result["status"] == "dispatched"
    assert "Never break prompt caching." in built["child_prompt"]


# ---------------------------------------------------------------------------
# Registry sync: `review` must be a first-class slot in every aux-task surface
# ---------------------------------------------------------------------------

def test_review_registered_in_every_aux_surface():
    """The /review slot must appear in every aux-model picker registry.

    Same contract as curator's registry test in tests/agent/test_curator.py:
    DEFAULT_CONFIG schema, CLI picker (_AUX_TASKS), and dashboard REST
    allowlist (_AUX_TASK_SLOTS). The desktop and web AUX_TASKS tsx arrays
    mirror _AUX_TASK_SLOTS by convention (shared "Must match" comments).
    """
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.main_provider_setup import _AUX_TASKS
    from hermes_cli.web_server_config import _AUX_TASK_SLOTS

    assert "review" in DEFAULT_CONFIG["auxiliary"], \
        "review missing from DEFAULT_CONFIG['auxiliary']"
    slot = DEFAULT_CONFIG["auxiliary"]["review"]
    assert slot["provider"] == "auto"
    assert slot["model"] == ""

    aux_keys = {k for k, _name, _desc in _AUX_TASKS}
    assert "review" in aux_keys, "review missing from _AUX_TASKS (CLI picker)"

    assert "review" in _AUX_TASK_SLOTS, \
        "review missing from _AUX_TASK_SLOTS (dashboard REST API)"


# ---------------------------------------------------------------------------
# format_dispatch_note
# ---------------------------------------------------------------------------

def test_format_dispatch_note_dispatched():
    prompt = "security\n" * 100
    note = format_dispatch_note(
        {"status": "dispatched", "review_model": "opus"}, prompt
    )
    assert "Review started" in note and "return here" in note
    assert "security" not in note and "\n" not in note
    assert len(note) < 80


def test_format_dispatch_note_sync_fallback():
    note = format_dispatch_note(
        {"results": [{"summary": "fine"}], "review_model": ""}, ""
    )
    assert "synchronously" in note
