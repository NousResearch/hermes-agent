"""Tests for the /review command engine — agent/review_engine.py.

Covers conversation snapshotting, reviewer-task composition,
auxiliary.review credential resolution, background dispatch through
delegate_task (including the internal per-call ``credentials_cfg``
override), and the shared dispatch-note formatter.
"""

import json
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
    goal, context = build_review_task(snap, "focus on security")
    assert "reviewer" in goal.lower()
    assert "[USER]" in context and "[PRIMARY AGENT]" in context
    assert "PR #99 opened" in context
    assert "focus on security" in context


def test_build_review_task_without_prompt_has_no_instruction_block():
    goal, context = build_review_task([{"role": "user", "text": "hi"}])
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
    assert "reviewer" in built["goal"].lower()

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
    from hermes_cli.main import _AUX_TASKS
    from hermes_cli.web_server import _AUX_TASK_SLOTS

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
    note = format_dispatch_note(
        {"status": "dispatched", "review_model": "opus"}, "security"
    )
    assert "dispatched on opus" in note
    assert "focus: security" in note
    assert "re-enter" in note


def test_format_dispatch_note_sync_fallback():
    note = format_dispatch_note(
        {"results": [{"summary": "fine"}], "review_model": ""}, ""
    )
    assert "synchronously" in note
