"""Tests for the skill/memory learning-loop pieces.

Covers:
- machine-extracted evidence digest for background review
- references/lessons.md overlay guard for non-agent skills
- skill_view auto-injection of lessons.md
- fact_store notification summarization
"""

from __future__ import annotations

import json
from unittest.mock import patch

from agent.background_review import (
    build_review_evidence_digest,
    summarize_background_review_actions,
)


# ---------------------------------------------------------------------------
# Evidence digest
# ---------------------------------------------------------------------------


def test_evidence_digest_extracts_skill_view_failures_and_corrections():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "skill_view",
                        "arguments": json.dumps({"name": "amazon-shopping"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": json.dumps({"success": True, "name": "amazon-shopping"}),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c2",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": "curl ..."}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c2",
            "content": json.dumps({"error": "401 Unauthorized"}),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c3",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": "curl -H Bearer ..."}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c3",
            "content": json.dumps({"success": True, "output": "ok"}),
        },
        {
            "role": "user",
            "content": "Stop doing that — remember this: always send the API token.",
        },
    ]
    digest = build_review_evidence_digest(messages)
    assert "amazon-shopping" in digest
    assert "401" in digest or "Unauthorized" in digest
    assert "succeeded" in digest.lower()
    assert "remember this" in digest.lower() or "Stop doing" in digest
    assert "MUST persist" in digest or "decision ladder" in digest


def test_evidence_digest_empty_transcript_has_noop_guidance():
    digest = build_review_evidence_digest([{"role": "user", "content": "hi"}])
    assert "Nothing to save" in digest
    assert "No skill loads" in digest or "no skill" in digest.lower()


# ---------------------------------------------------------------------------
# Lessons overlay guard
# ---------------------------------------------------------------------------


def test_background_review_allows_lessons_overlay_for_manual_skill(tmp_path, monkeypatch):
    from tools.skill_manager_tool import (
        _background_review_write_guard,
        skill_manage,
    )
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )

    hermes_home = tmp_path / ".hermes"
    skills_root = hermes_home / "skills"
    skill_dir = skills_root / "manual-skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: manual-skill\ndescription: Manual test skill.\n---\n\n# Manual\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with patch("tools.skill_manager_tool.SKILLS_DIR", skills_root), patch(
        "tools.skills_tool.SKILLS_DIR", skills_root
    ), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[skills_root]
    ), patch(
        "tools.skill_usage.load_usage",
        return_value={"manual-skill": {"created_by": None, "use_count": 10}},
    ), patch(
        "tools.skill_usage.get_record",
        side_effect=lambda n: (
            {"created_by": None, "use_count": 10, "pinned": False}
            if n == "manual-skill"
            else {}
        ),
    ), patch(
        "tools.skill_provenance.is_background_review", return_value=True
    ):
        # Body patch still refused
        body_guard = _background_review_write_guard(
            "manual-skill", skill_dir, "patch", file_path=None
        )
        assert body_guard is not None
        assert "manually authored" in body_guard["error"].lower()

        # Lessons overlay allowed at guard level
        overlay_guard = _background_review_write_guard(
            "manual-skill",
            skill_dir,
            "write_file",
            file_path="references/lessons.md",
        )
        assert overlay_guard is None

        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            # New file: no prior read required
            raw = skill_manage(
                action="write_file",
                name="manual-skill",
                file_path="references/lessons.md",
                file_content="- Always pass CRAWL4AI_API_TOKEN (401 without it).\n",
            )
        finally:
            reset_current_write_origin(token)

    result = json.loads(raw)
    assert result["success"] is True, result
    lessons = skill_dir / "references" / "lessons.md"
    assert lessons.exists()
    assert "CRAWL4AI_API_TOKEN" in lessons.read_text(encoding="utf-8")


def test_background_review_still_blocks_manual_skill_body_edit(tmp_path, monkeypatch):
    from tools.skill_manager_tool import skill_manage, mark_background_review_skill_read
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )

    hermes_home = tmp_path / ".hermes"
    skills_root = hermes_home / "skills"
    skill_dir = skills_root / "manual-skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: manual-skill\ndescription: Manual test skill.\n---\n\n# Manual\nStep 1.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with patch("tools.skill_manager_tool.SKILLS_DIR", skills_root), patch(
        "tools.skills_tool.SKILLS_DIR", skills_root
    ), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[skills_root]
    ), patch(
        "tools.skill_usage.load_usage",
        return_value={"manual-skill": {"created_by": None}},
    ), patch(
        "tools.skill_usage.get_record",
        return_value={"created_by": None, "pinned": False},
    ):
        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            mark_background_review_skill_read(skill_md)
            raw = skill_manage(
                action="patch",
                name="manual-skill",
                old_string="Step 1.",
                new_string="Step 1 (fixed).",
            )
        finally:
            reset_current_write_origin(token)

    result = json.loads(raw)
    assert result["success"] is False
    assert "manually authored" in result["error"].lower()
    assert "lessons.md" in result["error"]


# ---------------------------------------------------------------------------
# skill_view injection
# ---------------------------------------------------------------------------


def test_skill_view_injects_lessons_overlay(tmp_path, monkeypatch):
    from tools.skills_tool import skill_view, _append_lessons_overlay

    hermes_home = tmp_path / ".hermes"
    skills_root = hermes_home / "skills"
    skill_dir = skills_root / "demo-skill"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo.\n---\n\n# Demo\nDo the thing.\n",
        encoding="utf-8",
    )
    (skill_dir / "references" / "lessons.md").write_text(
        "- Prefer brightness_pct over raw brightness.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Unit the helper directly
    out = _append_lessons_overlay("# Demo\n", skill_dir)
    assert "Learned corrections (auto-loaded)" in out
    assert "brightness_pct" in out

    with patch("tools.skills_tool.SKILLS_DIR", skills_root), patch(
        "tools.skills_tool._skills_dir", return_value=skills_root
    ), patch(
        "agent.skill_utils.get_external_skills_dirs", return_value=[]
    ), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[skills_root]
    ):
        raw = skill_view("demo-skill", preprocess=False)
    data = json.loads(raw)
    assert data["success"] is True, data
    assert "Learned corrections (auto-loaded)" in data["content"]
    assert "brightness_pct" in data["content"]


def test_append_lessons_overlay_truncates_long_file(tmp_path):
    from tools.skills_tool import _append_lessons_overlay, _LESSONS_OVERLAY_MAX_CHARS

    skill_dir = tmp_path / "s"
    (skill_dir / "references").mkdir(parents=True)
    big = "A" * (_LESSONS_OVERLAY_MAX_CHARS + 500)
    (skill_dir / "references" / "lessons.md").write_text(big, encoding="utf-8")
    out = _append_lessons_overlay("# Title\n", skill_dir)
    assert "truncated" in out
    assert len(out) < len(big) + 200


# ---------------------------------------------------------------------------
# fact_store review notifications
# ---------------------------------------------------------------------------


def test_summarize_includes_fact_store_adds():
    review_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "f1",
                    "function": {
                        "name": "fact_store",
                        "arguments": json.dumps(
                            {
                                "action": "add",
                                "content": "HA brightness_pct tip",
                                "category": "tool",
                                "tags": "lesson,skill:home-assistant,validated",
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "f1",
            "content": json.dumps({"fact_id": 51, "status": "added"}),
        },
    ]
    actions = summarize_background_review_actions(review_messages, prior_snapshot=[])
    assert any("Fact store" in a for a in actions)


def test_background_review_whitelist_includes_fact_store_when_parent_has_it():
    """Review whitelist must allow fact_store when the parent memory manager has it."""
    import run_agent
    from hermes_cli import plugins as _plugins

    class _MM:
        def has_tool(self, name):
            return name in {"fact_store", "fact_feedback"}

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        raise RuntimeError("stop after capturing whitelist")

    agent = object.__new__(run_agent.AIAgent)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.session_id = "sess-123"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_manager = _MM()
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._cached_system_prompt = None
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.enabled_toolsets = ["memory", "skills"]
    agent.disabled_toolsets = None

    class _SyncThread:
        def __init__(self, *, target=None, daemon=None, name=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    def _no_init(self, *args, **kwargs):
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), patch.object(
        _plugins, "set_thread_tool_whitelist", _capture_whitelist
    ), patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=True,
        )

    assert "fact_store" in captured["whitelist"]
    assert "fact_feedback" in captured["whitelist"]
