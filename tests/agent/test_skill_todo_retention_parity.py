"""Retention parity at the compaction boundary (#84718).

Compaction re-injects the todo list verbatim (``TODO_INJECTION_HEADER`` +
``TodoStore.format_for_injection``) while skill instructions are pruned down
to ``[SKILL_PRUNED: ...]`` markers. The imperative crosses the boundary; the
policy that governed it does not. These tests pin the coupling fix: when the
compressed transcript carries prune markers AND a todo snapshot is being
re-injected, the snapshot block must also carry an explicit instruction to
reload those skills before acting on the preserved tasks.

Invariants covered:

* the notice names every pruned skill with its exact ``skill_view`` call;
* skill guidance recovery now survives at least as well as the todo snapshot
  (same message, same strip lifecycle);
* the notice rides AFTER ``TODO_INJECTION_HEADER`` so the stale-snapshot
  strip removes both together — repeated boundaries never accumulate;
* deterministic and bounded — same input, same bytes; marker cap shared with
  the summary re-injection path;
* absent when nothing was pruned (zero recurring cost for clean sessions).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    _MAX_PRUNED_SKILL_MARKERS,
    _skill_pruned_marker,
)
from agent.conversation_compression import (
    _PRUNED_SKILL_RELOAD_NOTICE_HEADER,
    _pruned_skill_reload_notice,
    _strip_stale_todo_snapshot,
)
from hermes_state import SessionDB
from tools.todo_tool import TODO_INJECTION_HEADER


def _build_agent_with_db(db: SessionDB, session_id: str, platform: str = "cli"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform=platform,
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent.compression_in_place = False
    return agent


def _msgs(n=20):
    # Large enough that the fake compressor's output is a genuine shrink —
    # the no-growth commit guard refuses compressions that grow the
    # transcript (see test_compression_rotation_state.py for the same shape).
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"m{i} " + "x" * 400,
        }
        for i in range(n)
    ]


class TestPrunedSkillReloadNotice:
    """Unit contract of the notice builder."""

    def test_names_every_pruned_skill_with_reload_call(self):
        summary = (
            "[CONTEXT COMPACTION] summary\n\n## Pruned Skills\n"
            + _skill_pruned_marker("hodle-design-system")
            + "\n"
            + _skill_pruned_marker("frontend-design")
        )
        notice = _pruned_skill_reload_notice(
            [{"role": "user", "content": summary}]
        )
        assert notice.startswith(_PRUNED_SKILL_RELOAD_NOTICE_HEADER)
        assert "skill_view(name='hodle-design-system')" in notice
        assert "skill_view(name='frontend-design')" in notice

    def test_collects_markers_from_pruned_tool_rows_in_tail(self):
        # A pruned skill_view row that survived inside the protected tail
        # carries the marker in tool-role content.
        rows = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "[skill_view] name=big-skill (18000 chars) "
                + _skill_pruned_marker("big-skill"),
            },
        ]
        notice = _pruned_skill_reload_notice(rows)
        assert "skill_view(name='big-skill')" in notice

    def test_deduplicates_and_preserves_first_seen_order(self):
        marker_a = _skill_pruned_marker("alpha")
        marker_b = _skill_pruned_marker("beta")
        rows = [
            {"role": "user", "content": f"{marker_a}\n{marker_b}\n{marker_a}"},
            {"role": "tool", "content": marker_a},
        ]
        notice = _pruned_skill_reload_notice(rows)
        assert notice.count("skill_view(name='alpha')") == 1
        assert notice.index("alpha") < notice.index("beta")

    def test_empty_when_nothing_pruned(self):
        rows = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
        assert _pruned_skill_reload_notice(rows) == ""

    def test_bounded_by_shared_marker_cap(self):
        text = "\n".join(
            _skill_pruned_marker(f"skill-{i}")
            for i in range(_MAX_PRUNED_SKILL_MARKERS + 15)
        )
        notice = _pruned_skill_reload_notice([{"role": "user", "content": text}])
        assert notice.count("skill_view(name=") == _MAX_PRUNED_SKILL_MARKERS

    def test_deterministic_bytes(self):
        rows = [
            {"role": "user", "content": _skill_pruned_marker("stable-skill")}
        ]
        assert _pruned_skill_reload_notice(rows) == _pruned_skill_reload_notice(
            rows
        )

    def test_notice_does_not_feed_the_marker_extractor(self):
        """The notice must never re-trigger marker extraction on the next
        boundary — it references skills WITHOUT the canonical prefix."""
        from agent.context_compressor import _extract_pruned_skill_names

        notice = _pruned_skill_reload_notice(
            [{"role": "user", "content": _skill_pruned_marker("once")}]
        )
        assert _extract_pruned_skill_names(notice) == []


class TestSkillGuidanceSurvivesWithTodos:
    """Behavioral: skill reload guidance rides the same boundary artifact as
    the preserved todo list — retention parity, not asymmetry."""

    def _run_compaction(self, tmp_path: Path, summary_content: str):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_SKILL_TODO_PARITY"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": summary_content},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": "tail"},
        ]
        agent._todo_store._items = [
            {
                "id": "remove",
                "content": "Remove the Lightning screen",
                "status": "pending",
            }
        ]
        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )
        db.close()
        return compressed

    def test_reload_instruction_travels_with_todo_snapshot(self, tmp_path):
        summary = (
            "[CONTEXT COMPACTION] summary\n\n## Pruned Skills\n"
            + _skill_pruned_marker("hodle-design-system")
        )
        compressed = self._run_compaction(tmp_path, summary)
        tail_text = str(compressed[-1]["content"])
        assert TODO_INJECTION_HEADER in tail_text
        assert "Remove the Lightning screen" in tail_text
        # Parity: the same message that preserved the imperative carries the
        # policy-recovery instruction.
        assert _PRUNED_SKILL_RELOAD_NOTICE_HEADER in tail_text
        assert "skill_view(name='hodle-design-system')" in tail_text
        # Ordering: header first (the synthetic-row classifier keys on it),
        # notice after, inside the same strip window.
        assert tail_text.index(TODO_INJECTION_HEADER) < tail_text.index(
            _PRUNED_SKILL_RELOAD_NOTICE_HEADER
        )

    def test_no_notice_when_no_skills_pruned(self, tmp_path):
        compressed = self._run_compaction(
            tmp_path, "[CONTEXT COMPACTION] summary"
        )
        tail_text = str(compressed[-1]["content"])
        assert TODO_INJECTION_HEADER in tail_text
        assert _PRUNED_SKILL_RELOAD_NOTICE_HEADER not in tail_text

    def test_synthetic_row_classification_unbroken(self, tmp_path):
        """A snapshot+notice appended as its own row must still classify as
        compression scaffolding, never as a real user turn."""
        from agent.context_compressor import ContextCompressor
        from agent.conversation_compression import _is_real_user_message

        summary = (
            "[CONTEXT COMPACTION] summary\n\n## Pruned Skills\n"
            + _skill_pruned_marker("frontend-design")
        )
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_SKILL_TODO_SYNTH"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        # Assistant tail → snapshot cannot merge; standalone flagged row.
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": summary},
            {"role": "assistant", "content": "acknowledged"},
        ]
        agent._todo_store._items = [
            {"id": "t1", "content": "task A", "status": "pending"}
        ]
        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )
        db.close()
        snapshot_rows = [
            m
            for m in compressed
            if isinstance(m, dict)
            and TODO_INJECTION_HEADER in str(m.get("content") or "")
        ]
        assert len(snapshot_rows) == 1
        row = snapshot_rows[0]
        assert _PRUNED_SKILL_RELOAD_NOTICE_HEADER in str(row["content"])
        assert row.get("_todo_snapshot_synthetic") is True
        assert not _is_real_user_message(row)
        assert ContextCompressor._is_synthetic_compression_user_turn(row)


class TestNoticeStripLifecycle:
    """The notice is stripped with the stale snapshot — never accumulates."""

    def test_strip_removes_snapshot_and_notice_together(self):
        content = (
            "real user words\n\n"
            + TODO_INJECTION_HEADER
            + "\n- [ ] t1. old task (pending)\n\n"
            + _PRUNED_SKILL_RELOAD_NOTICE_HEADER
            + "\nreload skill_view(name='old-skill') first."
        )
        stripped = _strip_stale_todo_snapshot(content)
        assert stripped == "real user words"
        assert _PRUNED_SKILL_RELOAD_NOTICE_HEADER not in stripped

    def test_repeated_boundaries_keep_single_notice(self, tmp_path):
        """Second compaction with a tail already carrying snapshot+notice
        refreshes in place instead of stacking duplicates (#26981 parity)."""
        summary = (
            "[CONTEXT COMPACTION] summary\n\n## Pruned Skills\n"
            + _skill_pruned_marker("hodle-design-system")
        )
        stale_tail = (
            "keep this human text\n\n"
            + TODO_INJECTION_HEADER
            + "\n- [ ] t0. stale task (pending)\n\n"
            + _PRUNED_SKILL_RELOAD_NOTICE_HEADER
            + "\nstale notice body skill_view(name='stale-skill')."
        )
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_SKILL_TODO_RESTRIP"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": summary},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": stale_tail},
        ]
        agent._todo_store._items = [
            {"id": "t1", "content": "fresh task", "status": "pending"}
        ]
        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )
        db.close()
        tail_text = str(compressed[-1]["content"])
        assert tail_text.count(TODO_INJECTION_HEADER) == 1
        assert tail_text.count(_PRUNED_SKILL_RELOAD_NOTICE_HEADER) == 1
        assert "stale task" not in tail_text
        assert "stale-skill" not in tail_text
        assert "fresh task" in tail_text
        assert "skill_view(name='hodle-design-system')" in tail_text
        assert "keep this human text" in tail_text
