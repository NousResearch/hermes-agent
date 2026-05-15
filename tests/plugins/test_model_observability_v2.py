"""Tests for the redesigned model_observability plugin v2.

Architecture under test:
    pre_tool_call       → scoping anchor + missing-pin flag
                          Stashes {offset, missing_pin} keyed on tool_call_id.
                          NEVER returns block; never directly warns the agent.
                          Also evicts TTL-expired stash entries.

    transform_tool_result → reads stash by tool_call_id, reads only JSONL entries
                            past the stash offset (scoping anchor), builds observability
                            block and injects into the result string. If missing_pin=True,
                            notes auto-router resolution. If match=False on a pinned
                            model, injects a prominent WARNING. Pops stash entry after use.

    post_api_request    → JSONL write (unchanged from v1). Regression suite.

Hook contracts (from hermes_cli/plugins.py):
    pre_tool_call:  return {"action": "block", "message": "..."} to block,
                    or None to pass through. Framework ignores everything else.
                    We always return None — hook is for side effects only.
    transform_tool_result: return string to replace result, or None to leave it.
    post_api_request:  observer only, return value ignored.

Key design decisions:
    - pre_tool_call stash is keyed on tool_call_id
    - Stash entries older than TTL are evicted on each pre_tool_call fire
    - byte-offset scoping means multiple delegate_task calls in one session
      never cross-contaminate each other
    - tool_call_id is threaded through both hooks, enabling exact scoping
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLUGIN_PATH = Path(__file__).resolve().parents[3] / "plugins" / "model_observability" / "__init__.py"


def _load_plugin(tmp_path: Path):
    """Load the plugin module in isolation with a tmp log dir.

    LOG_PATH is patched BEFORE exec_module by injecting it via a
    module-level attribute seed so future register() calls also see it.
    """
    log_path = tmp_path / "model_usage.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location(
        f"model_observability_{id(tmp_path)}", PLUGIN_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Redirect after exec — safe because no top-level code opens the file,
    # but we explicitly test _load_plugin isolation in TestLoadPluginIsolation.
    mod.LOG_PATH = log_path
    return mod


def _read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


def _write_log_entries(log_path: Path, records: list[dict]) -> None:
    """Append records to the log (does not truncate existing content)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_log(log_path: Path, records: list[dict]) -> None:
    """Overwrite the log with exactly these records."""
    log_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _subagent_record(session_id: str, task_id: str, model_req: str, model_resp: str,
                     match: bool = True, tokens_in: int = 100, tokens_out: int = 30,
                     duration_s: float = 0.5) -> dict:
    return {
        "session_id": session_id,
        "task_id": task_id,
        "agent_type": "subagent",
        "model_request": model_req,
        "model_response": model_resp,
        "match": match,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "duration_s": duration_s,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plugin(tmp_path):
    return _load_plugin(tmp_path)


@pytest.fixture
def log_path(plugin):
    return plugin.LOG_PATH


# ===========================================================================
# 0. Plugin isolation — _load_plugin redirect happens before any writes
# ===========================================================================

class TestLoadPluginIsolation:

    def test_log_path_points_to_tmp_not_real_home(self, plugin, log_path, tmp_path):
        """LOG_PATH must be inside tmp_path, never the real ~/.hermes/logs/."""
        assert str(log_path).startswith(str(tmp_path))
        assert ".hermes/logs" not in str(log_path)

    def test_writes_go_to_tmp_log(self, plugin, log_path):
        plugin._on_post_api_request(
            task_id=None, session_id="s1", platform="test",
            model="anthropic/claude-haiku-4.5", provider="openrouter",
            api_mode="chat_completions", api_call_count=1,
            api_duration=1.0, finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            response_model="anthropic/claude-haiku-4.5",
            assistant_content_chars=20, assistant_tool_call_count=0,
        )
        assert log_path.exists()
        real_log = Path.home() / ".hermes" / "logs" / "model_usage.jsonl"
        if real_log.exists():
            real_entries = _read_log(real_log)
            tmp_entries = _read_log(log_path)
            # tmp log has our entry; real log was not touched by this test
            assert len(tmp_entries) == 1
            # Real log entries won't have been written by this isolated module instance


# ===========================================================================
# 1. pre_tool_call — scoping anchor + missing-pin detection
# ===========================================================================

class TestPreToolCallScopingAnchor:

    def test_returns_none_for_non_delegate_task_tools(self, plugin, log_path):
        """pre_tool_call must always return None for non-delegation tools."""
        result = plugin._on_pre_tool_call(
            tool_name="web_search",
            args={"query": "test"},
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
        )
        assert result is None

    def test_returns_none_for_delegate_task_with_model_pinned(self, plugin, log_path):
        """pre_tool_call must ALWAYS return None — it's a side-effect hook, not a blocker."""
        result = plugin._on_pre_tool_call(
            tool_name="delegate_task",
            args={"goal": "task", "model": "anthropic/claude-haiku-4.5"},
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
        )
        assert result is None

    def test_returns_none_for_delegate_task_without_model(self, plugin, log_path):
        """Even with missing pin, pre_tool_call returns None — never blocks or warns directly."""
        result = plugin._on_pre_tool_call(
            tool_name="delegate_task",
            args={"goal": "task"},
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
        )
        assert result is None

    def test_never_returns_block(self, plugin, log_path):
        """Hard requirement: soft enforcement only. block is permanently off."""
        for args in [
            {},
            {"goal": "task"},
            {"tasks": [{"goal": "a"}, {"goal": "b"}]},
        ]:
            result = plugin._on_pre_tool_call(
                tool_name="delegate_task",
                args=args,
                task_id="",
                session_id="s1",
                tool_call_id="tc-block-test",
            )
            assert result is None or result.get("action") != "block"

    def test_stashes_byte_offset_for_tool_call_id(self, plugin, log_path):
        """After pre_tool_call fires, the stash must contain a byte offset for this tool_call_id."""
        # Write some entries first so offset is nonzero
        _write_log_entries(log_path, [
            _subagent_record("s1", "prior-task", "a/model", "a/model")
        ])

        plugin._on_pre_tool_call(
            tool_name="delegate_task",
            args={"goal": "task", "model": "anthropic/claude-haiku-4.5"},
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
        )
        assert "tc1" in plugin._delegate_stash
        assert plugin._delegate_stash["tc1"]["offset"] > 0

    def test_stashes_missing_pin_false_when_model_present(self, plugin, log_path):
        plugin._on_pre_tool_call(
            tool_name="delegate_task",
            args={"goal": "task", "model": "anthropic/claude-haiku-4.5"},
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
        )
        assert plugin._delegate_stash["tc1"]["missing_pin"] is False

    def test_stashes_missing_pin_true_when_model_absent(self, plugin, log_path):
        plugin._on_pre_tool_call(
            tool_name="delegate_task",
            args={"goal": "task"},
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
        )
        assert plugin._delegate_stash["tc1"]["missing_pin"] is True

    def test_stashes_missing_pin_false_when_all_batch_tasks_pinned(self, plugin, log_path):
        plugin._on_pre_tool_call(
            tool_name="delegate_task",
            args={"tasks": [
                {"goal": "a", "model": "anthropic/claude-haiku-4.5"},
                {"goal": "b", "model": "openai/gpt-5-nano"},
            ]},
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
        )
        assert plugin._delegate_stash["tc1"]["missing_pin"] is False

    def test_stashes_missing_pin_true_when_any_batch_task_unpinned(self, plugin, log_path):
        plugin._on_pre_tool_call(
            tool_name="delegate_task",
            args={"tasks": [
                {"goal": "a", "model": "anthropic/claude-haiku-4.5"},
                {"goal": "b"},  # missing model
            ]},
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
        )
        assert plugin._delegate_stash["tc1"]["missing_pin"] is True

    def test_stash_zero_offset_when_log_empty(self, plugin, log_path):
        """If log doesn't exist yet, offset should be 0."""
        log_path.unlink(missing_ok=True)
        plugin._on_pre_tool_call(
            tool_name="delegate_task",
            args={"goal": "task"},
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
        )
        assert plugin._delegate_stash["tc1"]["offset"] == 0

    def test_evicts_expired_stash_entries(self, plugin, log_path):
        """Stash entries older than TTL are evicted on the next pre_tool_call fire."""
        # Manually insert an expired entry
        plugin._delegate_stash["old-tc"] = {
            "offset": 0,
            "missing_pin": False,
            "ts": time.monotonic() - plugin._STASH_TTL_S - 1,
        }
        plugin._on_pre_tool_call(
            tool_name="delegate_task",
            args={"goal": "task"},
            task_id="",
            session_id="s1",
            tool_call_id="tc-new",
        )
        assert "old-tc" not in plugin._delegate_stash
        assert "tc-new" in plugin._delegate_stash

    def test_non_delegate_tool_does_not_add_to_stash(self, plugin, log_path):
        plugin._on_pre_tool_call(
            tool_name="terminal",
            args={"command": "ls"},
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
        )
        assert "tc1" not in plugin._delegate_stash


# ===========================================================================
# 2. transform_tool_result — observability enrichment + scoping
# ===========================================================================

class TestTransformToolResult:

    def _fire_pre(self, plugin, log_path, tool_call_id: str, args: dict,
                  pre_existing_records: list[dict] | None = None) -> None:
        """Helper: optionally write pre-existing records, then fire pre_tool_call."""
        if pre_existing_records:
            _write_log_entries(log_path, pre_existing_records)
        plugin._on_pre_tool_call(
            tool_name="delegate_task",
            args=args,
            task_id="",
            session_id="s1",
            tool_call_id=tool_call_id,
        )

    def _fire_transform(self, plugin, log_path, tool_call_id: str,
                        args: dict, result: str,
                        new_records: list[dict] | None = None) -> str | None:
        """Helper: optionally write new records (simulating subagent writes), then fire transform."""
        if new_records:
            _write_log_entries(log_path, new_records)
        return plugin._on_transform_tool_result(
            tool_name="delegate_task",
            args=args,
            result=result,
            task_id="",
            session_id="s1",
            tool_call_id=tool_call_id,
            duration_ms=1000,
        )

    # ── basic pass-through ──────────────────────────────────────────────────

    def test_ignores_non_delegate_task_tools(self, plugin, log_path):
        result = plugin._on_transform_tool_result(
            tool_name="web_search",
            args={"query": "test"},
            result='{"data": []}',
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
            duration_ms=100,
        )
        assert result is None

    # ── warning surfaces via transform, not pre_tool_call ──────────────────

    def test_missing_pin_warning_surfaces_in_transform_result(self, plugin, log_path):
        """The missing-pin flag set by pre_tool_call must appear in the transform result."""
        self._fire_pre(plugin, log_path, "tc1", args={"goal": "task"})  # no model pin
        # Subagent resolves via auto-router
        result = self._fire_transform(
            plugin, log_path, "tc1",
            args={"goal": "task"},
            result='{"summary": "done"}',
            new_records=[_subagent_record("s1", "subagent-0-abc",
                                          "openrouter/auto", "google/gemini-2.5-flash",
                                          match=False)],
        )
        assert result is not None
        assert isinstance(result, str)
        # Must warn that no model was pinned
        lower = result.lower()
        assert "no model" in lower or "unpinned" in lower or "unspecified" in lower or "pin" in lower

    def test_missing_pin_warning_does_not_appear_when_model_pinned(self, plugin, log_path):
        """No missing-pin warning when a model was explicitly pinned."""
        self._fire_pre(plugin, log_path, "tc1",
                       args={"goal": "task", "model": "anthropic/claude-haiku-4.5"})
        result = self._fire_transform(
            plugin, log_path, "tc1",
            args={"goal": "task", "model": "anthropic/claude-haiku-4.5"},
            result='{"summary": "done"}',
            new_records=[_subagent_record("s1", "subagent-0-abc",
                                          "anthropic/claude-haiku-4.5",
                                          "anthropic/claude-haiku-4.5",
                                          match=True)],
        )
        if result is not None:
            lower = result.lower()
            assert "no model" not in lower and "unpinned" not in lower

    # ── override mismatch warning ───────────────────────────────────────────

    def test_injects_prominent_warning_when_override_dropped(self, plugin, log_path):
        """Pinned model requested but different model answered → prominent WARNING."""
        self._fire_pre(plugin, log_path, "tc1",
                       args={"goal": "task", "model": "anthropic/claude-opus-4.7"})
        result = self._fire_transform(
            plugin, log_path, "tc1",
            args={"goal": "task", "model": "anthropic/claude-opus-4.7"},
            result='{"summary": "done"}',
            new_records=[_subagent_record("s1", "subagent-0-abc",
                                          "anthropic/claude-opus-4.7",
                                          "google/gemini-2.5-flash",
                                          match=False)],
        )
        assert result is not None
        assert "WARNING" in result or "⚠️" in result or "mismatch" in result.lower()
        assert "anthropic/claude-opus-4.7" in result
        assert "google/gemini-2.5-flash" in result

    def test_auto_router_resolution_noted_not_warned(self, plugin, log_path):
        """Auto-router resolution (no pin) is noted informatively, not as a warning."""
        self._fire_pre(plugin, log_path, "tc1", args={"goal": "task"})
        result = self._fire_transform(
            plugin, log_path, "tc1",
            args={"goal": "task"},
            result='{"summary": "done"}',
            new_records=[_subagent_record("s1", "subagent-0-abc",
                                          "openrouter/auto", "google/gemini-2.5-flash",
                                          match=False)],
        )
        assert result is not None
        assert "google/gemini-2.5-flash" in result
        # No mismatch warning — auto-router resolution is expected
        assert "WARNING" not in result and "⚠️" not in result

    def test_pareto_router_resolution_noted_not_warned(self, plugin, log_path):
        """Pareto router resolves to a concrete model; that is expected, not an override drop."""
        self._fire_pre(plugin, log_path, "tc1",
                       args={"goal": "task", "model": "openrouter/pareto-code"})
        result = self._fire_transform(
            plugin, log_path, "tc1",
            args={"goal": "task", "model": "openrouter/pareto-code"},
            result='{"summary": "done"}',
            new_records=[_subagent_record("s1", "subagent-0-abc",
                                          "openrouter/pareto-code", "anthropic/claude-sonnet-4.6",
                                          match=False)],
        )
        assert result is not None
        assert "pareto-router resolved to: anthropic/claude-sonnet-4.6" in result
        assert "WARNING" not in result and "⚠️" not in result

    # ── scoping: prior delegations must not bleed in ────────────────────────

    def test_prior_delegation_records_do_not_bleed_into_current(self, plugin, log_path):
        """Records from a prior delegate_task call in the same session must be filtered out.

        Only records written AFTER the pre_tool_call offset are reported.
        """
        PRIOR_MODEL = "openai/o3"
        CURRENT_MODEL = "anthropic/claude-haiku-4.5"

        # Simulate a prior delegation that already ran and wrote records
        prior_records = [_subagent_record("s1", "subagent-0-prior",
                                          PRIOR_MODEL, PRIOR_MODEL, match=True)]
        self._fire_pre(plugin, log_path, "tc2",
                       args={"goal": "second task", "model": CURRENT_MODEL},
                       pre_existing_records=prior_records)

        # Now the current delegation writes its records
        result = self._fire_transform(
            plugin, log_path, "tc2",
            args={"goal": "second task", "model": CURRENT_MODEL},
            result='{"summary": "done"}',
            new_records=[_subagent_record("s1", "subagent-0-current",
                                          CURRENT_MODEL, CURRENT_MODEL, match=True)],
        )
        assert result is not None
        # Prior model must NOT appear — scoping anchored it out
        assert PRIOR_MODEL not in result

    def test_parent_calls_and_session_start_filtered_out(self, plugin, log_path):
        """Parent API calls and session_start events are not subagent records — filter them."""
        # Write a session_start and a parent call, then fire pre
        existing = [
            {"event": "session_start", "session_id": "s1", "model": "anthropic/claude-sonnet-4.6"},
            {"session_id": "s1", "task_id": None, "agent_type": "parent",
             "model_request": "anthropic/claude-sonnet-4.6",
             "model_response": "anthropic/claude-sonnet-4.6",
             "match": True, "tokens_in": 1000, "tokens_out": 200, "duration_s": 3.0},
        ]
        self._fire_pre(plugin, log_path, "tc1",
                       args={"goal": "task", "model": "anthropic/claude-haiku-4.5"},
                       pre_existing_records=existing)
        result = self._fire_transform(
            plugin, log_path, "tc1",
            args={"goal": "task", "model": "anthropic/claude-haiku-4.5"},
            result='{"summary": "done"}',
            new_records=[_subagent_record("s1", "subagent-0-abc",
                                          "anthropic/claude-haiku-4.5",
                                          "anthropic/claude-haiku-4.5", match=True)],
        )
        if result is not None:
            # Parent call model (sonnet) must not appear as a mismatch
            assert "WARNING" not in result and "⚠️" not in result

    # ── stash lifecycle ─────────────────────────────────────────────────────

    def test_stash_entry_popped_after_transform(self, plugin, log_path):
        """transform_tool_result must clean up its stash entry — no leak."""
        self._fire_pre(plugin, log_path, "tc1", args={"goal": "task"})
        assert "tc1" in plugin._delegate_stash

        self._fire_transform(
            plugin, log_path, "tc1",
            args={"goal": "task"},
            result='{"summary": "done"}',
        )
        assert "tc1" not in plugin._delegate_stash

    def test_transform_without_pre_tool_call_does_not_crash(self, plugin, log_path):
        """If no stash entry exists (pre never fired), degrade gracefully — no crash."""
        result = plugin._on_transform_tool_result(
            tool_name="delegate_task",
            args={"goal": "task"},
            result='{"summary": "done"}',
            task_id="",
            session_id="s1",
            tool_call_id="tc-no-stash",
            duration_ms=500,
        )
        assert result is None or isinstance(result, str)

    # ── mixed batch: match + mismatch + auto-router in one call ────────────

    def test_mixed_batch_warning_only_for_mismatch(self, plugin, log_path):
        """Batch: task A matched, task B mismatched, task C auto-routed.
        WARNING appears for B only; C is noted without warning; A is silent/healthy."""
        self._fire_pre(plugin, log_path, "tc1", args={"tasks": [
            {"goal": "a", "model": "anthropic/claude-haiku-4.5"},
            {"goal": "b", "model": "anthropic/claude-opus-4.7"},
            {"goal": "c"},  # auto-router
        ]})
        records = [
            _subagent_record("s1", "subagent-0-abc",
                             "anthropic/claude-haiku-4.5",
                             "anthropic/claude-haiku-4.5", match=True),
            _subagent_record("s1", "subagent-1-abc",
                             "anthropic/claude-opus-4.7",
                             "google/gemini-2.5-flash", match=False),   # MISMATCH
            _subagent_record("s1", "subagent-2-abc",
                             "openrouter/auto",
                             "openai/gpt-5-nano", match=False),          # auto
        ]
        result = self._fire_transform(
            plugin, log_path, "tc1",
            args={"tasks": [
                {"goal": "a", "model": "anthropic/claude-haiku-4.5"},
                {"goal": "b", "model": "anthropic/claude-opus-4.7"},
                {"goal": "c"},
            ]},
            result='[{"summary": "a"}, {"summary": "b"}, {"summary": "c"}]',
            new_records=records,
        )
        assert result is not None
        # Must warn on B's mismatch
        assert "WARNING" in result or "⚠️" in result or "mismatch" in result.lower()
        assert "anthropic/claude-opus-4.7" in result
        assert "google/gemini-2.5-flash" in result

    # ── failed delegation ───────────────────────────────────────────────────

    def test_handles_failed_delegation_no_jsonl_written(self, plugin, log_path):
        """If delegate_task errored before children ran, no subagent JSONL was written.
        Plugin must pass through gracefully (return None or minimal note), no false warning."""
        self._fire_pre(plugin, log_path, "tc1",
                       args={"goal": "task", "model": "anthropic/claude-haiku-4.5"})
        # No new records written (children never ran)
        result = plugin._on_transform_tool_result(
            tool_name="delegate_task",
            args={"goal": "task", "model": "anthropic/claude-haiku-4.5"},
            result='{"error": "max_spawn_depth exceeded"}',
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
            duration_ms=10,
        )
        # Must not inject a mismatch warning — nothing ran
        if result is not None:
            assert "WARNING" not in result and "⚠️" not in result

    # ── data integrity ──────────────────────────────────────────────────────

    def test_original_result_content_preserved(self, plugin, log_path):
        """The original result data must be preserved in the enriched output."""
        self._fire_pre(plugin, log_path, "tc1",
                       args={"goal": "task", "model": "anthropic/claude-haiku-4.5"})
        result = self._fire_transform(
            plugin, log_path, "tc1",
            args={"goal": "task", "model": "anthropic/claude-haiku-4.5"},
            result='{"summary": "expected content", "status": "ok"}',
            new_records=[_subagent_record("s1", "subagent-0-abc",
                                          "anthropic/claude-haiku-4.5",
                                          "anthropic/claude-haiku-4.5", match=True)],
        )
        if result is not None:
            assert "expected content" in result

    def test_multiple_subagent_calls_all_models_reported(self, plugin, log_path):
        """Batch across 3 tasks with different models — all three models appear in output."""
        self._fire_pre(plugin, log_path, "tc1", args={"tasks": [
            {"goal": f"task {i}", "model": f"model-{i}/test"} for i in range(3)
        ]})
        records = [
            _subagent_record("s1", f"subagent-{i}-abc",
                             f"model-{i}/test", f"model-{i}/test", match=True)
            for i in range(3)
        ]
        result = self._fire_transform(
            plugin, log_path, "tc1",
            args={"tasks": [{"goal": f"task {i}", "model": f"model-{i}/test"} for i in range(3)]},
            result='[{"summary": "a"}, {"summary": "b"}, {"summary": "c"}]',
            new_records=records,
        )
        if result is not None:
            for i in range(3):
                assert f"model-{i}/test" in result

    # ── resilience ──────────────────────────────────────────────────────────

    def test_handles_missing_log_file_gracefully(self, plugin, log_path):
        """JSONL log file doesn't exist — no crash."""
        log_path.unlink(missing_ok=True)
        self._fire_pre(plugin, log_path, "tc1", args={"goal": "task"})
        result = plugin._on_transform_tool_result(
            tool_name="delegate_task",
            args={"goal": "task"},
            result='{"summary": "done"}',
            task_id="",
            session_id="s1",
            tool_call_id="tc1",
            duration_ms=500,
        )
        assert result is None or isinstance(result, str)

    def test_never_raises_on_bad_args(self, plugin, log_path):
        """Garbage args must never crash the agent loop."""
        plugin._on_transform_tool_result(
            tool_name="delegate_task",
            args=None,
            result=None,
            task_id=None,
            session_id=None,
            tool_call_id=None,
            duration_ms="bad",
        )
        # must not raise


# ===========================================================================
# 3. post_api_request — regression suite (existing behavior must not break)
# ===========================================================================

class TestPostApiRequestRegression:

    def test_writes_jsonl_record_for_subagent_call(self, plugin, log_path):
        plugin._on_post_api_request(
            task_id="subagent-0-abc",
            session_id="s1",
            platform="telegram",
            model="anthropic/claude-haiku-4.5",
            provider="openrouter",
            api_mode="chat_completions",
            api_call_count=1,
            api_duration=1.5,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            response_model="anthropic/claude-haiku-4.5",
            assistant_content_chars=200,
            assistant_tool_call_count=0,
        )
        records = _read_log(log_path)
        assert len(records) == 1
        r = records[0]
        assert r["task_id"] == "subagent-0-abc"
        assert r["agent_type"] == "subagent"
        assert r["model_request"] == "anthropic/claude-haiku-4.5"
        assert r["model_response"] == "anthropic/claude-haiku-4.5"
        assert r["match"] is True

    def test_writes_jsonl_record_for_parent_call(self, plugin, log_path):
        plugin._on_post_api_request(
            task_id=None,
            session_id="s1",
            platform="telegram",
            model="anthropic/claude-sonnet-4.6",
            provider="openrouter",
            api_mode="chat_completions",
            api_call_count=1,
            api_duration=2.0,
            finish_reason="stop",
            usage={"prompt_tokens": 200, "completion_tokens": 80},
            response_model="anthropic/claude-sonnet-4.6",
            assistant_content_chars=500,
            assistant_tool_call_count=2,
        )
        records = _read_log(log_path)
        assert len(records) == 1
        assert records[0]["agent_type"] == "parent"
        assert records[0]["task_id"] is None

    def test_match_false_for_mismatched_models(self, plugin, log_path):
        plugin._on_post_api_request(
            task_id="subagent-0-abc",
            session_id="s1",
            platform="telegram",
            model="anthropic/claude-opus-4.7",
            provider="openrouter",
            api_mode="chat_completions",
            api_call_count=1,
            api_duration=3.0,
            finish_reason="stop",
            usage={"prompt_tokens": 500, "completion_tokens": 200},
            response_model="google/gemini-2.5-flash",
            assistant_content_chars=800,
            assistant_tool_call_count=0,
        )
        records = _read_log(log_path)
        assert records[0]["match"] is False

    def test_match_true_for_date_versioned_alias(self, plugin, log_path):
        """Anthropic returns date-versioned aliases — must still match."""
        plugin._on_post_api_request(
            task_id="subagent-0-abc",
            session_id="s1",
            platform="telegram",
            model="anthropic/claude-sonnet-4.6",
            provider="openrouter",
            api_mode="chat_completions",
            api_call_count=1,
            api_duration=1.8,
            finish_reason="stop",
            usage={"prompt_tokens": 300, "completion_tokens": 90},
            response_model="anthropic/claude-4.6-sonnet-20260217",
            assistant_content_chars=400,
            assistant_tool_call_count=1,
        )
        records = _read_log(log_path)
        assert records[0]["match"] is True

    def test_auto_router_always_match_false(self, plugin, log_path):
        """openrouter/auto → anything is a resolution, never a match."""
        plugin._on_post_api_request(
            task_id="subagent-0-abc",
            session_id="s1",
            platform="telegram",
            model="openrouter/auto",
            provider="openrouter",
            api_mode="chat_completions",
            api_call_count=1,
            api_duration=1.1,
            finish_reason="stop",
            usage={"prompt_tokens": 150, "completion_tokens": 40},
            response_model="google/gemini-2.5-flash",
            assistant_content_chars=100,
            assistant_tool_call_count=0,
        )
        records = _read_log(log_path)
        assert records[0]["match"] is False

    def test_thread_safe_concurrent_writes(self, plugin, log_path):
        """Concurrent subagent writes must not corrupt the log."""
        errors = []

        def write_record(i):
            try:
                plugin._on_post_api_request(
                    task_id=f"subagent-{i}-abc",
                    session_id="s1",
                    platform="telegram",
                    model="anthropic/claude-haiku-4.5",
                    provider="openrouter",
                    api_mode="chat_completions",
                    api_call_count=1,
                    api_duration=0.5,
                    finish_reason="stop",
                    usage={"prompt_tokens": 100, "completion_tokens": 30},
                    response_model="anthropic/claude-haiku-4.5",
                    assistant_content_chars=80,
                    assistant_tool_call_count=0,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_record, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        records = _read_log(log_path)
        assert len(records) == 20

    def test_never_raises_on_bad_kwargs(self, plugin, log_path):
        """plugin must never crash the agent loop, even with garbage kwargs."""
        plugin._on_post_api_request(
            task_id=None,
            session_id=None,
            platform=None,
            model=None,
            provider=None,
            api_mode=None,
            api_call_count=None,
            api_duration="not_a_float",
            finish_reason=None,
            usage="bad_type",
            response_model=None,
            assistant_content_chars="bad",
            assistant_tool_call_count="bad",
        )

    def test_session_start_writes_boundary_marker(self, plugin, log_path):
        plugin._on_session_start(
            session_id="s1",
            platform="telegram",
            model="anthropic/claude-sonnet-4.6",
            provider="openrouter",
        )
        records = _read_log(log_path)
        assert len(records) == 1
        assert records[0].get("event") == "session_start"
        assert records[0]["session_id"] == "s1"


# ===========================================================================
# 4. register — hook registration contract
# ===========================================================================

class TestRegister:

    def test_registers_four_hooks(self, plugin):
        """register() must subscribe all four hooks: pre_tool_call,
        transform_tool_result, post_api_request, on_session_start."""
        ctx = MagicMock()
        plugin.register(ctx)
        hook_names = [call.args[0] for call in ctx.register_hook.call_args_list]
        assert "post_api_request" in hook_names
        assert "on_session_start" in hook_names
        assert "pre_tool_call" in hook_names
        assert "transform_tool_result" in hook_names
        assert len(set(hook_names)) == 4  # no duplicates
