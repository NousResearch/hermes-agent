"""Unit tests for agent.task_review — structured task-review engine."""

import pytest

from agent.task_review import (
    MemoryWriteCandidate,
    TaskReviewResult,
    _content_key,
    apply_memory_writeback,
    extract_memory_candidates,
    review_completed_task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload(
    *,
    trigger_reasons=None,
    tools_used=None,
    completed=True,
    interrupted=False,
    final_response="Done",
    original_user_message="do something",
):
    """Build a minimal task-completion payload dict for testing."""
    return {
        "session_id": "test-session",
        "platform": "",
        "model": "test/model",
        "completed": completed,
        "interrupted": interrupted,
        "original_user_message": original_user_message,
        "final_response": final_response,
        "tool_call_count": len(tools_used or []),
        "tools_used": tools_used or [],
        "trigger_reasons": trigger_reasons or [],
    }


# ---------------------------------------------------------------------------
# TaskReviewResult dataclass
# ---------------------------------------------------------------------------

class TestTaskReviewResult:
    def test_is_frozen(self):
        r = TaskReviewResult(
            should_review_memory=False,
            should_review_skills=False,
            review_reasons=[],
            payload={},
        )
        with pytest.raises(AttributeError):
            r.should_review_memory = True  # type: ignore[misc]

    def test_fields_accessible(self):
        p = {"key": "val"}
        r = TaskReviewResult(
            should_review_memory=True,
            should_review_skills=False,
            review_reasons=["reason"],
            payload=p,
        )
        assert r.should_review_memory is True
        assert r.should_review_skills is False
        assert r.review_reasons == ["reason"]
        assert r.payload is p


# ---------------------------------------------------------------------------
# review_completed_task — input validation
# ---------------------------------------------------------------------------

class TestReviewCompletedTaskValidation:
    def test_none_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            review_completed_task(None)  # type: ignore[arg-type]

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            review_completed_task({})


# ---------------------------------------------------------------------------
# review_completed_task — decision logic
# ---------------------------------------------------------------------------

class TestReviewCompletedTaskDecisions:
    def test_no_triggers_returns_no_review(self):
        result = review_completed_task(_payload(trigger_reasons=[]))
        assert result.should_review_memory is False
        assert result.should_review_skills is False
        assert result.review_reasons == []

    def test_tool_used_triggers_skill_review(self):
        result = review_completed_task(
            _payload(trigger_reasons=["tool_used"], tools_used=["web_search"])
        )
        assert result.should_review_skills is True
        assert result.should_review_memory is False
        assert any("tools were used" in r for r in result.review_reasons)

    def test_explicit_memory_request_triggers_memory_review(self):
        result = review_completed_task(
            _payload(
                trigger_reasons=["explicit_memory_request"],
                original_user_message="remember this preference",
            )
        )
        assert result.should_review_memory is True
        assert result.should_review_skills is False
        assert any("explicitly asked" in r for r in result.review_reasons)

    def test_memory_tool_used_triggers_both(self):
        result = review_completed_task(
            _payload(
                trigger_reasons=["tool_used"],
                tools_used=["web_search", "memory"],
            )
        )
        assert result.should_review_memory is True
        assert result.should_review_skills is True
        assert len(result.review_reasons) == 2

    def test_user_profile_tool_triggers_memory(self):
        result = review_completed_task(
            _payload(
                trigger_reasons=["tool_used"],
                tools_used=["user_profile"],
            )
        )
        assert result.should_review_memory is True
        assert result.should_review_skills is True

    def test_combined_triggers(self):
        result = review_completed_task(
            _payload(
                trigger_reasons=["tool_used", "explicit_memory_request"],
                tools_used=["code_editor"],
            )
        )
        assert result.should_review_memory is True
        assert result.should_review_skills is True
        assert len(result.review_reasons) == 2

    def test_payload_passthrough(self):
        p = _payload(trigger_reasons=["tool_used"], tools_used=["web_search"])
        result = review_completed_task(p)
        assert result.payload is p

    def test_unknown_trigger_reason_ignored(self):
        result = review_completed_task(
            _payload(trigger_reasons=["some_future_reason"])
        )
        assert result.should_review_memory is False
        assert result.should_review_skills is False
        assert result.review_reasons == []


# ---------------------------------------------------------------------------
# Helpers — fake memory store
# ---------------------------------------------------------------------------

class _FakeMemoryStore:
    """Minimal stand-in for MemoryStore with .add(target, content)."""

    def __init__(self, *, fail=False):
        self.writes: list = []
        self._fail = fail

    def add(self, target: str, content: str) -> dict:
        if self._fail:
            raise RuntimeError("store error")
        self.writes.append((target, content))
        return {"success": True, "message": "Entry added"}


# ---------------------------------------------------------------------------
# extract_memory_candidates
# ---------------------------------------------------------------------------

class TestExtractMemoryCandidates:
    def test_explicit_request_produces_user_facts(self):
        candidates = extract_memory_candidates(
            _payload(
                trigger_reasons=["explicit_memory_request"],
                original_user_message="remember this: I prefer dark mode",
            )
        )
        assert len(candidates) == 1
        assert candidates[0].category == "user_facts"
        assert candidates[0].content == "I prefer dark mode"
        assert candidates[0].source == "explicit_memory_request"

    def test_no_explicit_request_returns_empty(self):
        candidates = extract_memory_candidates(
            _payload(trigger_reasons=["tool_used"], tools_used=["web_search"])
        )
        assert candidates == []

    def test_empty_payload_returns_empty(self):
        assert extract_memory_candidates({}) == []

    def test_none_payload_returns_empty(self):
        assert extract_memory_candidates(None) == []


# ---------------------------------------------------------------------------
# _content_key
# ---------------------------------------------------------------------------

class TestContentKey:
    def test_normalizes_case_and_whitespace(self):
        assert _content_key("user_facts", "  Hello World  ") == "user_facts:hello world"

    def test_different_categories_produce_different_keys(self):
        assert _content_key("user_facts", "x") != _content_key("workflow_facts", "x")

    def test_identical_inputs_produce_same_key(self):
        assert _content_key("user_facts", "foo") == _content_key("user_facts", "foo")


# ---------------------------------------------------------------------------
# apply_memory_writeback — write routing
# ---------------------------------------------------------------------------

class TestApplyMemoryWritebackRouting:
    def test_user_facts_route_to_user_target(self):
        store = _FakeMemoryStore()
        candidates = [MemoryWriteCandidate("user_facts", "likes dark mode")]
        written = apply_memory_writeback(candidates, store)
        assert len(written) == 1
        assert store.writes == [("user", "likes dark mode")]

    def test_environment_facts_route_to_memory_target(self):
        store = _FakeMemoryStore()
        candidates = [MemoryWriteCandidate("environment_facts", "uses WSL2")]
        written = apply_memory_writeback(candidates, store)
        assert len(written) == 1
        assert store.writes == [("memory", "uses WSL2")]

    def test_workflow_facts_route_to_memory_target(self):
        store = _FakeMemoryStore()
        candidates = [MemoryWriteCandidate("workflow_facts", "prefers TDD")]
        written = apply_memory_writeback(candidates, store)
        assert len(written) == 1
        assert store.writes == [("memory", "prefers TDD")]

    def test_ignore_category_produces_no_write(self):
        store = _FakeMemoryStore()
        candidates = [MemoryWriteCandidate("ignore", "transient info")]
        written = apply_memory_writeback(candidates, store)
        assert written == []
        assert store.writes == []

    def test_empty_candidates_returns_empty(self):
        store = _FakeMemoryStore()
        assert apply_memory_writeback([], store) == []

    def test_none_store_returns_empty(self):
        candidates = [MemoryWriteCandidate("user_facts", "x")]
        assert apply_memory_writeback(candidates, None) == []

    def test_store_exception_skips_candidate(self):
        store = _FakeMemoryStore(fail=True)
        candidates = [MemoryWriteCandidate("user_facts", "x")]
        written = apply_memory_writeback(candidates, store)
        assert written == []

    def test_store_failure_response_skips_candidate(self):
        """Store returns {success: False} — candidate is not counted as written."""
        store = _FakeMemoryStore()
        store.add = lambda t, c: {"success": False, "error": "limit reached"}
        candidates = [MemoryWriteCandidate("user_facts", "x")]
        written = apply_memory_writeback(candidates, store)
        assert written == []


# ---------------------------------------------------------------------------
# apply_memory_writeback — duplicate suppression
# ---------------------------------------------------------------------------

class TestApplyMemoryWritebackDedup:
    def test_exact_duplicate_suppressed(self):
        store = _FakeMemoryStore()
        candidates = [
            MemoryWriteCandidate("user_facts", "likes dark mode"),
            MemoryWriteCandidate("user_facts", "likes dark mode"),
        ]
        written = apply_memory_writeback(candidates, store)
        assert len(written) == 1
        assert len(store.writes) == 1

    def test_case_insensitive_duplicate_suppressed(self):
        store = _FakeMemoryStore()
        candidates = [
            MemoryWriteCandidate("user_facts", "Likes Dark Mode"),
            MemoryWriteCandidate("user_facts", "likes dark mode"),
        ]
        written = apply_memory_writeback(candidates, store)
        assert len(written) == 1

    def test_whitespace_normalized_duplicate_suppressed(self):
        store = _FakeMemoryStore()
        candidates = [
            MemoryWriteCandidate("user_facts", "  likes dark mode  "),
            MemoryWriteCandidate("user_facts", "likes dark mode"),
        ]
        written = apply_memory_writeback(candidates, store)
        assert len(written) == 1

    def test_different_categories_not_treated_as_duplicates(self):
        store = _FakeMemoryStore()
        candidates = [
            MemoryWriteCandidate("user_facts", "foo"),
            MemoryWriteCandidate("workflow_facts", "foo"),
        ]
        written = apply_memory_writeback(candidates, store)
        assert len(written) == 2
        assert len(store.writes) == 2

    def test_written_keys_accumulate_across_calls(self):
        store = _FakeMemoryStore()
        shared_keys: set = set()

        # First call writes one entry.
        written1 = apply_memory_writeback(
            [MemoryWriteCandidate("user_facts", "x")],
            store,
            written_keys=shared_keys,
        )
        assert len(written1) == 1

        # Second call with the same content is suppressed.
        written2 = apply_memory_writeback(
            [MemoryWriteCandidate("user_facts", "x")],
            store,
            written_keys=shared_keys,
        )
        assert len(written2) == 0
        # Store received only one write total.
        assert len(store.writes) == 1

    def test_pre_populated_keys_suppress_matching_candidate(self):
        store = _FakeMemoryStore()
        pre_keys = {_content_key("user_facts", "existing")}
        candidates = [MemoryWriteCandidate("user_facts", "existing")]
        written = apply_memory_writeback(candidates, store, written_keys=pre_keys)
        assert written == []
        assert store.writes == []
