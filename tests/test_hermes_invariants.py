"""Tests for hermes_invariants.InvariantChecker."""

import os
import sys
import types
from unittest import mock

# Set silent mode so the checker doesn't log warnings during tests
os.environ["HERMES_INVARIANT_MODE"] = "silent"

import pytest

from hermes_invariants import InvariantChecker


# =========================================================================
# Message Integrity
# =========================================================================


class TestMessageIntegrity:
    def test_message_integrity_valid(self):
        """A well-formed message list should produce no violations."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "calc", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "4"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]
        violations = InvariantChecker.check_message_integrity(messages)
        assert violations == []

    def test_message_integrity_orphan_tool(self):
        """A tool result after an assistant with no tool_calls should be caught."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        violations = InvariantChecker.check_message_integrity(messages)
        assert len(violations) >= 1
        assert any("ORPHAN_TOOL_RESULT" in v for v in violations)

    def test_message_integrity_empty(self):
        """An empty message list should be caught."""
        violations = InvariantChecker.check_message_integrity([])
        assert len(violations) >= 1
        assert any("EMPTY_MESSAGES" in v for v in violations)


# =========================================================================
# Cron Job Invariants
# =========================================================================


class TestCronInvariants:
    """Tests for cron invariant checking.

    We patch cron.jobs.JOB_STATES to avoid pulling in the full cron
    import chain (scheduler, gateway, etc.) which has heavy dependencies
    that aren't relevant to invariant checking.
    """

    @pytest.fixture(autouse=True)
    def _patch_cron_jobs(self):
        """Ensure cron.jobs is importable with JOB_STATES available."""
        # If cron.jobs is already imported, we're fine.
        # Otherwise, ensure it's importable by pre-seeding the module
        # with at least JOB_STATES so the invariant checker can find it.
        _already = "cron.jobs" in sys.modules
        if not _already:
            # Create a minimal stub so `from cron.jobs import JOB_STATES` works
            stub = types.ModuleType("cron.jobs")
            stub.JOB_STATES = frozenset(
                {"scheduled", "running", "paused", "completed", "failed"}
            )
            # Also ensure the parent package exists
            if "cron" not in sys.modules:
                cron_stub = types.ModuleType("cron")
                cron_stub.__path__ = []  # make it a package
                sys.modules["cron"] = cron_stub
            sys.modules["cron.jobs"] = stub

        yield

        # Don't remove if it was already there before the test
        if not _already and "cron.jobs" in sys.modules:
            del sys.modules["cron.jobs"]

    def test_cron_invariants_valid(self):
        """Valid jobs with proper states and unique IDs should pass."""
        jobs = [
            {"id": "job_1", "state": "scheduled", "next_run_at": 1700000000},
            {"id": "job_2", "state": "paused"},
            {"id": "job_3", "state": "completed"},
        ]
        violations = InvariantChecker.check_cron_invariants(jobs)
        assert violations == []

    def test_cron_invariants_duplicate_id(self):
        """Duplicate job IDs should be caught."""
        jobs = [
            {"id": "job_1", "state": "scheduled"},
            {"id": "job_1", "state": "paused"},
        ]
        violations = InvariantChecker.check_cron_invariants(jobs)
        assert len(violations) >= 1
        assert any("DUPLICATE_JOB_ID" in v for v in violations)

    def test_cron_invariants_invalid_state(self):
        """An invalid job state should be caught."""
        jobs = [
            {"id": "job_1", "state": "banana"},
        ]
        violations = InvariantChecker.check_cron_invariants(jobs)
        assert len(violations) >= 1
        assert any("INVALID_STATE" in v for v in violations)


# =========================================================================
# Process Registry Invariants
# =========================================================================


class _FakeSession:
    """Minimal stand-in for ProcessSession in invariant tests."""

    def __init__(self, sid, exited=False):
        self.id = sid
        self.exited = exited


class TestProcessRegistry:
    def test_process_registry_overlap(self):
        """A session in both running and finished should be caught."""
        session = _FakeSession("proc_abc", exited=True)
        running = {"proc_abc": session}
        finished = {"proc_abc": session}
        violations = InvariantChecker.check_process_registry(running, finished)
        assert len(violations) >= 1
        assert any("DUAL_REGISTRY" in v for v in violations)

    def test_process_registry_clean(self):
        """Disjoint running/finished sets should produce no violations."""
        s1 = _FakeSession("proc_aaa", exited=False)
        s2 = _FakeSession("proc_bbb", exited=True)
        running = {"proc_aaa": s1}
        finished = {"proc_bbb": s2}
        violations = InvariantChecker.check_process_registry(running, finished)
        assert violations == []
