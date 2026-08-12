#!/usr/bin/env python3
"""Acceptance-criteria tests for the Idea Box subsystem.

This file targets the acceptance criteria that the core suite
(tests/ideabox/test_ideabox.py) does not exercise directly:

  * No task is created without explicit approval.
  * Approval creates an idempotent, correctly-scoped Kanban task.
  * Confidence calculation boundary values.
  * The Discord button component handler (approve/reject/amend) and its
    authorisation gate.
  * Store round-trip persistence of approval state and audit events.

These tests use a real (temporary) Idea Box store and a real (temporary)
Kanban database so the full approval -> task-creation path is exercised
without touching the operator's live data.
"""

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Resolve imports from the checkout containing this test.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.platforms.discord.ideabox import handler as ideabox_handler
from plugins.platforms.discord.ideabox.models import (
    ApprovalState, ApprovalStatus, AuditEventType, Classification,
    Provenance, RoutingDecision, Source, SourceSubmission, SourceType,
    TriageSummary,
)
from plugins.platforms.discord.ideabox.store import IdeaBoxStore


def _make_source(**overrides) -> Source:
    base = dict(
        url="https://github.com/owner/repo",
        source_type=SourceType.GITHUB_REPO,
        title="Test Repo",
        author=None,
        published_date=None,
        content_snippet="A test repository for testing purposes",
        content_hash="hash_acceptance",
        url_fingerprint="https://github.com/owner/repo",
        raw_text="https://github.com/owner/repo",
        provenance=Provenance("user1", 1000, "ch1", "msg1", "guild1"),
    )
    base.update(overrides)
    return Source(**base)


def _make_summary(source: Source, triage_id: str = "t_acceptance") -> TriageSummary:
    return TriageSummary(
        triage_id=triage_id,
        source=source,
        classification=Classification(category="tech", tags=["github"]),
        confidence="high",
        risks=[],
        effort="m",
        recommendation="proceed",
        routing=RoutingDecision("octacon-frontend", 0.85, "Tech task"),
        reasoning="Classified as tech | Confidence: high",
        created_at=1000,
    )


class _IsolatedStoreMixin:
    """Reset the module-level store singleton to a fresh temp DB per test."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="ideabox_accept_")
        self._home = Path(self._tmp)
        self._kanban_db = self._home / "kanban.db"
        # Point the store singleton at a fresh temp DB.
        self._store = IdeaBoxStore(self._home / "ideabox" / "ideabox.db")
        ideabox_handler._store = self._store
        # Point the real kanban task-creation path at a temp DB.
        self._old_home = os.environ.get("HERMES_HOME")
        self._old_kanban = os.environ.get("HERMES_KANBAN_DB")
        os.environ["HERMES_HOME"] = str(self._home)
        os.environ["HERMES_KANBAN_DB"] = str(self._kanban_db)
        # Initialise the kanban schema so task counts are queryable even
        # before any task is created (the real path only inits on approve).
        from hermes_cli import kanban_db
        kanban_db.init_db(self._kanban_db)

    def tearDown(self):
        ideabox_handler._store = None
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home
        if self._old_kanban is None:
            os.environ.pop("HERMES_KANBAN_DB", None)
        else:
            os.environ["HERMES_KANBAN_DB"] = self._old_kanban


class TestNoUnapprovedTaskCreation(_IsolatedStoreMixin, unittest.TestCase):
    """Acceptance: no task is created without explicit approval."""

    def _submit(self, raw_text: str) -> dict:
        sub = SourceSubmission(
            raw_text, "user1", "ch1", "msg1", "guild1", "text", int(time.time()),
        )
        return asyncio.run(ideabox_handler.handle_ideabox_submission(sub))

    def _kanban_task_count(self) -> int:
        import sqlite3
        conn = sqlite3.connect(str(self._kanban_db))
        try:
            return conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        finally:
            conn.close()

    def test_submission_alone_creates_no_task(self):
        """A submission that produces a triage embed must NOT create a task."""
        result = self._submit("https://github.com/owner/repo")
        self.assertTrue(result["success"])
        self.assertFalse(result["is_duplicate"])
        # No approval happened, so no Kanban task may exist.
        self.assertEqual(self._kanban_task_count(), 0)

    def test_reject_creates_no_task(self):
        """Rejecting a triage item must not create a task."""
        result = self._submit("https://github.com/owner/repo")
        triage_id = result["triage_summary"].triage_id
        sm = ideabox_handler.ApprovalStateMachine(self._store)
        asyncio.run(sm.reject(triage_id, "user1", "User One", "Not relevant"))
        self.assertEqual(self._kanban_task_count(), 0)
        state = self._store.get_approval(triage_id)
        self.assertEqual(state.status, ApprovalStatus.REJECTED.value)

    def test_amend_creates_no_task(self):
        """Amending a triage item must not create a task."""
        result = self._submit("https://github.com/owner/repo")
        triage_id = result["triage_summary"].triage_id
        sm = ideabox_handler.ApprovalStateMachine(self._store)
        asyncio.run(sm.amend(triage_id, "user1", "User One", "Needs detail"))
        self.assertEqual(self._kanban_task_count(), 0)
        state = self._store.get_approval(triage_id)
        self.assertEqual(state.status, ApprovalStatus.AMENDED.value)


class TestApprovalCreatesScopedTask(_IsolatedStoreMixin, unittest.TestCase):
    """Acceptance: approval creates a correctly-scoped, idempotent task."""

    def _approve(self, triage_id: str):
        sm = ideabox_handler.ApprovalStateMachine(self._store)
        return asyncio.run(sm.approve(triage_id, "user1", "User One"))

    def test_approve_creates_task_with_routing_and_backlog(self):
        result = asyncio.run(ideabox_handler.handle_ideabox_submission(
            SourceSubmission(
                "https://github.com/owner/repo", "user1", "ch1", "msg1",
                "guild1", "text", int(time.time()),
            )
        ))
        triage_id = result["triage_summary"].triage_id
        action = self._approve(triage_id)
        self.assertIsNotNone(action.kanban_task_id)

        import sqlite3
        conn = sqlite3.connect(str(self._kanban_db))
        try:
            row = conn.execute(
                "SELECT id, title, assignee, status, created_by, priority "
                "FROM tasks WHERE id = ?", (action.kanban_task_id,),
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row)
        # Deterministic routing: specialist from the triage summary.
        self.assertEqual(row[2], "octacon-frontend")
        # Approval-created tasks land in backlog (never auto-dispatched).
        self.assertEqual(row[3], "backlog")
        self.assertEqual(row[4], "ideabox")
        # Effort 'm' maps to priority 3.
        self.assertEqual(row[5], 3)

    def test_approve_is_idempotent(self):
        result = asyncio.run(ideabox_handler.handle_ideabox_submission(
            SourceSubmission(
                "https://github.com/owner/repo", "user1", "ch1", "msg1",
                "guild1", "text", int(time.time()),
            )
        ))
        triage_id = result["triage_summary"].triage_id
        action1 = self._approve(triage_id)
        # A second approve on the same (now-approved) item must be rejected.
        with self.assertRaises(ValueError):
            self._approve(triage_id)
        # Exactly one task exists.
        import sqlite3
        conn = sqlite3.connect(str(self._kanban_db))
        try:
            count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 1)
        self.assertIsNotNone(action1.kanban_task_id)

    def test_approve_records_audit_event(self):
        result = asyncio.run(ideabox_handler.handle_ideabox_submission(
            SourceSubmission(
                "https://github.com/owner/repo", "user1", "ch1", "msg1",
                "guild1", "text", int(time.time()),
            )
        ))
        triage_id = result["triage_summary"].triage_id
        self._approve(triage_id)
        events = self._store.get_events(triage_id)
        types = [e.event_type for e in events]
        # The approval transition is recorded in the audit log.
        self.assertIn(AuditEventType.APPROVE.value, types)
        # The created task id is captured on the approval event payload.
        approve_events = [e for e in events if e.event_type == AuditEventType.APPROVE.value]
        self.assertTrue(approve_events)
        self.assertIn("kanban_task_id", approve_events[0].payload)


class TestConfidenceBoundaries(unittest.TestCase):
    """Confidence calculation boundary values (0, 1/3, 2/3, 1)."""

    def _confidence_for(self, *, has_url, has_title, snippet_len):
        source = _make_source(
            url="https://github.com/owner/repo" if has_url else None,
            title="Title" if has_title else None,
            content_snippet="x" * snippet_len,
        )
        summary = ideabox_handler.triage_source(source)
        return summary.confidence

    def test_full_signal_high(self):
        self.assertEqual(self._confidence_for(has_url=True, has_title=True, snippet_len=60), "high")

    def test_two_thirds_medium(self):
        # 2/3 = 0.6667 < 0.67 -> medium (boundary just below high)
        self.assertEqual(self._confidence_for(has_url=True, has_title=True, snippet_len=10), "medium")

    def test_one_third_medium(self):
        # 1/3 = 0.333 >= 0.33 -> medium
        self.assertEqual(self._confidence_for(has_url=True, has_title=False, snippet_len=10), "medium")

    def test_no_signal_low(self):
        self.assertEqual(self._confidence_for(has_url=False, has_title=False, snippet_len=10), "low")


class TestStoreRoundTrip(_IsolatedStoreMixin, unittest.TestCase):
    """Approval state and audit events survive a store round-trip."""

    def test_approval_state_round_trip(self):
        source = _make_source()
        summary = _make_summary(source)
        state = ApprovalState(
            triage_id="t_roundtrip", status=ApprovalStatus.PENDING.value,
            source=source, triage_summary=summary, created_at=1000,
        )
        self._store.save_approval(state)
        loaded = self._store.get_approval("t_roundtrip")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.triage_id, "t_roundtrip")
        self.assertEqual(loaded.status, ApprovalStatus.PENDING.value)
        self.assertEqual(loaded.source.url, source.url)
        self.assertEqual(loaded.source.source_type, SourceType.GITHUB_REPO)
        self.assertEqual(loaded.triage_summary.routing.specialist, "octacon-frontend")
        self.assertEqual(loaded.triage_summary.classification.category, "tech")

    def test_audit_event_round_trip(self):
        from plugins.platforms.discord.ideabox.models import AuditEvent, generate_event_id
        event = AuditEvent(
            event_id=generate_event_id(), event_type=AuditEventType.INTAKE.value,
            triage_id="t_audit", timestamp=1000, actor_id="user1",
            payload={"source_type": "github_repo", "url": "https://github.com/o/r"},
        )
        self._store.log_event(event)
        events = self._store.get_events("t_audit")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, AuditEventType.INTAKE.value)
        self.assertEqual(events[0].payload["url"], "https://github.com/o/r")


class TestComponentAuth(unittest.TestCase):
    """Authorisation gate for Discord approval buttons."""

    def test_allowed_user_id_passes(self):
        interaction = SimpleNamespace(
            user=SimpleNamespace(id="42", roles=()),
        )
        self.assertTrue(
            ideabox_handler._component_check_auth(interaction, {"42"}, None)
        )

    def test_disallowed_user_fails(self):
        interaction = SimpleNamespace(
            user=SimpleNamespace(id="99", roles=()),
        )
        self.assertFalse(
            ideabox_handler._component_check_auth(interaction, {"42"}, None)
        )

    def test_allowed_role_passes(self):
        role = SimpleNamespace(id="role1")
        interaction = SimpleNamespace(
            user=SimpleNamespace(id="99", roles=(role,)),
        )
        self.assertTrue(
            ideabox_handler._component_check_auth(interaction, set(), {"role1"})
        )

    def test_no_allowed_roles_and_not_user_fails(self):
        interaction = SimpleNamespace(
            user=SimpleNamespace(id="99", roles=()),
        )
        self.assertFalse(
            ideabox_handler._component_check_auth(interaction, set(), set())
        )


class TestComponentHandler(_IsolatedStoreMixin, unittest.TestCase):
    """The Discord button component handler (approve/reject/amend)."""

    def _make_interaction(self, user_id="42", display_name="User", embeds=None):
        if embeds is None:
            embeds = [SimpleNamespace(
                title="🔍 Idea Box — Test",
                description="desc",
                color=0x5865F2,
                fields=[],
                footer=SimpleNamespace(text="Idea Box · Triage ID: t_comp"),
                timestamp=None,
            )]
        return SimpleNamespace(
            user=SimpleNamespace(id=user_id, display_name=display_name),
            response=SimpleNamespace(
                send_message=AsyncMock(),
                edit_message=AsyncMock(),
                defer=AsyncMock(),
            ),
            message=SimpleNamespace(embeds=embeds),
            followup=SimpleNamespace(send=AsyncMock()),
        )

    def _seed_pending(self, triage_id="t_comp"):
        source = _make_source()
        summary = _make_summary(source, triage_id=triage_id)
        state = ApprovalState(
            triage_id=triage_id, status=ApprovalStatus.PENDING.value,
            source=source, triage_summary=summary, created_at=1000,
        )
        self._store.save_approval(state)
        return triage_id

    def test_approve_button_creates_task(self):
        triage_id = self._seed_pending()
        interaction = self._make_interaction()
        asyncio.run(ideabox_handler.handle_ideabox_component(
            interaction, f"ideabox_approval:approve:{triage_id}", {"42"}, None,
        ))
        # Followup confirms the created task id.
        self.assertTrue(interaction.followup.send.called)
        msg = interaction.followup.send.call_args.args[0]
        self.assertIn("Approved", msg)
        self.assertIn("t_", msg)
        # State transitioned to approved.
        state = self._store.get_approval(triage_id)
        self.assertEqual(state.status, ApprovalStatus.APPROVED.value)

    def test_reject_button(self):
        triage_id = self._seed_pending()
        interaction = self._make_interaction()
        asyncio.run(ideabox_handler.handle_ideabox_component(
            interaction, f"ideabox_approval:reject:{triage_id}", {"42"}, None,
        ))
        msg = interaction.followup.send.call_args.args[0]
        self.assertIn("Rejected", msg)
        state = self._store.get_approval(triage_id)
        self.assertEqual(state.status, ApprovalStatus.REJECTED.value)

    def test_amend_button(self):
        triage_id = self._seed_pending()
        interaction = self._make_interaction()
        asyncio.run(ideabox_handler.handle_ideabox_component(
            interaction, f"ideabox_approval:amend:{triage_id}", {"42"}, None,
        ))
        msg = interaction.followup.send.call_args.args[0]
        self.assertIn("Amend", msg)
        state = self._store.get_approval(triage_id)
        self.assertEqual(state.status, ApprovalStatus.AMENDED.value)

    def test_unauthorised_user_blocked(self):
        triage_id = self._seed_pending()
        interaction = self._make_interaction(user_id="99")
        asyncio.run(ideabox_handler.handle_ideabox_component(
            interaction, f"ideabox_approval:approve:{triage_id}", {"42"}, None,
        ))
        # Unauthorised -> ephemeral denial, no state change.
        self.assertTrue(interaction.response.send_message.called)
        denial = interaction.response.send_message.call_args.args[0]
        self.assertIn("not authorised", denial.lower())
        state = self._store.get_approval(triage_id)
        self.assertEqual(state.status, ApprovalStatus.PENDING.value)

    def test_malformed_custom_id(self):
        interaction = self._make_interaction()
        asyncio.run(ideabox_handler.handle_ideabox_component(
            interaction, "garbage", {"42"}, None,
        ))
        self.assertTrue(interaction.response.send_message.called)
        msg = interaction.response.send_message.call_args.args[0]
        self.assertIn("Malformed", msg)

    def test_unknown_action(self):
        interaction = self._make_interaction()
        asyncio.run(ideabox_handler.handle_ideabox_component(
            interaction, "ideabox_approval:explode:t_comp", {"42"}, None,
        ))
        self.assertTrue(interaction.response.send_message.called)
        msg = interaction.response.send_message.call_args.args[0]
        self.assertIn("Unknown", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
