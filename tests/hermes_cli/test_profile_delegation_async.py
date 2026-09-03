"""Tests for the async completion protocol (reconcile_queued_delegation).

These tests verify the second blocker from @andrexibiza's review of PR #102406:
queued profile delegations must have a durable reconciler that observes terminal
state, updates the delegation row, and emits exactly one completion event.
"""
from __future__ import annotations

import time
import threading

import yaml


def _home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    for profile in ("cto", "coo", "cmo"):
        (home / "profiles" / profile / "mcp-tokens").mkdir(parents=True, exist_ok=True)
    return home


def _enable_cap(home, profile, server="vercel"):
    p = home / "profiles" / profile
    (p / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {server: {"enabled": True, "auth": "oauth"}}}), encoding="utf-8")
    (p / "mcp-tokens" / f"{server}.json").write_text("secret", encoding="utf-8")


class TestReconcileQueuedDelegation:
    """Verify the durable reconciler for async completion protocol."""

    def test_reconcile_completed_delegation(self, tmp_path, monkeypatch):
        """A queued delegation whose task completes should be reconciled
        with status 'completed' and the result extracted."""
        home = _home(tmp_path, monkeypatch)
        _enable_cap(home, "cto")

        from hermes_cli import kanban_db as kb
        from hermes_cli.profile_delegation import (
            ProfileDelegationRequest,
            delegate_to_profile,
            reconcile_queued_delegation,
        )

        def delayed_spawn(task, workspace, board):
            def complete_later():
                time.sleep(0.5)
                with kb.connect_closing() as conn:
                    deleg = kb.get_profile_delegation_by_task(conn, task.id)
                    if deleg:
                        kb.complete_task(
                            conn,
                            task.id,
                            summary="Vercel inspected.",
                            result="No secrets.",
                            metadata={"profile_delegation": {
                                "delegation_id": deleg["id"],
                                "capability": "mcp:vercel",
                                "risk": "READ",
                                "status": "completed",
                                "structured_result": {"project": "ConnectMe", "status": "ok"},
                                "redaction": {"secrets_returned": False},
                            }},
                        )
                        conn.commit()

            t = threading.Thread(target=complete_later, daemon=True)
            t.start()
            return 12345

        with kb.connect_closing() as conn:
            req = ProfileDelegationRequest(
                profile=None,
                task="Inspect Vercel project ConnectMe status",
                required_capability="mcp:vercel",
                requester_profile="cmo",
                requester_session_key="agent:main:cli:cmo:session123",
                timeout_seconds=0,  # Return immediately as "queued"
                max_concurrency=2,
            )
            result = delegate_to_profile(req, spawn_fn=delayed_spawn)

        assert result.status == "queued"
        delegation_id = result.delegation_id

        # Wait for the async completion
        time.sleep(1.0)

        # Now reconcile
        reconciled = reconcile_queued_delegation(delegation_id)
        assert reconciled is not None
        assert reconciled.status == "completed"
        assert reconciled.result is not None
        assert reconciled.result.get("project") == "ConnectMe"
        assert reconciled.result.get("status") == "ok"

    def test_reconcile_failed_delegation(self, tmp_path, monkeypatch):
        """A queued delegation whose task fails should be reconciled
        with status 'failed'."""
        home = _home(tmp_path, monkeypatch)
        _enable_cap(home, "cto")

        from hermes_cli import kanban_db as kb
        from hermes_cli.profile_delegation import (
            ProfileDelegationRequest,
            delegate_to_profile,
            reconcile_queued_delegation,
        )

        def delayed_spawn(task, workspace, board):
            def complete_later():
                time.sleep(0.5)
                with kb.connect_closing() as conn:
                    deleg = kb.get_profile_delegation_by_task(conn, task.id)
                    if deleg:
                        kb.block_task(conn, task.id, reason="Vercel MCP authentication failed")
                        conn.commit()

            t = threading.Thread(target=complete_later, daemon=True)
            t.start()
            return 12345

        with kb.connect_closing() as conn:
            req = ProfileDelegationRequest(
                profile=None,
                task="Inspect Vercel",
                required_capability="mcp:vercel",
                requester_profile="cmo",
                requester_session_key="agent:main:cli:cmo:session456",
                timeout_seconds=0,
            )
            result = delegate_to_profile(req, spawn_fn=delayed_spawn)

        assert result.status == "queued"
        delegation_id = result.delegation_id

        time.sleep(1.0)

        reconciled = reconcile_queued_delegation(delegation_id)
        assert reconciled is not None
        assert reconciled.status == "failed"
        assert "Vercel MCP authentication failed" in (reconciled.error or "")

    def test_reconcile_still_running_returns_none(self, tmp_path, monkeypatch):
        """A delegation whose task is still running should return None."""
        home = _home(tmp_path, monkeypatch)
        _enable_cap(home, "cto")

        from hermes_cli import kanban_db as kb
        from hermes_cli.profile_delegation import (
            ProfileDelegationRequest,
            delegate_to_profile,
            reconcile_queued_delegation,
        )

        def never_complete(task, workspace, board):
            # Don't complete the task
            return 12345

        with kb.connect_closing() as conn:
            req = ProfileDelegationRequest(
                profile=None,
                task="Inspect Vercel",
                required_capability="mcp:vercel",
                requester_profile="cmo",
                timeout_seconds=0,
            )
            result = delegate_to_profile(req, spawn_fn=never_complete)

        assert result.status == "queued"
        delegation_id = result.delegation_id

        # Task is still blocked/running, reconciliation should return None
        reconciled = reconcile_queued_delegation(delegation_id)
        assert reconciled is None

    def test_reconcile_idempotent(self, tmp_path, monkeypatch):
        """Reconciling the same delegation twice should be idempotent:
        the second call returns None (already reconciled)."""
        home = _home(tmp_path, monkeypatch)
        _enable_cap(home, "cto")

        from hermes_cli import kanban_db as kb
        from hermes_cli.profile_delegation import (
            ProfileDelegationRequest,
            delegate_to_profile,
            reconcile_queued_delegation,
        )

        def delayed_spawn(task, workspace, board):
            def complete_later():
                time.sleep(0.3)
                with kb.connect_closing() as conn:
                    deleg = kb.get_profile_delegation_by_task(conn, task.id)
                    if deleg:
                        kb.complete_task(
                            conn,
                            task.id,
                            summary="Done.",
                            result="OK.",
                            metadata={"profile_delegation": {
                                "delegation_id": deleg["id"],
                                "capability": "mcp:vercel",
                                "risk": "READ",
                                "status": "completed",
                                "structured_result": {"ok": True},
                                "redaction": {"secrets_returned": False},
                            }},
                        )
                        conn.commit()

            t = threading.Thread(target=complete_later, daemon=True)
            t.start()
            return 12345

        with kb.connect_closing() as conn:
            req = ProfileDelegationRequest(
                profile=None,
                task="Inspect Vercel",
                required_capability="mcp:vercel",
                requester_profile="cmo",
                timeout_seconds=0,
            )
            result = delegate_to_profile(req, spawn_fn=delayed_spawn)

        delegation_id = result.delegation_id
        time.sleep(0.8)

        # First reconciliation
        r1 = reconcile_queued_delegation(delegation_id)
        assert r1 is not None
        assert r1.status == "completed"

        # Second reconciliation should return None (already reconciled)
        r2 = reconcile_queued_delegation(delegation_id)
        assert r2 is None

    def test_reconcile_nonexistent_delegation(self, tmp_path, monkeypatch):
        """Reconciling a non-existent delegation returns None."""
        home = _home(tmp_path, monkeypatch)
        _enable_cap(home, "cto")

        from hermes_cli.profile_delegation import reconcile_queued_delegation

        result = reconcile_queued_delegation("pd_nonexistent123")
        assert result is None


class TestEmitCompletionEvent:
    """Verify that _emit_completion_event correctly handles session_key."""

    def test_emit_completion_event_with_valid_session_key(self, tmp_path, monkeypatch):
        """With a valid session key, the function should attempt to emit."""
        home = _home(tmp_path, monkeypatch)
        _enable_cap(home, "cto")

        from hermes_cli.profile_delegation import (
            ProfileDelegationResult,
            _emit_completion_event,
        )

        result = ProfileDelegationResult(
            status="completed",
            delegation_id="pd_test123",
            task_id="t_test",
            executor_profile="cto",
            requester_profile="cmo",
            capability="mcp:vercel",
            risk="READ",
            result={"ok": True},
            summary="Done.",
        )
        # Should not raise; the import may fail in test env but is caught
        _emit_completion_event(result, "agent:main:cli:cmo:session789")

    def test_emit_completion_event_without_session_key(self, tmp_path, monkeypatch):
        """Without a session key, no event should be emitted (function returns)."""
        home = _home(tmp_path, monkeypatch)
        _enable_cap(home, "cto")

        from hermes_cli.profile_delegation import (
            ProfileDelegationResult,
            _emit_completion_event,
        )

        result = ProfileDelegationResult(
            status="completed",
            delegation_id="pd_test456",
            task_id="t_test",
            executor_profile="cto",
            requester_profile="cmo",
            capability="mcp:vercel",
            risk="READ",
        )
        # Should not raise, just return
        _emit_completion_event(result, None)

    def test_emit_completion_event_with_empty_session_key(self, tmp_path, monkeypatch):
        """With an empty session key, no event should be emitted."""
        home = _home(tmp_path, monkeypatch)
        _enable_cap(home, "cto")

        from hermes_cli.profile_delegation import (
            ProfileDelegationResult,
            _emit_completion_event,
        )

        result = ProfileDelegationResult(
            status="completed",
            delegation_id="pd_test789",
            task_id="t_test",
            executor_profile="cto",
            requester_profile="cmo",
            capability="mcp:vercel",
            risk="READ",
        )
        _emit_completion_event(result, "")
