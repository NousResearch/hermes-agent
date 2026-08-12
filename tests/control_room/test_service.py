"""Phase 1 tests: snapshot service + /control command backbone (CR-106).

Covers: service composition with injected providers, typed unavailable when
providers raise, deterministic attention ordering in snapshots, bounded cache
behaviour, section normalization, plain-text rendering, and no snapshot
exceptions when optional plugins/services are absent.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from control_room import (
    AttentionItem,
    AttentionKind,
    AttentionSeverity,
    ControlRoomSnapshot,
    MessageRow,
    SystemSummary,
    TaskRow,
    rank_attention,
)
from control_room.contract import SourceMeta
from control_room.service import ControlRoomService
from control_room.text import (
    attention_status_line,
    normalize_section,
    render_home,
    render_section,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ok_providers() -> Dict[str, Any]:
    return {
        "agents": lambda ctx: [
            {"kind": "agent", "id": "a1", "name": "main", "status": "running", "elapsed_seconds": 12}
        ],
        "processes": lambda ctx: [],
        "delegations": lambda ctx: [],
        "kanban": lambda ctx: [
            {"id": "t1", "title": "Fix gateway", "state": "blocked", "owner": "octacon"},
            {"id": "t2", "title": "Write docs", "state": "ready", "owner": "light"},
        ],
        "peer": lambda ctx: [
            {"kind": "peer_message", "id": "m1", "title": "incoming request", "state": "held", "sender": "remii"}
        ],
        "system": lambda ctx: [
            SystemSummary(
                state="healthy",
                severity=AttentionSeverity.info,
                detail="ok",
                source=SourceMeta(provider="system", state="ok"),
            )
        ],
    }


def _raising_providers() -> Dict[str, Any]:
    def boom(ctx):
        raise RuntimeError("provider down")

    return {
        "agents": boom,
        "processes": boom,
        "delegations": boom,
        "kanban": boom,
        "peer": boom,
        "system": boom,
    }


class TestServiceComposition:
    def test_build_snapshot_with_ok_providers(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot(profile="kensei")
        assert isinstance(snap, ControlRoomSnapshot)
        assert snap.profile == "kensei"
        assert snap.version == 1
        assert snap.counts.agents_active == 1
        assert snap.counts.tasks_running == 1  # t2 ready
        assert snap.counts.messages_unread == 1
        assert snap.counts.needs_you == 1  # held peer message is critical
        assert snap.capabilities.peer_messages is True

    def test_raising_providers_yield_typed_unavailable_not_crash(self):
        service = ControlRoomService(providers=_raising_providers())
        snap = service.build_snapshot(profile="kensei")
        # No exception: each provider degrades to a typed unavailable source.
        assert snap.capabilities.peer_messages is False
        assert snap.capabilities.kanban_actions is False
        assert snap.system.state == "unknown"
        assert snap.counts.agents_active == 0
        assert snap.counts.needs_you == 0

    def test_missing_provider_is_unavailable(self):
        service = ControlRoomService(providers={"system": lambda ctx: []})
        snap = service.build_snapshot()
        assert snap.system.state == "unknown"
        assert snap.capabilities.peer_messages is False

    def test_attention_ordering_contract_in_snapshot(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot()
        # held_message (critical) must rank above blocked task (warning).
        if snap.attention:
            assert snap.attention[0].severity == AttentionSeverity.critical
        kinds = [a.kind for a in snap.attention]
        assert AttentionKind.held_message in kinds
        assert AttentionKind.blocked_task in kinds
        assert kinds.index(AttentionKind.held_message) < kinds.index(AttentionKind.blocked_task)

    def test_capabilities_false_when_approvals_not_wired(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot()
        assert snap.capabilities.approvals is False


class TestCache:
    def test_cache_returns_same_object_within_ttl(self):
        calls = {"n": 0}

        def counting_provider(name):
            def provider(ctx):
                if name == "agents":
                    calls["n"] += 1
                    return [{"kind": "agent", "id": "a", "name": "x", "status": "running"}]
                if name == "system":
                    return [
                        SystemSummary(state="healthy", severity=AttentionSeverity.info, source=SourceMeta(provider="system"))
                    ]
                return []

            return provider

        providers = {n: counting_provider(n) for n in ("agents", "processes", "delegations", "kanban", "peer", "system")}
        service = ControlRoomService(providers=providers, cache_max_age_seconds=60)
        s1 = service.build_snapshot()
        s2 = service.build_snapshot()
        assert s1 is s2

    def test_refresh_bypasses_cache(self):
        calls = {"n": 0}

        def provider(ctx):
            calls["n"] += 1
            return []

        providers = {n: provider for n in ("agents", "processes", "delegations", "kanban", "peer")}
        providers["system"] = lambda ctx: [SystemSummary(state="healthy", severity=AttentionSeverity.info)]
        service = ControlRoomService(providers=providers, cache_max_age_seconds=60)
        service.build_snapshot()
        service.build_snapshot(refresh=True)
        assert calls["n"] >= 2

    def test_invalidate_clears_cache(self):
        calls = {"n": 0}

        def provider(ctx):
            calls["n"] += 1
            return []

        providers = {n: provider for n in ("agents", "processes", "delegations", "kanban", "peer")}
        providers["system"] = lambda ctx: [SystemSummary(state="healthy", severity=AttentionSeverity.info)]
        service = ControlRoomService(providers=providers, cache_max_age_seconds=60)
        service.build_snapshot()
        service.invalidate()
        service.build_snapshot()
        assert calls["n"] >= 2


class TestCommandResolution:
    def test_section_normalization(self):
        assert normalize_section("home") == "home"
        assert normalize_section("needs-you") == "needs-you"
        assert normalize_section("needs_you") == "needs-you"
        assert normalize_section("agents") == "agents"
        assert normalize_section("messages") == "messages"
        assert normalize_section("system") == "system"
        assert normalize_section("bogus") == "home"
        assert normalize_section("") == "home"

    def test_unknown_section_falls_back_home(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot()
        out = render_section("bogus", snap)
        assert "Control Room" in out

    def test_control_command_registered_in_central_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY, _build_command_lookup

        lookup = _build_command_lookup()
        assert "control" in lookup
        cmd = lookup["control"]
        assert cmd.name == "control"
        assert cmd.args_hint.startswith("[home")
        assert "needs-you" in cmd.subcommands
        assert cmd.busy_policy == "dispatch"

    def test_control_command_survives_resolve(self):
        from hermes_cli.commands import resolve_command

        resolved = resolve_command("/control")
        assert resolved is not None
        assert resolved.name == "control"


class TestTextRendering:
    def test_home_contains_contract_rows(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot()
        out = render_home(snap)
        assert "Needs You" in out
        assert "Agents" in out
        assert "Tasks" in out
        assert "Messages" in out
        assert "System" in out
        assert "+ New Task" in out
        assert "Ctrl+P" in out

    def test_needs_you_section_lists_urgent_items(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot()
        out = render_section("needs-you", snap)
        assert "Peer message" in out
        # Blocked task is warning severity — correctly NOT in needs-you
        # (needs-you = critical/error only).
        assert "Blocked" not in out

    def test_agents_section(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot()
        out = render_section("agents", snap)
        assert "main" in out

    def test_tasks_section(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot()
        out = render_section("tasks", snap)
        assert "Fix gateway" in out
        assert "Write docs" in out

    def test_messages_section(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot()
        out = render_section("messages", snap)
        assert "incoming request" in out

    def test_system_section(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot()
        out = render_section("system", snap)
        assert "healthy" in out

    def test_status_line_compact(self):
        service = ControlRoomService(providers=_ok_providers())
        snap = service.build_snapshot()
        out = attention_status_line(snap)
        assert "need you" in out
        assert "Ctrl+P Control Room" in out

    def test_render_snapshot_with_no_rows_does_not_crash(self):
        service = ControlRoomService(providers=_raising_providers())
        snap = service.build_snapshot()
        for section in ("home", "needs-you", "agents", "tasks", "messages", "system"):
            render_section(section, snap)  # must not raise
        attention_status_line(snap)


class TestNoOptionalDependencyCrashes:
    def test_service_imports_without_peer_plugin(self):
        # The service itself must import cleanly; the peer provider isolates
        # the optional hermes_peer import behind try/except.
        import control_room.service  # noqa: F401

    def test_service_without_kanban_db_degrades(self):
        # kanban provider isolates its import; absence -> typed unavailable.
        service = ControlRoomService(
            providers={
                "kanban": lambda ctx: (_ for _ in ()).throw(ImportError("no kanban_db")),
                "system": lambda ctx: [SystemSummary(state="healthy", severity=AttentionSeverity.info)],
            }
        )
        snap = service.build_snapshot()
        assert snap.capabilities.kanban_actions is False


class TestRankingStability:
    def test_rank_attention_stable_with_mixed_inputs(self):
        items = [
            AttentionItem(
                kind=AttentionKind.info, id="i", severity=AttentionSeverity.info,
                title="t", source=SourceMeta(provider="x"),
            ),
            AttentionItem(
                kind=AttentionKind.running, id="r", severity=AttentionSeverity.info,
                title="t", source=SourceMeta(provider="x"),
            ),
        ]
        assert rank_attention(items) == rank_attention(list(reversed(items)))
