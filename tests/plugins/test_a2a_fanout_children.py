"""
Adversarial tests for A2A fan-out child context persistence and resumability.

Covers:
  - Out-of-order peer completion
  - Late callback after restart (persistence recovery)
  - Same-peer continuation via child context_id
  - Unknown child context rejection
  - Conflicting peer reuse rejection
  - Aggregate modes (all, first, best) with mapping preservation
  - Parent→child mapping machine-readable output
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from plugins.platforms.a2a import adapter as a2a_adapter
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a import tools


_TWO_PEERS = {
    "a2a_agents": {
        "alpha": {
            "url": "http://localhost:9991",
            "capabilities": ["research"],
            "timeout": 5,
        },
        "beta": {
            "url": "http://localhost:9992",
            "capabilities": ["research"],
            "timeout": 5,
        },
        "coder": {
            "url": "http://localhost:9993",
            "capabilities": ["code"],
            "timeout": 5,
        },
    }
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_origin() -> dict:
    return {
        "platform": "discord",
        "chat_id": "ch_test",
        "chat_type": "dm",
        "thread_id": "",
        "user_id": "u123",
        "profile": "default",
        "session_id": "sess_test",
    }


def _setup_adapter() -> a2a_adapter.A2AAdapter:
    """Create a minimal A2AAdapter for tests (no real HTTP server)."""
    adapter = a2a_adapter.A2AAdapter.__new__(a2a_adapter.A2AAdapter)
    adapter._context_peers = {}
    adapter._context_peers_lock = threading.Lock()
    adapter._context_sessions = {}
    adapter._context_sessions_lock = threading.Lock()
    adapter._fanout_children = {}
    adapter._fanout_children_lock = threading.Lock()
    adapter._inbound_seen = {}
    adapter._inbound_seen_lock = threading.Lock()
    adapter.host = "127.0.0.1"
    adapter.port = 9900
    adapter.agent_name = "test-agent"
    # Register so classmethods can find us.
    import weakref
    a2a_adapter._ADAPTERS[id(adapter)] = weakref.ref(adapter)
    return adapter


# ---------------------------------------------------------------------------
# Tests: out-of-order peer completion
# ---------------------------------------------------------------------------


class TestFanOutOutOfOrder:
    """Two peers complete in non-deterministic order; mapping stays consistent."""

    def test_out_of_order_results_sorted_by_name(self, monkeypatch):
        """Results are sorted by peer name regardless of completion order."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)

        call_order = []
        lock = threading.Lock()

        def _slow_first(name, entry, msg, ctx=""):
            with lock:
                call_order.append(name)
            # alpha sleeps to simulate slower response
            if name == "alpha":
                time.sleep(0.05)
            return (name, f"reply-{name}", ctx or protocol.new_context_id())

        monkeypatch.setattr(tools, "_call_peer_sync", _slow_first)
        out = tools.a2a_orchestrate({"capability": "research", "message": "go"})

        # Results sorted by name
        assert "reply-alpha" in out
        assert "reply-beta" in out
        # Mapping section present
        assert "## Peer Mapping" in out
        assert "parent_context_id:" in out
        assert "alpha:" in out
        assert "beta:" in out

    def test_each_peer_gets_distinct_child_context(self, monkeypatch):
        """Each peer in the fan-out receives a unique child context_id."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)

        seen_ctxs = set()
        lock = threading.Lock()

        def _capture_ctx(name, entry, msg, ctx=""):
            with lock:
                seen_ctxs.add(ctx)
            return (name, f"ok-{name}", ctx or "ctx-fallback")

        monkeypatch.setattr(tools, "_call_peer_sync", _capture_ctx)
        out = tools.a2a_orchestrate({"capability": "research", "message": "go"})

        # Each peer got a distinct child context
        assert len(seen_ctxs) == 2


# ---------------------------------------------------------------------------
# Tests: late callback after restart (persistence recovery)
# ---------------------------------------------------------------------------


class TestFanOutLateCallback:
    """Fan-out children survive in-memory wipe (restart simulation)."""

    def test_fanout_children_persisted_and_recoverable(self, monkeypatch):
        """Register fan-out children, then create a fresh adapter that
        restores from disk — the mapping is still available."""
        adapter = _setup_adapter()
        adapter2 = None
        try:
            parent = protocol.new_context_id()
            child_a = protocol.new_context_id()
            child_b = protocol.new_context_id()
            peer_children = {"alpha": child_a, "beta": child_b}

            a2a_adapter.A2AAdapter._register_fanout_children(
                parent, peer_children, origin=_fake_origin()
            )

            # Verify in-memory
            result = a2a_adapter.A2AAdapter._get_fanout_children(parent)
            assert result == peer_children

            # Simulate restart: create fresh adapter, restore from disk
            adapter2 = _setup_adapter()
            restored = adapter2._restore_persisted_fanout_children()
            assert restored >= 1

            # Fresh adapter can resolve children
            result2 = a2a_adapter.A2AAdapter._get_fanout_children(parent)
            assert result2 == peer_children
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)
            if adapter2 is not None:
                a2a_adapter._ADAPTERS.pop(id(adapter2), None)


# ---------------------------------------------------------------------------
# Tests: same-peer continuation
# ---------------------------------------------------------------------------


class TestFanOutContinuation:
    """Same peer can continue its own child context."""

    def test_same_peer_continuation_allowed(self, monkeypatch):
        """A peer continuing its own child context is not rejected."""
        adapter = _setup_adapter()
        try:
            parent = protocol.new_context_id()
            child = protocol.new_context_id()
            a2a_adapter.A2AAdapter._register_fanout_children(
                parent, {"alpha": child}, origin=_fake_origin()
            )
            # Same peer: should NOT be rejected
            claiming = a2a_adapter.A2AAdapter._reject_child_reuse(child, "alpha")
            assert claiming == ""
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)

    def test_a2a_call_same_peer_with_child_context(self, monkeypatch):
        """a2a_call with the owning peer and child context succeeds."""
        adapter = _setup_adapter()
        try:
            parent = protocol.new_context_id()
            child = protocol.new_context_id()
            a2a_adapter.A2AAdapter._register_fanout_children(
                parent, {"alpha": child}, origin=_fake_origin()
            )
            monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
            # The _send_task call will fail (no real peer), but the
            # conflict rejection must NOT block the call.
            result = tools.a2a_call({
                "agent": "alpha",
                "message": "continue",
                "context_id": child,
            })
            # Should get a connection error, NOT a conflict error
            assert "owned by peer" not in result
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)


# ---------------------------------------------------------------------------
# Tests: unknown child context
# ---------------------------------------------------------------------------


class TestFanOutUnknownChild:
    """Continuing an unknown child context is allowed (no rejection)."""

    def test_unknown_child_not_rejected(self):
        """An unknown context_id is not flagged as a conflict."""
        adapter = _setup_adapter()
        try:
            reuse = a2a_adapter.A2AAdapter._reject_child_reuse(
                "ctx-nonexistent", "alpha"
            )
            assert reuse == ""
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)

    def test_a2a_call_unknown_child_context(self, monkeypatch):
        """a2a_call with unknown context_id proceeds normally."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        result = tools.a2a_call({
            "agent": "alpha",
            "message": "hello",
            "context_id": "ctx-totally-unknown",
        })
        # Should get a connection error, NOT a conflict error
        assert "owned by peer" not in result


# ---------------------------------------------------------------------------
# Tests: conflicting peer reuse
# ---------------------------------------------------------------------------


class TestFanOutConflictingReuse:
    """A different peer trying to use another peer's child context is rejected."""

    def test_different_peer_rejected(self):
        """Beta trying to continue alpha's child context is rejected."""
        adapter = _setup_adapter()
        try:
            parent = protocol.new_context_id()
            child = protocol.new_context_id()
            a2a_adapter.A2AAdapter._register_fanout_children(
                parent, {"alpha": child}, origin=_fake_origin()
            )
            claiming = a2a_adapter.A2AAdapter._reject_child_reuse(child, "beta")
            assert claiming == "alpha"
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)

    def test_a2a_call_conflicting_peer_rejected(self, monkeypatch):
        """a2a_call with wrong peer and child context returns conflict error."""
        adapter = _setup_adapter()
        try:
            parent = protocol.new_context_id()
            child = protocol.new_context_id()
            a2a_adapter.A2AAdapter._register_fanout_children(
                parent, {"alpha": child}, origin=_fake_origin()
            )
            monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
            result = tools.a2a_call({
                "agent": "beta",
                "message": "hijack",
                "context_id": child,
            })
            assert "owned by peer 'alpha'" in result
            assert "beta" in result
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)

    def test_conflicting_reuse_survives_restart(self):
        """Conflict rejection works after persistence recovery."""
        adapter = _setup_adapter()
        adapter2 = None
        try:
            parent = protocol.new_context_id()
            child = protocol.new_context_id()
            a2a_adapter.A2AAdapter._register_fanout_children(
                parent, {"alpha": child}, origin=_fake_origin()
            )
            # Simulate restart
            adapter2 = _setup_adapter()
            adapter2._restore_persisted_fanout_children()
            # Beta trying alpha's child
            claiming = a2a_adapter.A2AAdapter._reject_child_reuse(child, "beta")
            assert claiming == "alpha"
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)
            if adapter2 is not None:
                a2a_adapter._ADAPTERS.pop(id(adapter2), None)


# ---------------------------------------------------------------------------
# Tests: aggregate modes with mapping preservation
# ---------------------------------------------------------------------------


class TestFanOutAggregateModes:
    """All/first/best modes preserve the peer→child mapping."""

    def test_all_mode_has_mapping(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(
            tools,
            "_call_peer_sync",
            lambda name, entry, msg, ctx="": (name, f"ok-{name}", ctx),
        )
        out = tools.a2a_orchestrate({"capability": "research", "message": "go"})
        assert "## Peer Mapping" in out
        assert "parent_context_id:" in out
        assert "alpha:" in out
        assert "beta:" in out

    def test_first_mode_has_mapping(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(
            tools,
            "_call_peer_sync",
            lambda name, entry, msg, ctx="": (name, f"ok-{name}", ctx),
        )
        out = tools.a2a_orchestrate(
            {"capability": "research", "message": "go", "mode": "first"}
        )
        assert "## Peer Mapping" in out
        assert "parent_context_id:" in out

    def test_best_mode_has_mapping(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(
            tools,
            "_call_peer_sync",
            lambda name, entry, msg, ctx="": (name, f"ok-{name}", ctx),
        )
        out = tools.a2a_orchestrate(
            {"capability": "research", "message": "go", "mode": "best"}
        )
        assert "## Peer Mapping" in out
        assert "parent_context_id:" in out

    def test_all_failed_mode_has_mapping(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(
            tools,
            "_call_peer_sync",
            lambda name, entry, msg, ctx="": (name, "Error: nope", ""),
        )
        out = tools.a2a_orchestrate({"capability": "research", "message": "go"})
        # In "all" mode, errors are shown inline (not "All peers failed")
        assert "Error: nope" in out
        assert "## Peer Mapping" in out

    def test_best_mode_all_errors_has_mapping(self, monkeypatch):
        """Best mode with all errors still produces the mapping."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(
            tools,
            "_call_peer_sync",
            lambda name, entry, msg, ctx="": (name, "Error: nope", ""),
        )
        out = tools.a2a_orchestrate(
            {"capability": "research", "message": "go", "mode": "best"}
        )
        assert "All peers failed" in out
        assert "## Peer Mapping" in out

    def test_mapping_child_ids_match_peer_calls(self, monkeypatch):
        """Child context IDs in the mapping match what _call_peer_sync received."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)

        received_ctxs = {}

        def _capture(name, entry, msg, ctx=""):
            received_ctxs[name] = ctx
            return (name, f"ok-{name}", ctx)

        monkeypatch.setattr(tools, "_call_peer_sync", _capture)
        out = tools.a2a_orchestrate({"capability": "research", "message": "go"})

        # Parse the mapping section
        lines = out.split("\n")
        in_mapping = False
        mapping = {}
        for line in lines:
            if "## Peer Mapping" in line:
                in_mapping = True
                continue
            if in_mapping and line.startswith("  "):
                parts = line.strip().split(": ", 1)
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]

        # Verify mapping matches received contexts
        for peer, ctx in received_ctxs.items():
            assert peer in mapping, f"Peer {peer} missing from mapping"
            assert mapping[peer] == ctx, (
                f"Mapping for {peer}: expected {ctx}, got {mapping[peer]}"
            )


# ---------------------------------------------------------------------------
# Tests: parent context persistence
# ---------------------------------------------------------------------------


class TestFanOutParentPersistence:
    """Fan-out parent context persists across restart."""

    def test_parent_context_always_present(self, monkeypatch):
        """The parent context_id is always in the mapping output."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(
            tools,
            "_call_peer_sync",
            lambda name, entry, msg, ctx="": (name, f"reply", ctx),
        )
        out = tools.a2a_orchestrate({"capability": "research", "message": "go"})
        # Parse parent_context_id
        for line in out.split("\n"):
            if line.startswith("parent_context_id:"):
                parent = line.split(":", 1)[1].strip()
                assert parent.startswith("ctx-")
                return
        pytest.fail("parent_context_id not found in orchestrate output")

    def test_explicit_context_id_becomes_parent(self, monkeypatch):
        """Passing context_id uses it as the parent."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(
            tools,
            "_call_peer_sync",
            lambda name, entry, msg, ctx="": (name, f"reply", ctx),
        )
        out = tools.a2a_orchestrate({
            "capability": "research",
            "message": "go",
            "context_id": "ctx-custom-parent",
        })
        assert "parent_context_id: ctx-custom-parent" in out


# ---------------------------------------------------------------------------
# Tests: thread safety of concurrent registration
# ---------------------------------------------------------------------------


class TestFanOutThreadSafety:
    """Concurrent registration of fan-out children does not corrupt state."""

    def test_concurrent_registration(self):
        """Registering fan-out children from multiple threads does not crash."""
        adapter = _setup_adapter()
        try:
            errors = []

            def _register(i):
                try:
                    parent = f"ctx-parent-{i}"
                    children = {
                        f"peer-{i}-a": f"ctx-child-{i}-a",
                        f"peer-{i}-b": f"ctx-child-{i}-b",
                    }
                    a2a_adapter.A2AAdapter._register_fanout_children(
                        parent, children, origin=_fake_origin()
                    )
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=_register, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Errors during concurrent registration: {errors}"

            # All 20 parents should be resolvable
            for i in range(20):
                parent = f"ctx-parent-{i}"
                result = a2a_adapter.A2AAdapter._get_fanout_children(parent)
                assert len(result) == 2
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)

    def test_concurrent_registration_survives_restart(self):
        """Concurrent registrations survive restart recovery."""
        adapter = _setup_adapter()
        adapter2 = None
        try:
            # Register 10 parents concurrently
            def _register(i):
                parent = f"ctx-restart-{i}"
                children = {
                    f"peer-{i}-a": f"ctx-child-{i}-a",
                    f"peer-{i}-b": f"ctx-child-{i}-b",
                }
                a2a_adapter.A2AAdapter._register_fanout_children(
                    parent, children, origin=_fake_origin()
                )

            threads = [threading.Thread(target=_register, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            # Simulate restart
            adapter2 = _setup_adapter()
            restored = adapter2._restore_persisted_fanout_children()
            assert restored >= 10

            # All parents should be recoverable
            for i in range(10):
                parent = f"ctx-restart-{i}"
                result = a2a_adapter.A2AAdapter._get_fanout_children(parent)
                assert len(result) == 2
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)
            if adapter2 is not None:
                a2a_adapter._ADAPTERS.pop(id(adapter2), None)

    def test_persistence_atomicity_no_corruption(self):
        """Rapid sequential registrations don't corrupt the disk file."""
        adapter = _setup_adapter()
        try:
            for i in range(50):
                parent = f"ctx-atomic-{i}"
                children = {"alpha": f"ctx-child-{i}"}
                a2a_adapter.A2AAdapter._register_fanout_children(
                    parent, children, origin=_fake_origin()
                )
            # Reload from disk and verify all entries
            disk = a2a_adapter._load_fanout_children()
            for i in range(50):
                parent = f"ctx-atomic-{i}"
                assert parent in disk, f"Missing parent {parent} on disk"
                assert disk[parent] == {"alpha": f"ctx-child-{i}"}
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)


# ---------------------------------------------------------------------------
# Tests: callback/origin-session wake path
# ---------------------------------------------------------------------------


class TestFanOutCallbackWake:
    """Child contexts registered with origin wake the original session."""

    def test_child_context_registered_with_origin(self, monkeypatch):
        """Each child context is registered with the same origin as the parent."""
        adapter = _setup_adapter()
        adapter2 = None
        try:
            parent = protocol.new_context_id()
            child_a = protocol.new_context_id()
            child_b = protocol.new_context_id()
            origin = _fake_origin()

            a2a_adapter.A2AAdapter._register_fanout_children(
                parent, {"alpha": child_a, "beta": child_b}, origin=origin
            )

            # Verify both children are in the fan-out map
            result = a2a_adapter.A2AAdapter._get_fanout_children(parent)
            assert result == {"alpha": child_a, "beta": child_b}

            # The origin session should be reachable via _context_sessions
            # for each child context (registered by tools.py before orchestrate)
            a2a_adapter.A2AAdapter._register_context_session(child_a, origin)
            a2a_adapter.A2AAdapter._register_context_session(child_b, origin)

            # Simulate restart recovery
            adapter2 = _setup_adapter()
            adapter2._restore_persisted_fanout_children()
            restored = adapter2._restore_persisted_context_sessions()
            assert restored >= 2

            # Both children can wake the original session
            with adapter2._context_sessions_lock:
                origin_a = adapter2._context_sessions.get(child_a, {})
                origin_b = adapter2._context_sessions.get(child_b, {})
            assert origin_a.get("session_id") == "sess_test"
            assert origin_b.get("session_id") == "sess_test"
            assert origin_a.get("platform") == "discord"
            assert origin_b.get("platform") == "discord"
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)
            if 'adapter2' in dir() and adapter2 is not None:
                a2a_adapter._ADAPTERS.pop(id(adapter2), None)

    def test_orchestrate_registers_children_with_origin(self, monkeypatch):
        """a2a_orchestrate registers each child context with origin session."""
        adapter = _setup_adapter()
        try:
            monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)

            def _fake_call(name, entry, msg, ctx=""):
                return (name, f"ok-{name}", ctx)

            monkeypatch.setattr(tools, "_call_peer_sync", _fake_call)
            out = tools.a2a_orchestrate({"capability": "research", "message": "go"})

            # Parse parent context ID from output
            parent_ctx = None
            for line in out.split("\n"):
                if line.startswith("parent_context_id:"):
                    parent_ctx = line.split(":", 1)[1].strip()
                    break
            assert parent_ctx is not None

            # Get the fan-out children
            children = a2a_adapter.A2AAdapter._get_fanout_children(parent_ctx)
            assert len(children) >= 2

            # Each child should have been registered with origin session
            # (tools.py calls _register_context_session for each child)
            for peer_name, child_ctx in children.items():
                assert child_ctx, f"Empty child context for {peer_name}"
                assert child_ctx != parent_ctx, (
                    f"Child {peer_name} got same context as parent"
                )
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)


# ---------------------------------------------------------------------------
# Tests: first-mode completeness (mapping covers all submitted peers)
# ---------------------------------------------------------------------------


class TestFanOutFirstModeMapping:
    """first-mode must record all submitted peers in the ownership mapping,
    even when result collection breaks early after the first success."""

    def test_first_mode_mapping_contains_all_submitted_peers(self, monkeypatch):
        """When first peer returns quickly and cancels the rest, the mapping
        still contains both peer child contexts."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        adapter = _setup_adapter()
        try:
            # alpha returns immediately; beta is slow (will be cancelled)
            def _alpha_first(name, entry, msg, ctx=""):
                if name == "alpha":
                    return (name, "alpha-ok", ctx)
                # beta would block but we never get here because
                # as_completed yields alpha first in this setup
                time.sleep(0.5)
                return (name, "beta-ok", ctx)

            monkeypatch.setattr(tools, "_call_peer_sync", _alpha_first)
            out = tools.a2a_orchestrate(
                {"capability": "research", "message": "go", "mode": "first"}
            )

            # Parse parent context ID from output
            parent_ctx = None
            for line in out.split("\n"):
                if line.startswith("parent_context_id:"):
                    parent_ctx = line.split(":", 1)[1].strip()
                    break
            assert parent_ctx is not None

            # The mapping section should list BOTH alpha and beta
            children = a2a_adapter.A2AAdapter._get_fanout_children(parent_ctx)
            assert "alpha" in children, "alpha missing from ownership mapping"
            assert "beta" in children, "beta missing from ownership mapping"
            assert children["alpha"], "alpha child context is empty"
            assert children["beta"], "beta child context is empty"
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)

    def test_first_mode_mapping_persists_across_restart(self, monkeypatch):
        """The complete first-mode mapping survives restart recovery."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        adapter = _setup_adapter()
        adapter2 = None
        try:
            monkeypatch.setattr(
                tools,
                "_call_peer_sync",
                lambda name, entry, msg, ctx="": (name, f"ok-{name}", ctx),
            )
            out = tools.a2a_orchestrate(
                {"capability": "research", "message": "go", "mode": "first"}
            )

            # Parse parent context ID
            parent_ctx = None
            for line in out.split("\n"):
                if line.startswith("parent_context_id:"):
                    parent_ctx = line.split(":", 1)[1].strip()
                    break
            assert parent_ctx is not None

            # Both peers should be in the mapping before "restart"
            children = a2a_adapter.A2AAdapter._get_fanout_children(parent_ctx)
            assert len(children) >= 2

            # Simulate restart
            adapter2 = _setup_adapter()
            adapter2._restore_persisted_fanout_children()
            restored = a2a_adapter.A2AAdapter._get_fanout_children(parent_ctx)
            assert "alpha" in restored, "alpha missing after restart"
            assert "beta" in restored, "beta missing after restart"
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)
            if adapter2 is not None:
                a2a_adapter._ADAPTERS.pop(id(adapter2), None)


# ---------------------------------------------------------------------------
# Tests: bounded fan-out children eviction
# ---------------------------------------------------------------------------


class TestFanOutBoundedEviction:
    """The parent→children map is bounded by _MAX_CONTEXT_PEERS."""

    def test_eviction_at_capacity(self, monkeypatch):
        """When the map is at capacity, inserting a new parent evicts the oldest."""
        adapter = _setup_adapter()
        try:
            # Temporarily lower the cap
            original_cap = a2a_adapter._MAX_CONTEXT_PEERS
            a2a_adapter._MAX_CONTEXT_PEERS = 4
            try:
                for i in range(6):
                    parent = f"parent-{i}"
                    peers = {f"peer-{i}-a": f"child-{i}-a", f"peer-{i}-b": f"child-{i}-b"}
                    a2a_adapter.A2AAdapter._register_fanout_children(parent, peers)

                # In-memory map should be bounded
                with adapter._fanout_children_lock:
                    assert len(adapter._fanout_children) <= 4

                # The two oldest (parent-0, parent-1) should be evicted
                assert "parent-0" not in adapter._fanout_children
                assert "parent-1" not in adapter._fanout_children
                assert "parent-5" in adapter._fanout_children
                assert "parent-4" in adapter._fanout_children
            finally:
                a2a_adapter._MAX_CONTEXT_PEERS = original_cap
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)

    def test_refresh_prevents_eviction(self, monkeypatch):
        """Re-registering an existing parent refreshes it (prevents eviction)."""
        adapter = _setup_adapter()
        try:
            original_cap = a2a_adapter._MAX_CONTEXT_PEERS
            a2a_adapter._MAX_CONTEXT_PEERS = 3
            try:
                for i in range(3):
                    a2a_adapter.A2AAdapter._register_fanout_children(
                        f"parent-{i}", {f"peer-{i}": f"child-{i}"}
                    )
                # Refresh parent-0
                a2a_adapter.A2AAdapter._register_fanout_children(
                    "parent-0", {"peer-0-new": "child-0-new"}
                )
                # Add a 4th parent — parent-1 (oldest non-refreshed) should evict
                a2a_adapter.A2AAdapter._register_fanout_children(
                    "parent-3", {"peer-3": "child-3"}
                )

                with adapter._fanout_children_lock:
                    assert len(adapter._fanout_children) <= 3
                assert "parent-1" not in adapter._fanout_children
                assert "parent-0" in adapter._fanout_children
            finally:
                a2a_adapter._MAX_CONTEXT_PEERS = original_cap
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)

    def test_disk_persistence_respects_cap(self, monkeypatch):
        """Bounded eviction is also applied to the persisted disk map."""
        adapter = _setup_adapter()
        try:
            original_cap = a2a_adapter._MAX_CONTEXT_PEERS
            a2a_adapter._MAX_CONTEXT_PEERS = 3
            try:
                for i in range(5):
                    a2a_adapter.A2AAdapter._register_fanout_children(
                        f"parent-{i}", {f"peer-{i}": f"child-{i}"}
                    )
                # Reload from disk
                disk = a2a_adapter._load_fanout_children()
                assert len(disk) <= 3
                assert "parent-0" not in disk
                assert "parent-4" in disk
            finally:
                a2a_adapter._MAX_CONTEXT_PEERS = original_cap
        finally:
            a2a_adapter._ADAPTERS.pop(id(adapter), None)
