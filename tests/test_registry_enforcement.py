"""Tests for ToolRegistry.set_enforcement_fn / enforcement hook."""

import json

import pytest

from tools.registry import EnforcementDenied, ToolRegistry


def _make_registry_with_tool(name: str = "test_tool") -> ToolRegistry:
    r = ToolRegistry()
    r.register(
        name=name,
        toolset="test",
        schema={"name": name, "description": "test"},
        handler=lambda args, **kw: json.dumps({"result": "ok"}),
    )
    return r


class TestEnforcementFn:

    def test_dispatch_succeeds_without_enforcement_fn(self):
        r = _make_registry_with_tool()
        result = json.loads(r.dispatch("test_tool", {}))
        assert result == {"result": "ok"}

    def test_enforcement_denied_blocks_dispatch(self):
        """Raising EnforcementDenied blocks the call with the denial message."""
        r = _make_registry_with_tool()
        r.set_enforcement_fn(
            lambda name, args, **kw: (_ for _ in ()).throw(
                EnforcementDenied(f"{name} denied by policy")
            )
        )
        result = json.loads(r.dispatch("test_tool", {}))
        assert "error" in result
        assert "denied" in result["error"]

    def test_non_denied_exception_fails_closed_with_operator_message(self):
        """A bug in the enforcement fn (non-EnforcementDenied) fails closed.

        The agent sees a generic operator-visible error, not the raw exception
        message — prevents leaking TypeError/AttributeError details as if they
        were authorization policy messages.
        """
        r = _make_registry_with_tool()
        r.set_enforcement_fn(
            lambda name, args, **kw: (_ for _ in ()).throw(
                TypeError("missing 1 required positional argument: 'token'")
            )
        )
        result = json.loads(r.dispatch("test_tool", {}))
        assert "error" in result
        # Must NOT expose the raw TypeError message to the model
        assert "positional argument" not in result["error"]
        # Must surface as an operator-visible internal error including the tool name
        assert "test_tool" in result["error"]
        assert "unavailable" in result["error"].lower() or "authoris" in result["error"].lower()

    def test_enforcement_fn_allow_passes_through(self):
        """An enforcement fn that does NOT raise allows the call."""
        r = _make_registry_with_tool()
        r.set_enforcement_fn(lambda name, args, **kw: None)  # always allow
        result = json.loads(r.dispatch("test_tool", {}))
        assert result == {"result": "ok"}

    def test_enforcement_fn_receives_tool_name_and_args(self):
        received = {}
        def capture(name, args, **kw):
            received["name"] = name
            received["args"] = args

        r = _make_registry_with_tool()
        r.set_enforcement_fn(capture)
        r.dispatch("test_tool", {"key": "value"})
        assert received["name"] == "test_tool"
        assert received["args"] == {"key": "value"}

    def test_enforcement_fn_receives_kwargs(self):
        received = {}
        def capture(name, args, **kw):
            received.update(kw)

        r = _make_registry_with_tool()
        r.set_enforcement_fn(capture)
        r.dispatch("test_tool", {}, task_id="session-abc", user_id="u1")
        assert received.get("task_id") == "session-abc"

    def test_set_enforcement_fn_none_removes_hook(self):
        r = _make_registry_with_tool()
        r.set_enforcement_fn(
            lambda n, a, **kw: (_ for _ in ()).throw(EnforcementDenied("blocked"))
        )
        result = json.loads(r.dispatch("test_tool", {}))
        assert "error" in result
        r.set_enforcement_fn(None)
        result2 = json.loads(r.dispatch("test_tool", {}))
        assert result2 == {"result": "ok"}

    def test_enforcement_denied_message_in_result(self):
        r = _make_registry_with_tool()
        r.set_enforcement_fn(
            lambda n, a, **kw: (_ for _ in ()).throw(
                EnforcementDenied("path constraint violated: /etc not in /data")
            )
        )
        result = json.loads(r.dispatch("test_tool", {"path": "/etc/passwd"}))
        assert "path constraint violated" in result["error"]

    def test_enforcement_fn_does_not_affect_unknown_tool(self):
        r = _make_registry_with_tool()
        r.set_enforcement_fn(lambda n, a, **kw: None)
        result = json.loads(r.dispatch("nonexistent_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_enforcement_fn_fires_for_unknown_tool(self):
        """The fn is invoked BEFORE tool lookup, so unknown names still call it.

        Enforcement is the audit point — every dispatch attempt must be
        observed, including probes for tools that do not exist.  Locks
        in the "Invocation timing" guarantee in the docstring.
        """
        received = []
        def capture(name, args, **kw):
            received.append(name)

        r = _make_registry_with_tool()
        r.set_enforcement_fn(capture)
        json.loads(r.dispatch("test_tool", {}))
        json.loads(r.dispatch("nonexistent_tool", {}))
        assert received == ["test_tool", "nonexistent_tool"]

    def test_enforcement_fn_denial_takes_precedence_over_unknown_tool(self):
        """A denial replaces the "Unknown tool" fallback.

        An enforcement policy that doesn't trust the registry's default
        error response (e.g. wants every unauthorized call to look like
        a policy denial regardless of whether the tool exists) can rely
        on EnforcementDenied short-circuiting before lookup.
        """
        r = _make_registry_with_tool()
        r.set_enforcement_fn(
            lambda n, a, **kw: (_ for _ in ()).throw(
                EnforcementDenied("not in capability list")
            )
        )
        result = json.loads(r.dispatch("nonexistent_tool", {}))
        assert "not in capability list" in result["error"]
        assert "Unknown tool" not in result["error"]

    # -----------------------------------------------------------------------
    # New: async fn rejection, return value contract, re-entrancy
    # -----------------------------------------------------------------------

    def test_async_enforcement_fn_registration_is_rejected(self):
        """Async enforcement fns are rejected at registration time.

        An async fn would silently bypass enforcement: dispatch calls it,
        gets back a coroutine (no exception raised), and the call proceeds.
        Registering one is almost certainly a mistake — fail early and loudly.
        """
        r = _make_registry_with_tool()
        async def bad_fn(name, args, **kw): pass
        with pytest.raises(TypeError, match="synchronous"):
            r.set_enforcement_fn(bad_fn)

    def test_enforcement_fn_return_value_is_ignored(self):
        """Returning a value (including False) from the fn has no effect.

        Only raising EnforcementDenied blocks the call. Returning False,
        a dict, or anything else is silently ignored.
        """
        r = _make_registry_with_tool()
        r.set_enforcement_fn(lambda n, a, **kw: False)
        assert json.loads(r.dispatch("test_tool", {})) == {"result": "ok"}

        r.set_enforcement_fn(lambda n, a, **kw: {"action": "block"})
        assert json.loads(r.dispatch("test_tool", {})) == {"result": "ok"}

        r.set_enforcement_fn(lambda n, a, **kw: "denied")
        assert json.loads(r.dispatch("test_tool", {})) == {"result": "ok"}

    def test_reentrant_dispatch_from_enforcement_fn(self):
        """An enforcement fn may call registry.dispatch for a different tool.

        Concrete policy scenario: 'before allowing write_file, check that the
        agent has a read_file capability' (by dispatching read_file as a probe).
        The registry must not deadlock.
        """
        r = ToolRegistry()
        r.register(
            name="probe_tool",
            toolset="test",
            schema={"name": "probe_tool"},
            handler=lambda args, **kw: json.dumps({"probe": "ok"}),
        )
        r.register(
            name="protected_tool",
            toolset="test",
            schema={"name": "protected_tool"},
            handler=lambda args, **kw: json.dumps({"result": "protected_ok"}),
        )

        def enforcement_fn(name, args, **kw):
            if name == "protected_tool":
                # Re-entrant dispatch — must not deadlock
                probe_result = json.loads(r.dispatch("probe_tool", {}))
                if probe_result.get("probe") != "ok":
                    raise EnforcementDenied("probe failed")

        r.set_enforcement_fn(enforcement_fn)
        result = json.loads(r.dispatch("protected_tool", {}))
        assert result == {"result": "protected_ok"}

    def test_sentinel_vs_bug_distinct_error_messages(self):
        """EnforcementDenied and buggy exceptions produce different error shapes.

        A policy denial should look like a tool result with the denial reason.
        A buggy enforcement fn should look like a system error, not like policy.
        This test locks in the contract so the two cases don't accidentally merge.
        """
        r = _make_registry_with_tool()

        # Policy denial
        r.set_enforcement_fn(
            lambda n, a, **kw: (_ for _ in ()).throw(EnforcementDenied("policy: tool banned"))
        )
        denial = json.loads(r.dispatch("test_tool", {}))

        # Bug in fn
        r.set_enforcement_fn(
            lambda n, a, **kw: (_ for _ in ()).throw(ValueError("unexpected None"))
        )
        bug = json.loads(r.dispatch("test_tool", {}))

        # Policy denial message surfaces directly
        assert "policy: tool banned" in denial["error"]
        # Bug message is hidden; agent sees generic operator error
        assert "unexpected None" not in bug["error"]
        assert denial["error"] != bug["error"]
