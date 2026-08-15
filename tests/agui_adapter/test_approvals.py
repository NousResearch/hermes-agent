import concurrent.futures
from types import SimpleNamespace

from agui_adapter import approvals

# The autouse `_clean_parked_registry` fixture (tests/agui_adapter/conftest.py)
# clears approvals._parked around every test in this package.


def _entry(interrupt_id, status, payload=None):
    return SimpleNamespace(interrupt_id=interrupt_id, status=status, payload=payload)


def test_resolved_approved_once():
    r = approvals.resume_to_decision([_entry("i1", "resolved", {"approved": True})], "i1")
    assert r == "once"


def test_resolved_approved_session_scope():
    r = approvals.resume_to_decision([_entry("i1", "resolved", {"approved": True, "scope": "session"})], "i1")
    assert r == "session"


def test_resolved_not_approved_is_deny():
    r = approvals.resume_to_decision([_entry("i1", "resolved", {"approved": False})], "i1")
    assert r == "deny"


def test_cancelled_is_deny():
    assert approvals.resume_to_decision([_entry("i1", "cancelled")], "i1") == "deny"


def test_unaddressed_interrupt_is_deny():
    assert approvals.resume_to_decision([_entry("other", "resolved", {"approved": True})], "i1") == "deny"


def test_registry_register_take_is_pop():
    fut = concurrent.futures.Future()
    p = approvals.PendingApproval("i1", "rm -rf x", "danger", None, True, fut)
    run = approvals.ParkedRun("t1", object(), p)
    approvals.register(run)
    assert approvals.take("t1") is run
    assert approvals.take("t1") is None  # popped


def test_register_refuses_concurrent_same_thread():
    fut_a = concurrent.futures.Future()
    p_a = approvals.PendingApproval("i1", "rm -rf a", "danger", None, True, fut_a)
    run_a = approvals.ParkedRun("t1", object(), p_a)

    fut_b = concurrent.futures.Future()
    p_b = approvals.PendingApproval("i2", "rm -rf b", "danger", None, True, fut_b)
    run_b = approvals.ParkedRun("t1", object(), p_b)

    assert approvals.register(run_a) is True
    assert approvals.register(run_b) is False
    # The first registration is NOT overwritten by the refused second one.
    assert approvals.take("t1") is run_a


def test_discard_is_identity_scoped():
    fut = concurrent.futures.Future()
    p = approvals.PendingApproval("i1", "rm -rf x", "danger", None, True, fut)
    run_a = approvals.ParkedRun("t1", object(), p)
    approvals.register(run_a)

    other_fut = concurrent.futures.Future()
    other_p = approvals.PendingApproval("i2", "rm -rf y", "danger", None, True, other_fut)
    other_run = approvals.ParkedRun("t1", object(), other_p)

    # Discarding with a mismatched identity leaves the registered run in place.
    approvals.discard("t1", expected=other_run)
    assert approvals.take("t1") is run_a

    # Re-register (take() above popped it) and discard with the correct identity.
    approvals.register(run_a)
    approvals.discard("t1", expected=run_a)
    assert approvals.take("t1") is None


def test_response_schema_excludes_always_when_not_permanent():
    assert approvals.approval_response_schema(False)["properties"]["scope"]["enum"] == ["once", "session"]
    assert approvals.approval_response_schema(True)["properties"]["scope"]["enum"] == ["once", "session", "always"]


def test_approval_callback_emits_interrupt_then_returns_resolved_decision():
    import threading, time

    emitted = []
    cb = approvals.make_approval_callback(
        thread_id="t1",
        emit=emitted.append, queue=object(),
        last_tool_call_id=lambda: "tc-1",
        new_id=lambda p: "int-1",
        timeout=5.0,
    )
    box = {}
    # daemon=True: if an assertion below fails before the future is resolved,
    # the worker stays blocked in decision.result(); a daemon thread does not
    # hold up interpreter shutdown (the finally still resolves it on the normal
    # and most failure paths).
    th = threading.Thread(
        target=lambda: box.__setitem__("d", cb("rm -rf build", "danger")), daemon=True)
    th.start()

    parked = None
    try:
        # Wait for the callback to register the parked run + emit the lifecycle.
        for _ in range(200):
            parked = approvals.take("t1")
            if parked is not None:
                break
            time.sleep(0.01)
        assert parked is not None
        # Registration happens before the single emit() call, so spin briefly
        # until it has landed before indexing into `emitted`.
        for _ in range(200):
            if len(emitted) >= 1:
                break
            time.sleep(0.01)
        assert len(emitted) >= 1
        # Emitted: a single (PARK, interrupt) tuple.
        assert emitted[0][0] is approvals.PARK
        intr = emitted[0][1]
        assert intr.id == "int-1" and intr.reason == "tool_call" and intr.tool_call_id == "tc-1"
        assert "rm -rf build" in (intr.message or "")

        # Resolve → callback returns the decision.
        parked.pending.decision.set_result("session")
        th.join(timeout=2)
        assert box["d"] == "session"
        # Exactly one emit — the single (PARK, interrupt) marker, nothing else.
        assert len(emitted) == 1
    finally:
        # Never let an assertion above strand the worker thread blocked on an
        # unresolved future (it holds no daemon flag): resolve it so the thread
        # unblocks and the interpreter can exit cleanly even on failure.
        if parked is not None and not parked.pending.decision.done():
            parked.pending.decision.set_result("deny")
        th.join(timeout=2)


def test_unexpected_status_is_deny():
    # Any non-"resolved" status (including an unexpected/future one) fails closed.
    r = approvals.resume_to_decision(
        [_entry("i1", "weird", {"approved": True})], "i1"
    )
    assert r == "deny"


def test_approved_non_boolean_is_deny():
    # Only a literal boolean True grants; a truthy non-bool (e.g. the string
    # "false", or an int) must fail closed. Guards the round-3 fail-open fix.
    r = approvals.resume_to_decision(
        [_entry("i1", "resolved", {"approved": "false"})], "i1"
    )
    assert r == "deny"

    r2 = approvals.resume_to_decision(
        [_entry("i1", "resolved", {"approved": 1})], "i1"
    )
    assert r2 == "deny"


def test_resolved_approved_always_scope():
    r = approvals.resume_to_decision(
        [_entry("i1", "resolved", {"approved": True, "scope": "always"})], "i1"
    )
    assert r == "always"


def test_resume_skips_non_matching_entry():
    r = approvals.resume_to_decision(
        [
            _entry("other", "resolved", {"approved": True}),
            _entry("i1", "resolved", {"approved": True}),
        ],
        "i1",
    )
    assert r == "once"


def test_non_dict_payload_is_deny():
    # A non-dict payload (e.g. a bare string or list, from a malformed/hostile
    # resume body) must fail closed to "deny" WITHOUT raising.
    assert approvals.resume_to_decision(
        [SimpleNamespace(interrupt_id="i1", status="resolved", payload="approved")], "i1"
    ) == "deny"
    assert approvals.resume_to_decision(
        [SimpleNamespace(interrupt_id="i1", status="resolved", payload=[1, 2])], "i1"
    ) == "deny"


def test_approval_log_uses_redacted_command(caplog):
    # The "awaiting approval" INFO line must log the secret-redacted, truncated
    # command (the server-side audit record), never the raw command.
    import logging

    from agent.redact import redact_sensitive_text

    cmd = "curl -H 'Authorization: Bearer supersecrettoken123' https://x"
    cb = approvals.make_approval_callback(
        thread_id="t-log", emit=lambda e: None, queue=object(),
        last_tool_call_id=lambda: "tc", new_id=lambda *_: "int", timeout=0.05)
    with caplog.at_level(logging.INFO, logger="agui_adapter.approvals"):
        assert cb(cmd, "danger") == "deny"  # logs, parks, times out -> deny
    logged = " ".join(r.getMessage() for r in caplog.records)
    expected = redact_sensitive_text(cmd, force=True)[:80]
    assert expected in logged  # the log used the force-redacted+truncated form
    if expected != cmd[:80]:   # if redaction scrubbed it, the raw secret is gone
        assert "supersecrettoken123" not in logged


def test_callback_timeout_returns_deny():
    # Fail-closed core ("silence != consent"): with no resume, the callback's
    # decision future times out and it returns "deny", clearing its parked
    # entry — verifying the returned decision VALUE, not just that the registry
    # cleared. Called synchronously (it blocks ~timeout then denies).
    emitted = []
    cb = approvals.make_approval_callback(
        thread_id="t-timeout", emit=emitted.append, queue=object(),
        last_tool_call_id=lambda: "tc", new_id=lambda *_: "int", timeout=0.05)
    assert cb("rm -rf build", "danger") == "deny"
    assert not approvals.is_parked("t-timeout")  # self-cleaned on timeout
    assert emitted and emitted[0][0] is approvals.PARK  # interrupt was surfaced


def test_callback_denies_when_thread_already_parked():
    # Fail-closed at the callback level: if a run is already parked for this
    # thread_id, a second dangerous command on the same thread must deny
    # (register() refuses to overwrite) rather than clobber the pending
    # approval or block. Returns synchronously with no PARK emitted.
    fut = concurrent.futures.Future()
    approvals.register(approvals.ParkedRun(
        "t-busy", object(),
        approvals.PendingApproval("i0", "rm -rf a", "d", None, True, fut)))

    emitted = []
    cb = approvals.make_approval_callback(
        thread_id="t-busy", emit=emitted.append, queue=object(),
        last_tool_call_id=lambda: "tc", new_id=lambda *_: "int", timeout=5.0)
    assert cb("rm -rf b", "danger") == "deny"
    assert emitted == []  # nothing parked → no interrupt emitted


def test_dangerous_command_reaches_interactive_callback_not_gateway_fallback(monkeypatch):
    """Regression guard for the platform="agui" routing bug.

    _run_turn bootstraps a dangerous command's approval by installing a
    thread-local interactive callback (the PARK/interrupt mechanism) and
    calling set_session_vars(session_key=..., async_delivery=False). It MUST
    NOT pass a `platform`: a non-empty platform makes
    tools.approval._is_gateway_approval_context() return True, which diverts
    check_all_command_guards() into the gateway branch. That branch requires a
    registered gateway notify callback (this adapter registers none) and
    otherwise falls through to submit_pending() returning status
    "pending_approval" — so the interactive approval callback (the whole
    feature) NEVER fires and no interrupt is emitted.

    This test drives a real dangerous command through the REAL guard
    dispatcher under a bootstrap that MIRRORS what _run_turn installs
    (set_hermes_interactive_context + set_session_vars with no platform), and
    asserts the interactive callback is invoked (CLI branch) rather than the
    silent gateway fallback. It reconstructs the bootstrap rather than calling
    _run_turn, so it guards the dispatcher-routing contract; the end-to-end
    guard that a regression in _run_turn's OWN bootstrap can't slip through is
    test_e2e_aimock.py::test_interrupt_via_real_guard_dispatcher_end_to_end.
    """
    from tools import approval as _approval
    from gateway.session_context import set_session_vars, clear_session_vars

    # Deterministic manual-approval mode (not off/yolo/smart), so the guard
    # reaches the interactive prompt path rather than auto-approving/denying.
    monkeypatch.setattr(_approval, "_get_approval_mode", lambda: "manual")

    calls = []

    def _cb(command, description, *, allow_permanent=True, **_):
        # Signature mirrors agui_adapter.approvals.make_approval_callback._cb
        # and what prompt_dangerous_approval passes.
        calls.append((command, allow_permanent))
        return "deny"  # fail-closed decision; the assertion is that WE ran

    interactive_token = _approval.set_hermes_interactive_context(True)
    # NOTE: no platform= — exactly what the fixed _run_turn does.
    session_tokens = set_session_vars(session_key="t-guard-route", async_delivery=False)
    try:
        result = _approval.check_all_command_guards(
            "rm -rf build", env_type="local", approval_callback=_cb
        )
    finally:
        clear_session_vars(session_tokens)
        _approval.reset_hermes_interactive_context(interactive_token)

    # The interactive callback fired => the CLI branch was reached.
    assert calls == [("rm -rf build", True)], (
        "approval_callback was not invoked exactly once — the dangerous command "
        "was diverted away from the interactive PARK path (gateway routing regression)"
    )
    # And NOT the gateway submit_pending fallback (which never calls the
    # callback and returns status 'pending_approval').
    assert result.get("status") != "pending_approval"
    assert result.get("approved") is False  # our callback denied
