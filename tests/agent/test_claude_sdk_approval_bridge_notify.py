"""Gateway approval bridge — session-scoped approver, notify lifecycle and cards — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from tools import approval_context as approval_ctx
from tools import approval_sdk_gateway as sdk_gateway
from tools import approval_session_notify as session_notify
from tests.agent.claude_sdk_fakes import (
    _make_session,
    _make_turn,
    _make_agent,
    isolate_provider_config,
    ApprovalBridgeCase,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


class TestGatewayApprovalBridgeNotify(ApprovalBridgeCase):
    """Session-scoped approver/notify lifecycle, context snapshots and the
    redaction/truncation of operator cards (split from ``TestGatewayApprovalBridge``)."""

    def test_no_registered_notify_denies(self, monkeypatch, caplog):
        # UPDATED (W8): this test used to pin the bare-"deny" return — which
        # the SDK layer translated to "denied by user" for a prompt no user
        # ever saw, with no log line (the 2026-08-06 incident's approval
        # face). The no-approver deny is now structured with an honest
        # reason and logged at WARNING.
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-no-notify")
        try:
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            with caplog.at_level(logging.WARNING, logger="tools.approval"):
                result = self._call_gateway(cb, "ls")
            assert result == {
                "choice": "deny",
                "reason": "no approver available (background context)",
            }
            messages = [record.getMessage() for record in caplog.records]
            assert messages == [
                "SDK approval request has no approver available; denying "
                "without user attribution"
            ]
            assert "Bash(command=ls)" not in caplog.text
            assert "sess-no-notify" not in caplog.text
        finally:
            approval_ctx.reset_current_session_key(token)
    def test_background_turn_pages_operator_via_session_scoped_approver(
        self, monkeypatch,
    ):
        # The incident lane: deliver_background_results expects CLI-initiated
        # turns BETWEEN hermes turns — exactly when the turn-scoped
        # registration is gone. The session-scoped entry (refreshed by every
        # gateway turn, surviving its teardown) keeps a paging path alive.
        sk = "sess-bg-approver"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            notify, seen = self._resolve_with(approval_mod, sk, "once")
            # The gateway turn registers both; then the turn ends.
            approval_mod.register_gateway_notify(sk, notify)
            session_notify.register_session_notify(sk, notify)
            approval_mod.unregister_gateway_notify(sk)
            try:
                cb = sdk_gateway.build_sdk_gateway_approval_callback()
                assert self._call_gateway(cb, "ls") == "once"
                assert len(seen) == 1  # the operator WAS paged
                assert seen[0]["command"] == "Bash(command=ls)"
            finally:
                session_notify.unregister_session_notify(sk)
        finally:
            approval_ctx.reset_current_session_key(token)
    def test_no_approver_deny_is_not_attributed_to_user(
        self, monkeypatch, caplog,
    ):
        # End to end across the widened channel: bridge (no approver) →
        # _make_can_use_tool → PermissionResultDeny carrying the honest
        # reason. "denied by user" is reserved for the trusted bridge's
        # structural operator-denial result.
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-bg-honest")
        try:
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            session, _ = _make_session(
                approval_callback=cb, permission_mode="default"
            )
            fn = session._make_can_use_tool()
            with caplog.at_level(logging.WARNING, logger="tools.approval"):
                res = asyncio.run(fn("Bash", {"command": "ls"}, None))
            assert type(res).__name__ == "PermissionResultDeny"
            assert res.message == "no approver available (background context)"
            assert "denied by user" not in res.message
            assert any(
                r.getMessage() == (
                    "SDK approval request has no approver available; denying "
                    "without user attribution"
                )
                for r in caplog.records
            )
            assert "sess-bg-honest" not in caplog.text
            assert "Bash(command=ls)" not in caplog.text

            # Plain callback denies are not trusted operator attribution. The
            # structural marker is accepted only from the registered bridge.
            session2, _ = _make_session(
                approval_callback=lambda *a, **k: "deny",
                permission_mode="default",
            )
            res2 = asyncio.run(session2._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
            assert res2.message == "approval denied by callback"
            session3, _ = _make_session(
                approval_callback=lambda *a, **k: "once",
                permission_mode="default",
            )
            res3 = asyncio.run(session3._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
            assert type(res3).__name__ == "PermissionResultAllow"
        finally:
            approval_ctx.reset_current_session_key(token)
    def test_session_scoped_entry_lifecycle(self, monkeypatch):
        # Leak guard: the entry survives turn teardown (the feature), dies at
        # the conversation boundary (clear_session — the gateway's boundary
        # funnel) and at shutdown (clear_all_session_notify); re-register is
        # idempotent.
        sk = "sess-lifecycle"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            notify, _ = self._resolve_with(approval_mod, sk, "once")
            approval_mod.register_gateway_notify(sk, notify)
            session_notify.register_session_notify(sk, notify)
            approval_mod.unregister_gateway_notify(sk)
            assert sk in session_notify._session_notify_cbs  # survives the turn

            # Idempotent refresh: latest cb wins, still a single entry.
            def other(_data):
                pass

            session_notify.register_session_notify(sk, other)
            assert session_notify._session_notify_cbs[sk] is other

            # Conversation boundary removes it; the bridge then denies
            # honestly instead of paging a rotated-away session.
            approval_mod.clear_session(sk)
            assert sk not in session_notify._session_notify_cbs
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(cb, "ls")
            assert result["reason"] == "no approver available (background context)"

            # Unknown-key unregister is a no-op; clear-all empties.
            session_notify.unregister_session_notify("never-registered")
            session_notify.register_session_notify(sk, notify)
            session_notify.clear_all_session_notify()
            assert session_notify._session_notify_cbs == {}
        finally:
            session_notify.unregister_session_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_unanswered_and_failed_prompts_carry_honest_reasons(
        self, monkeypatch,
    ):
        # The model must never hear "denied by user" for a prompt no user
        # answered: timeout, notify-failure and /deny <reason> each carry
        # their own truth.
        sk = "sess-honest-reasons"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            cb = sdk_gateway.build_sdk_gateway_approval_callback()

            # Timeout: the operator was paged but never answered.
            monkeypatch.setattr(
                approval_ctx, "_get_approval_timeout", lambda: 0.0
            )
            paged = []
            session_notify.register_session_notify(sk, paged.append)
            result = self._call_gateway(cb, "sleep")
            assert result == {
                "choice": "deny",
                "reason": "approval timed out — no operator response",
            }
            assert len(paged) == 1

            # Notify failure: the prompt never reached the operator.
            def broken(_data):
                raise RuntimeError("adapter send failed")

            session_notify.register_session_notify(sk, broken)
            result = self._call_gateway(cb, "x")
            assert result["choice"] == "deny"
            assert "notify failed" in result["reason"]
            assert "denied by user" not in result["reason"]

            # /deny <reason>: a REAL user deny — attributed, in their words.
            # Restore a sane timeout: the 0.0 above would hit the deadline
            # break before the wait ever observes the (already-set) event.
            monkeypatch.setattr(
                approval_ctx, "_get_approval_timeout", lambda: 5.0
            )

            def deny_with_reason(_data):
                approval_mod.resolve_gateway_approval(
                    sk, "deny", reason="not now"
                )

            session_notify.register_session_notify(sk, deny_with_reason)
            result = self._call_gateway(cb, "y")
            assert result == {
                "choice": "deny",
                "operator_denial": True,
                "reason": "not now",
            }
        finally:
            session_notify.unregister_session_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_cron_born_session_approves_in_later_interactive_turn(
        self, monkeypatch,
    ):
        # Sticky-session freeze (incident defect 3): the SDK session and its
        # approval callback are frozen at creation; a session FIRST created
        # during a cron turn got callback=None forever — every
        # un-allowlisted tool silently denied even in later interactive
        # turns, until a session retire. Per-call resolution: the SAME
        # callback object denies honestly during cron turns and pages the
        # operator normally once an interactive turn refreshes the context.
        from tools import approval as approval_mod

        sk = "sess-cron-born"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        # What the cron turn's per-turn refresh writes into the holder.
        holder = {"gateway": False, "session_key": ""}
        cb = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: dict(holder)
        )
        # RED pre-fix: the builder returned None for cron contexts, which
        # is exactly the freeze.
        assert cb is not None

        # Prompt during the cron turn: immediate honest deny — no paging,
        # no blocking, posture preserved.
        result = self._call_gateway(cb, "ls")
        assert result["reason"] == "no approver available (background context)"

        # Later INTERACTIVE turn on the SAME SDK session: the runtime's
        # per-turn refresh rewrites the holder; the gateway registers its
        # turn notify. No session retire happened.
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        holder.update({"gateway": True, "session_key": sk})
        notify, seen = self._resolve_with(approval_mod, sk, "once")
        approval_mod.register_gateway_notify(sk, notify)
        try:
            # Invoke from a FRESH thread — the SDK loop-thread reality:
            # contextvars invisible, so the holder must carry the context.
            out = {}
            t = threading.Thread(
                target=lambda: out.update(r=self._call_gateway(cb, "uname"))
            )
            t.start()
            t.join(timeout=10)
            assert out.get("r") == "once"
            assert len(seen) == 1
            assert seen[0]["command"] == "Bash(command=uname)"
        finally:
            approval_mod.unregister_gateway_notify(sk)
    def test_context_provider_allows_bounded_additive_fields(self, monkeypatch):
        """Internal context evolution must not disable an otherwise valid approver."""
        from tools import approval as approval_mod

        sk = "sess-additive-context"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        callback = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: {
                "gateway": True,
                "session_key": sk,
                "turn_kind": "interactive",
            },
        )
        notify, seen = self._resolve_with(approval_mod, sk, "once")
        approval_mod.register_gateway_notify(sk, notify)
        try:
            out = {}
            thread = threading.Thread(
                target=lambda: out.update(result=self._call_gateway(callback, "uname"))
            )
            thread.start()
            thread.join(timeout=10)
        finally:
            approval_mod.unregister_gateway_notify(sk)

        assert out.get("result") == "once"
        assert [item["command"] for item in seen] == ["Bash(command=uname)"]
    def test_context_provider_rejects_oversized_additive_state(
        self, monkeypatch, caplog,
    ):
        """Additive compatibility remains bounded and payload-safe."""
        from tools import approval as approval_mod

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        marker = "OVERSIZED_CONTEXT_SECRET"
        provided = {"gateway": True, "session_key": "sess-oversized-context"}
        provided.update({f"extra_{index}": marker for index in range(7)})
        callback = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: provided,
        )
        notify, seen = self._resolve_with(
            approval_mod, provided["session_key"], "once"
        )
        approval_mod.register_gateway_notify(provided["session_key"], notify)
        try:
            with caplog.at_level(logging.DEBUG, logger="tools.approval"):
                result = self._call_gateway(callback, "uname")
        finally:
            approval_mod.unregister_gateway_notify(provided["session_key"])

        assert result == {
            "choice": "deny",
            "reason": "no approver available (background context)",
        }
        assert seen == []
        assert marker not in caplog.text
    def test_silent_denies_logged_with_tool_and_reason(
        self, monkeypatch, caplog,
    ):
        # P2.d: every deny that transits the SDK lane WITHOUT an operator
        # tap must be observable — the incident's silent denies had no log
        # line at all. The choke point is _make_can_use_tool; "denied by
        # user" is the trustworthy operator-attribution prefix (W8/W11)
        # and is deliberately NOT logged as silent.
        SILENT = "silent deny (no operator choice)"

        def _records(cl):
            return [r for r in cl.records if SILENT in r.getMessage()]

        def _deny_via(callback, cl):
            session, _ = _make_session(
                approval_callback=callback, permission_mode="default",
                hermes_session_id="sess-w13",
            )
            with cl.at_level(
                logging.INFO,
                logger="agent.transports.claude_agent_sdk_session",
            ):
                return asyncio.run(
                    session._make_can_use_tool()("Bash", {"command": "x"}, None)
                )

        # Class 1 — no-approver, via the REAL bridge (nothing registered).
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-w13-none")
        try:
            caplog.clear()
            res = _deny_via(
                sdk_gateway.build_sdk_gateway_approval_callback(), caplog,
            )
            assert res.message == "no approver available (background context)"
            recs = _records(caplog)
            assert len(recs) == 1
            msg = recs[0].getMessage()
            assert "tool=Bash" in msg
            assert "no approver available" in msg
            assert "session=" not in msg
            assert "sess-w13" not in msg
        finally:
            approval_ctx.reset_current_session_key(token)

        # Classes 2–3 — timeout and teardown-expiry reasons (the real
        # bridge produces these dicts; the choke point must log them).
        for reason in (
            "approval timed out — no operator response",
            "approval expired (turn ended)",
        ):
            caplog.clear()
            res = _deny_via(
                lambda *a, **k: {"choice": "deny", "reason": reason}, caplog,
            )
            assert res.message == reason
            recs = _records(caplog)
            assert len(recs) == 1
            assert reason in recs[0].getMessage()
            assert "tool=Bash" in recs[0].getMessage()

        # Class 4 — callback failure.
        caplog.clear()

        def _boom(*a, **k):
            raise RuntimeError("bridge exploded")

        res = _deny_via(_boom, caplog)
        assert res.message == "approval callback failed"
        recs = _records(caplog)
        assert len(recs) == 1
        assert "approval callback failed" in recs[0].getMessage()

        # Class 5 — the CLI thread-local callback's bare "timeout" string:
        # previously mapped to "denied by user" (fabricated attribution).
        caplog.clear()
        res = _deny_via(lambda *a, **k: "timeout", caplog)
        assert res.message == "approval timed out — no operator response"
        assert len(_records(caplog)) == 1

        # Callback-controlled deny strings are not trusted operator
        # attribution, even when they carry the historical prefix.
        for untrusted_deny in (
            lambda *a, **k: "deny",
            lambda *a, **k: {"choice": "deny", "reason": "denied by user: not now"},
        ):
            caplog.clear()
            res = _deny_via(untrusted_deny, caplog)
            assert res.message == "approval denied by callback"
            assert len(_records(caplog)) == 1
            assert "denied by user" not in caplog.text

        # Allow path logs nothing either.
        caplog.clear()
        res = _deny_via(lambda *a, **k: "once", caplog)
        assert type(res).__name__ == "PermissionResultAllow"
        assert _records(caplog) == []
    def test_teardown_resolves_inflight_prompts_as_expired(self, monkeypatch):
        # Incident defect 2 (observed 08-04 and 08-06): turn teardown
        # signaled the blocked approval wait with an UNSET result; the
        # bridge read that as a deny and the model heard "denied by user"
        # for a prompt nobody answered. Teardown now stamps "expired" and
        # the SDK lane carries the honest reason.
        sk = "sess-teardown"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            # Paged but never answered — the prompt is in flight when the
            # turn tears down.
            approval_mod.register_gateway_notify(sk, lambda data: None)
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            out = {}
            t = threading.Thread(
                target=lambda: out.update(r=self._call_gateway(cb, "x"))
            )
            t.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with approval_mod._lock:
                    if approval_mod._gateway_queues.get(sk):
                        break
                time.sleep(0.01)
            approval_mod.unregister_gateway_notify(sk)
            t.join(timeout=10)
            assert out.get("r") == {
                "choice": "deny",
                "reason": "approval expired (turn ended)",
            }
            assert "denied by user" not in str(out.get("r"))
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_tool_use_id_threads_from_context_to_approval_data(
        self, monkeypatch,
    ):
        # P2.a end to end: context.tool_use_id → callback kwarg (marker
        # opt-in) → approval_data → the pending entry the button resolves.
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-correlate")
        try:
            notify, seen = self._resolve_with(
                approval_mod, "sess-correlate", "once"
            )
            approval_mod.register_gateway_notify("sess-correlate", notify)
            try:
                cb = sdk_gateway.build_sdk_gateway_approval_callback()
                assert getattr(cb, "_accepts_tool_use_id", False) is True
                assert self._call_gateway(cb, "a", tool_use_id="toolu_T") == "once"
                assert seen[0]["tool_use_id"] == "toolu_T"
            finally:
                approval_mod.unregister_gateway_notify("sess-correlate")

            # Session layer: a marker-bearing callback receives the SDK
            # context's id...
            got = {}

            def marked(command, description, *, allow_permanent=False,
                       tool_use_id=""):
                got["tool_use_id"] = tool_use_id
                return "once"

            marked._accepts_tool_use_id = True
            session, _ = _make_session(
                approval_callback=marked, permission_mode="default"
            )
            ctx_obj = SimpleNamespace(tool_use_id="toolu_CTX")
            res = asyncio.run(
                session._make_can_use_tool()("Bash", {"command": "x"}, ctx_obj)
            )
            assert type(res).__name__ == "PermissionResultAllow"
            assert got["tool_use_id"] == "toolu_CTX"

            # ...and a marker-less (CLI-style) callback keeps its exact
            # signature — invoked without the kwarg, no TypeError.
            calls = {}

            def plain(command, description, *, allow_permanent=False):
                calls["ok"] = True
                return "deny"

            session2, _ = _make_session(
                approval_callback=plain, permission_mode="default"
            )
            res2 = asyncio.run(
                session2._make_can_use_tool()(
                    "Bash", {"command": "true"}, ctx_obj,
                )
            )
            assert calls["ok"] is True
            assert type(res2).__name__ == "PermissionResultDeny"
        finally:
            approval_ctx.reset_current_session_key(token)
    def test_no_context_deny_is_honest(self, monkeypatch, caplog):
        # A background prompt on a session whose latest turn context is
        # empty (no gateway, no key) must deny with the honest reason —
        # never the "denied by user" lie, never silently.

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        cb = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: {}
        )
        assert cb is not None
        out = {}
        with caplog.at_level(logging.WARNING, logger="tools.approval"):
            t = threading.Thread(
                target=lambda: out.update(r=self._call_gateway(
                    cb, tool_name="Read", tool_input={"file_path": "/x"},
                ))
            )
            t.start()
            t.join(timeout=10)
        assert out.get("r") == {
            "choice": "deny",
            "reason": "no approver available (background context)",
        }
        assert [record.getMessage() for record in caplog.records] == [
            "SDK approval request has no approver available; denying "
            "without user attribution"
        ]
        assert "Read" not in caplog.text
    def test_turn_refreshes_sdk_approval_context_snapshot(self, monkeypatch):
        # Runtime seam: the holder is rewritten at the TOP of
        # run_claude_agent_sdk_turn on every call — that per-turn refresh is
        # what un-freezes a cron-born session.
        import hermes_cli.config as cfg

        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input, **kw):
                return _make_turn()

            def close(self):
                pass

        monkeypatch.setattr(
            "agent.transports.claude_agent_sdk_session.ClaudeAgentSdkSession",
            SpySession,
        )
        monkeypatch.setattr(
            cfg, "load_config_readonly", lambda *a, **k: {}, raising=False
        )
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

        agent = _make_agent()
        agent._claude_sdk_session = None
        token = approval_ctx.set_current_session_key("turn-key-1")
        try:
            run_claude_agent_sdk_turn(
                agent, user_message="hi", original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="t",
            )
        finally:
            approval_ctx.reset_current_session_key(token)
        assert agent._sdk_approval_turn_ctx == {
            "gateway": True, "session_key": "turn-key-1",
        }
        assert captured.get("approval_callback") is not None  # bridge wired

        # Second call under a DIFFERENT key: the snapshot is rewritten (the
        # refresh runs before any session logic; forcing re-creation keeps
        # the spy simple — the refresh itself is call-scoped, not
        # creation-scoped).
        agent._claude_sdk_session = None
        token = approval_ctx.set_current_session_key("turn-key-2")
        try:
            run_claude_agent_sdk_turn(
                agent, user_message="again", original_user_message="again",
                messages=[{"role": "user", "content": "again"}],
                effective_task_id="t",
            )
        finally:
            approval_ctx.reset_current_session_key(token)
        assert agent._sdk_approval_turn_ctx == {
            "gateway": True, "session_key": "turn-key-2",
        }
    def test_embedded_prefixed_secret_is_hidden_on_actual_sdk_gateway_card(
        self, monkeypatch,
    ):
        session_key = "sess-embedded-tool-secret"
        secret = "sk-1234567890ABCDEF"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    f"Odd-{secret}", {"payload": "must-not-render"}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        assert len(seen) == 1
        rendered = " ".join(str(value) for value in seen[0].values())
        assert secret not in rendered
        assert "must-not-render" not in rendered
        assert seen[0]["command"] == "SDK tool unknown"
    def test_embedded_secret_guard_survives_global_redaction_opt_out(
        self, monkeypatch,
    ):
        from agent import redact as redact_mod

        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", False)
        session_key = "sess-redaction-disabled"
        secret = "sk-1234567890ABCDEF"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    f"Odd-{secret}", {"payload": "must-not-render"}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        rendered = " ".join(str(value) for value in seen[0].values())
        assert secret not in rendered
        assert seen[0]["command"] == "SDK tool unknown"
    @pytest.mark.parametrize("redaction_enabled", [True, False])
    def test_prefix_at_identity_offset_zero_always_collapses_to_unknown(
        self, monkeypatch, redaction_enabled,
    ):
        from agent import redact as redact_mod

        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", redaction_enabled)
        session_key = f"sess-offset-zero-{redaction_enabled}"
        secret = "sk-1234567890ABCDEF"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    secret, {"payload": "must-not-render"}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        assert seen[0]["command"] == "SDK tool unknown"
        assert secret not in " ".join(str(value) for value in seen[0].values())
    @pytest.mark.parametrize("hostile_kind", ["get", "str"])
    def test_hostile_context_provider_result_is_fixed_fail_closed(
        self, monkeypatch, caplog, hostile_kind,
    ):
        marker = f"CTX_RESULT_SECRET_{hostile_kind}"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        if hostile_kind == "get":
            class HostileContext(dict):
                def get(self, *_args, **_kwargs):
                    raise RuntimeError(marker)

            provided = HostileContext(gateway=True, session_key="safe")
        else:
            class HostileKey:
                def __str__(self):
                    return marker

            provided = {"gateway": True, "session_key": HostileKey()}


        callback = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: provided,
        )
        with caplog.at_level(logging.DEBUG, logger="tools.approval"):
            result = self._call_gateway(callback, "true")
        assert result == {
            "choice": "deny",
            "reason": "no approver available (background context)",
        }
        assert marker not in caplog.text
    def test_long_bash_card_discloses_truncation_and_dangerous_tail(
        self, monkeypatch,
    ):
        session_key = "sess-long-bash-tail"
        # Keep this below the immutable root-delete floor: this regression is
        # about bounded manual-card presentation, not hardline authorization.
        command = "printf safe " + ("A" * 1_000) + "; rm -rf ./build"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    "Bash", {"command": command}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        card_command = seen[0]["command"]
        assert "[truncated]" in card_command
        assert "; rm -rf ./build" in card_command
        assert len(card_command.encode("utf-8")) <= 512
    def test_long_path_card_discloses_truncation_and_target_tail(
        self, monkeypatch,
    ):
        session_key = "sess-long-path-tail"
        path = "/tmp/" + ("A" * 1_000) + "/dangerous-target"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    "Write", {"file_path": path, "content": "must-not-render"}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        card_command = seen[0]["command"]
        assert "[truncated]" in card_command
        assert "/dangerous-target" in card_command
        assert "must-not-render" not in card_command
        assert len(card_command.encode("utf-8")) <= 512
    def test_redaction_expansion_does_not_discard_actual_bash_tail(
        self, monkeypatch,
    ):
        session_key = "sess-redaction-expansion-tail"
        # Exercise redaction expansion on the manual path without weakening
        # the unconditional root-delete floor.
        command = ("API_KEY=x " * 6_000) + "; rm -rf ./build"
        assert len(command.encode("utf-8")) < 64 * 1024
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    "Bash", {"command": command}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        card_command = seen[0]["command"]
        assert "[truncated]" in card_command
        assert "; rm -rf ./build" in card_command
        assert "API_KEY=x" not in card_command
        assert len(card_command.encode("utf-8")) <= 512
    def test_redaction_expansion_does_not_discard_actual_path_tail(
        self, monkeypatch,
    ):
        session_key = "sess-path-redaction-expansion-tail"
        path = ("API_KEY=x " * 6_000) + "dangerous-target"
        assert len(path.encode("utf-8")) < 64 * 1024
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    "Write", {"file_path": path, "content": "must-not-render"}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        card_command = seen[0]["command"]
        assert "[truncated]" in card_command
        assert "dangerous-target" in card_command
        assert "API_KEY=x" not in card_command
        assert "must-not-render" not in card_command
        assert len(card_command.encode("utf-8")) <= 512
