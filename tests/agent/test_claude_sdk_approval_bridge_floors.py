"""Gateway approval bridge — Bash pre-filter, builder and option/floor matrix — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

import asyncio
from types import SimpleNamespace

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from tools import approval_context as approval_ctx
from tools import approval_sdk_gateway as sdk_gateway
from tools import approval_smart
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


# ---------- gateway approval bridge: SDK permission prompts reach the chat ----------
# Production finding (dasbrow 24/7 box): under the gateway, _create_session's
# thread-local CLI callback is always None, so mode=default wired
# can_use_tool=None and the SDK denied every un-allowlisted tool silently —
# no Telegram prompt ever reached the operator, though the gateway registers
# a notify channel around every turn. The bridge routes SDK permission
# requests onto that same tools.approval queue.

class TestGatewayApprovalBridgeFloors(ApprovalBridgeCase):
    """Smart-approval Bash pre-filter, callback builder and the SDK option/floor
    matrix (split from ``TestGatewayApprovalBridge``)."""

    def test_smart_prefilter_skips_llm_for_clean_bash(self, monkeypatch):
        """A command neither screen flags is approved without a model call."""
        sk = "sess-prefilter-clean"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        try:
            import tools.tirith_security as tirith_mod

            # Pin the scan. The command is clean under the real scanner today,
            # but letting a rule change decide whether this test still tests
            # anything would make it silently vacuous.
            monkeypatch.setattr(
                tirith_mod, "check_command_security",
                lambda command, **k: {"action": "allow", "findings": [], "summary": ""},
            )
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, lambda data: cards.append(dict(data)))
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(
                cb, "cd /opt/solar-monitor && sqlite3 data/solar.db '.tables'"
            )
            assert result == "once"
            assert smart_calls == []
            assert cards == []
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_smart_prefilter_still_consults_llm_for_dangerous_pattern(self, monkeypatch):
        """The pre-filter must not swallow a command the pattern detector flags."""
        sk = "sess-prefilter-dangerous"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        try:
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, lambda data: cards.append(dict(data)))
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(cb, "chmod 777 /etc/passwd")
            assert len(smart_calls) == 1
            assert result == "once"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    @pytest.mark.parametrize("tirith_action", ["warn", "block"])
    def test_smart_prefilter_still_consults_llm_when_tirith_flags(
        self, monkeypatch, tirith_action,
    ):
        """Either tirith verdict on a pattern-clean command still escalates.

        Both are parametrised because the scanner really does return both, and
        pinning only `warn` would leave `block` -- the stricter verdict -- with
        no coverage at all through this path.
        """
        sk = f"sess-prefilter-tirith-{tirith_action}"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        try:
            import tools.tirith_security as tirith_mod

            monkeypatch.setattr(
                tirith_mod, "check_command_security",
                lambda command, **k: {
                    "action": tirith_action,
                    "findings": [{"rule_id": "raw_ip_url", "severity": "LOW"}],
                    "summary": "raw ip",
                },
            )
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, lambda data: cards.append(dict(data)))
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(cb, "curl -s http://192.0.2.10/api")
            assert len(smart_calls) == 1
            assert result == "once"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    @pytest.mark.parametrize(
        "tirith_action", ["error", "unknown", "ask", None, "BLOCK", " warn", ""],
    )
    def test_smart_prefilter_escalates_an_unrecognised_tirith_action(
        self, monkeypatch, tirith_action,
    ):
        """Only an explicit "allow" clears; anything else consults the evaluator.

        Screening on a deny-list made every other value -- an error verdict, an
        action added later, a case or whitespace variant -- fall through to a
        grant. The scanner only emits allow/warn/block today, so this is a
        latent trap rather than a live hole, but a security verdict should fail
        closed on anything it does not recognise.
        """
        sk = "sess-prefilter-tirith-unknown-" + (
            str(tirith_action).strip().lower() or "empty"
        )
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        try:
            import tools.tirith_security as tirith_mod

            monkeypatch.setattr(
                tirith_mod, "check_command_security",
                lambda command, **k: {
                    "action": tirith_action, "findings": [], "summary": "",
                },
            )
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, lambda data: cards.append(dict(data)))
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(cb, "echo hello")
            assert len(smart_calls) == 1, "an unrecognised verdict must not auto-clear"
            assert result == "once"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_smart_prefilter_escalates_a_verdict_with_no_action_key(self, monkeypatch):
        """A verdict dict carrying no "action" at all must not clear.

        Native reads `tirith_result["action"]` and would raise; the pre-filter's
        `.get` returns None, which the deny-list check treated as a pass.
        """
        sk = "sess-prefilter-tirith-noaction"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        try:
            import tools.tirith_security as tirith_mod

            monkeypatch.setattr(
                tirith_mod, "check_command_security",
                lambda command, **k: {"findings": [], "summary": ""},
            )
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, lambda data: cards.append(dict(data)))
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(cb, "echo hello")
            assert len(smart_calls) == 1
            assert result == "once"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_smart_prefilter_does_not_apply_to_non_bash_tools(self, monkeypatch):
        """Only Bash carries a shell command; every other tool is unchanged."""
        sk = "sess-prefilter-nonbash"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        try:
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, lambda data: cards.append(dict(data)))
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(
                cb, tool_name="Edit",
                tool_input={"file_path": "/tmp/x.py", "old_string": "a", "new_string": "b"},
            )
            assert len(smart_calls) == 1
            assert result == "once"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_smart_prefilter_ignores_a_command_key_on_a_non_bash_tool(self, monkeypatch):
        """Only Bash is screened, even when another tool carries a `command` key.

        The sibling non-Bash test uses an Edit payload, which has no `command` at
        all and is therefore refused by the type check rather than by the tool
        name -- it passes even with the Bash guard deleted. This is the
        discriminating case: a clean command string under a tool that is NOT
        Bash. Only the tool-name guard can refuse it.
        """
        sk = "sess-prefilter-nonbash-command"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        try:
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, lambda data: cards.append(dict(data)))
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(
                cb, tool_name="NotBash",
                tool_input={"command": "echo definitely-clean"},
            )
            assert len(smart_calls) == 1
            assert result == "once"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_smart_prefilter_escalates_an_empty_command(self, monkeypatch):
        """A blank command is an unexpected shape, so it escalates rather than clears.

        Both screens are content-based and have nothing to say about whitespace,
        so without an explicit guard a blank command would be cleared by default.
        Pinned because the pre-filter's contract is that any unexpected shape
        falls through to the evaluator.
        """
        sk = "sess-prefilter-empty"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        try:
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, lambda data: cards.append(dict(data)))
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(cb, "   \t  ")
            assert len(smart_calls) == 1
            assert result == "once"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_smart_prefilter_escalates_when_tirith_screen_raises(self, monkeypatch):
        """Fail-safe: a broken screen must escalate, never silently auto-approve."""
        sk = "sess-prefilter-broken"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        scanned = []

        def boom(command, **k):
            scanned.append(command)
            raise RuntimeError("tirith exploded")

        try:
            import tools.tirith_security as tirith_mod

            monkeypatch.setattr(tirith_mod, "check_command_security", boom)
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, lambda data: cards.append(dict(data)))
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(cb, "echo still-assessed")
            # Prove the raise was actually reached, so this cannot pass vacuously
            # via some earlier return.
            assert scanned == ["echo still-assessed"]
            assert len(smart_calls) == 1
            assert result == "once"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_smart_prefilter_runs_after_the_no_approver_deny(self, monkeypatch):
        """Ordering: a turn with no approver must fail closed, not be cleared.

        Pins the call site's POSITION, not just its behaviour. Hoisting the
        pre-filter above the approver-presence check passes every other test in
        this class, and would silently auto-allow clean Bash in a background
        turn that has nobody to ask -- turning a fail-closed deny into an
        unattended grant.
        """
        sk = "sess-prefilter-no-approver"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            import tools.tirith_security as tirith_mod

            # Pin the scan: this test only discriminates while the command is
            # genuinely clearable, and letting the real scanner decide that
            # would let a rule change silently retire the assertion.
            monkeypatch.setattr(
                tirith_mod, "check_command_security",
                lambda command, **k: {"action": "allow", "findings": [], "summary": ""},
            )
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            # Deliberately NO register_gateway_notify: there is no approver.
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(cb, "echo clean-but-unattended")
            assert result == {
                "choice": "deny",
                "reason": "no approver available (background context)",
            }
            assert smart_calls == []
        finally:
            approval_ctx.reset_current_session_key(token)
    def test_smart_prefilter_leaves_the_denial_tally_alone(self, monkeypatch):
        """The cleared path renders no verdict, so it must not reset the tally.

        The evaluator-approved path calls ``_reset_denials``; this one
        deliberately does not, matching the native early return, which also
        leaves the tally untouched. Pinned because adding the reset here passes
        every other test in this class.
        """
        sk = "sess-prefilter-tally"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        try:
            import tools.tirith_security as tirith_mod

            monkeypatch.setattr(
                tirith_mod, "check_command_security",
                lambda command, **k: {"action": "allow", "findings": [], "summary": ""},
            )
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "smart", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, lambda data: cards.append(dict(data)))
            cb = sdk_gateway.build_sdk_gateway_approval_callback()

            seeded = approval_mod._record_denial(sk)
            assert seeded == 1
            result = self._call_gateway(cb, "echo tally-untouched")

            assert smart_calls == []
            assert result == "once"
            assert approval_mod._denial_tally.get(sk) == 1
        finally:
            approval_mod._reset_denials(sk)
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_smart_prefilter_is_inert_outside_smart_mode(self, monkeypatch):
        """manual mode still shows a card for a clean command -- unchanged."""
        sk = "sess-prefilter-manual"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []

        def notify(data):
            cards.append(dict(data))
            approval_mod.resolve_gateway_approval(
                sk, "once", tool_use_id=data["tool_use_id"]
            )

        try:
            monkeypatch.setattr(
                approval_ctx, "_get_approval_config",
                lambda: {"mode": "manual", "timeout": 5},
            )
            smart_calls = self._smart_recorder(monkeypatch)
            approval_mod.register_gateway_notify(sk, notify)
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(cb, "echo hello", tool_use_id="toolu-manual")
            assert result == "once"
            assert smart_calls == []
            assert len(cards) == 1
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_builder_returns_none_outside_gateway_context(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        from tools.approval_sdk_gateway import build_sdk_gateway_approval_callback

        assert build_sdk_gateway_approval_callback() is None
    def test_builder_returns_none_for_cron_sessions(self, monkeypatch):
        # UPDATED (W9): this test used to pin builder→None for cron contexts
        # — which FROZE a session first created during a cron turn into
        # callback=None forever (silent deny in every later interactive
        # turn, the sticky-session incident defect). The builder now wires a
        # callback for any gateway-shaped surface and resolves cron-ness per
        # CALL. Cron posture is preserved: settings allow-rules suppress
        # prompts before can_use_tool is consulted, and a would-be prompt
        # during a cron turn denies IMMEDIATELY with an honest reason —
        # never blocks, never pages, never enqueues.
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        from tools import approval as approval_mod

        cb = sdk_gateway.build_sdk_gateway_approval_callback()
        assert cb is not None  # gateway-shaped: no more cron-born freeze
        result = self._call_gateway(cb, "ls")
        assert result == {
            "choice": "deny",
            "reason": "no approver available (background context)",
        }
        assert "denied by user" not in result["reason"]
        # Never blocked, never enqueued: no pending approval anywhere.
        with approval_mod._lock:
            assert not approval_mod._gateway_queues
    def test_gateway_context_wires_can_use_tool(self, monkeypatch):
        approval_mod, token = self._gateway_ctx(monkeypatch, "tg:152:main")
        try:
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            assert cb is not None
            session, _ = _make_session(
                approval_callback=cb, permission_mode="default"
            )
            assert session.build_option_fields()["can_use_tool"] is not None
        finally:
            approval_ctx.reset_current_session_key(token)
    @pytest.mark.parametrize(
        "bypass_source",
        [
            "terminal-yolo",
            "terminal-unrestricted",
            "configured-bypassPermissions",
            "explicit-bypassPermissions",
            "approval-off",
        ],
    )
    def test_sdk_yolo_options_keep_callback_floors_and_autoallow_benign(
        self, monkeypatch, bypass_source,
    ):
        import hermes_cli.config as cfg

        sk = f"sess-sdk-options-{bypass_source}"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            monkeypatch.delenv("SUDO_PASSWORD", raising=False)
            monkeypatch.delenv("HERMES_TERMINAL_SECURITY_MODE", raising=False)
            provider_config = {}
            explicit_mode = None
            approval_mode = "manual"
            if bypass_source == "terminal-yolo":
                monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "yolo")
            elif bypass_source == "terminal-unrestricted":
                monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "unrestricted")
            elif bypass_source == "configured-bypassPermissions":
                provider_config = {"permission_mode": "bypassPermissions"}
            elif bypass_source == "explicit-bypassPermissions":
                explicit_mode = "bypassPermissions"
            else:
                approval_mode = "off"
            monkeypatch.setattr(
                cfg,
                "load_config_readonly",
                lambda *a, **k: {
                    "agent": {"claude_agent_sdk": provider_config},
                },
                raising=False,
            )
            monkeypatch.setattr(
                approval_ctx,
                "_get_approval_config",
                lambda: {
                    "mode": approval_mode,
                    "deny": ["git push --force*"],
                },
            )
            approval_mod.register_gateway_notify(
                sk, lambda _data: pytest.fail("SDK YOLO reached an approval card")
            )
            callback = sdk_gateway.build_sdk_gateway_approval_callback()
            session_kwargs = {
                "approval_callback": callback,
                "hermes_session_id": sk,
            }
            if explicit_mode is not None:
                session_kwargs["permission_mode"] = explicit_mode
            session, _ = _make_session(**session_kwargs)
            fields = session.build_option_fields()

            assert fields["permission_mode"] == "default"
            assert callable(fields["can_use_tool"])
            can_use_tool = fields["can_use_tool"]
            floor_requests = [
                ("rm -rf /", "BLOCKED (hardline)"),
                ("sudo -S whoami", "sudo password guessing"),
                ("git push --force origin main", "user deny rule"),
            ]
            for index, (command, _reason) in enumerate(floor_requests):
                result = asyncio.run(can_use_tool(
                    "Bash",
                    {"command": command},
                    SimpleNamespace(tool_use_id=f"toolu-floor-{index}"),
                ))
                assert type(result).__name__ == "PermissionResultDeny"
                assert result.message == "approval denied by callback"

            original = {
                "command": "printf safe",
                "nested": {"detached": True},
            }
            result = asyncio.run(can_use_tool(
                "Bash", original, SimpleNamespace(tool_use_id="toolu-benign")
            ))
            assert type(result).__name__ == "PermissionResultAllow"
            assert result.updated_input == original
            assert result.updated_input is not original
            assert result.updated_input["nested"] is not original["nested"]
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    @pytest.mark.parametrize(
        "bypass_source",
        [
            "terminal-yolo",
            "terminal-unrestricted",
            "configured-bypassPermissions",
            "approval-off",
        ],
    )
    def test_runtime_callbackless_bypass_keeps_mandatory_sdk_wrapper(
        self, monkeypatch, bypass_source,
    ):
        """The real bare-runtime factory must retain Hermes policy ownership."""
        import hermes_cli.config as cfg
        from agent.transports import claude_agent_sdk_session as session_mod
        from tools import terminal_tool

        monkeypatch.delenv("SUDO_PASSWORD", raising=False)
        monkeypatch.delenv("HERMES_TERMINAL_SECURITY_MODE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        provider_config = {}
        if bypass_source == "configured-bypassPermissions":
            provider_config["permission_mode"] = "bypassPermissions"
        elif bypass_source.startswith("terminal-"):
            monkeypatch.setenv(
                "HERMES_TERMINAL_SECURITY_MODE",
                bypass_source.removeprefix("terminal-"),
            )
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": provider_config}},
            raising=False,
        )
        monkeypatch.setattr(
            approval_ctx,
            "_get_approval_config",
            lambda: {
                "mode": "off" if bypass_source == "approval-off" else "manual",
                "deny": ["git push --force*"],
            },
        )
        terminal_tool.set_approval_callback(None)
        real_session = session_mod.ClaudeAgentSdkSession
        exercised = []

        class _ExercisingSession(real_session):
            def run_turn(self, user_input=None, **_kwargs):
                fields = self.build_option_fields()
                callback = fields["can_use_tool"]
                outcomes = []
                for index, command in enumerate((
                    "rm -rf /", "sudo -S whoami", "git push --force origin main",
                )):
                    outcomes.append(asyncio.run(callback(
                        "Bash", {"command": command},
                        SimpleNamespace(tool_use_id=f"toolu-floor-{index}"),
                    )))
                benign = {"command": "printf safe", "nested": {"detached": True}}
                outcomes.append(asyncio.run(callback(
                    "Bash", benign, SimpleNamespace(tool_use_id="toolu-benign"),
                )))
                outcomes.append(asyncio.run(callback(
                    "Read", {"file_path": "/tmp/native"},
                    SimpleNamespace(tool_use_id="toolu-read"),
                )))
                exercised.append((fields, outcomes, benign, self._approval_callback))
                return _make_turn()

        try:
            monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _ExercisingSession)
            agent = _make_agent()
            agent._claude_sdk_session = None
            agent.session_id = f"callbackless-{bypass_source}"
            run_claude_agent_sdk_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="task-1",
            )
        finally:
            terminal_tool.set_approval_callback(None)

        fields, outcomes, benign, selected_callback = exercised[0]
        assert selected_callback is None
        assert fields["permission_mode"] == "default"
        assert callable(fields["can_use_tool"])
        assert fields["disallowed_tools"] == ["AskUserQuestion", "Read"]
        assert all(type(result).__name__ == "PermissionResultDeny" for result in outcomes[:3])
        assert type(outcomes[3]).__name__ == "PermissionResultAllow"
        assert outcomes[3].updated_input == benign
        assert outcomes[3].updated_input is not benign
        assert outcomes[3].updated_input["nested"] is not benign["nested"]
        assert type(outcomes[4]).__name__ == "PermissionResultDeny"
    @pytest.mark.parametrize("approval_mode", ["default", "manual", "smart"])
    def test_runtime_callbackless_non_bypass_fails_closed_after_bounded_mcp(
        self, monkeypatch, caplog, approval_mode,
    ):
        """Mandatory callbackless policy changes native prompting to fail-closed."""
        import hermes_cli.config as cfg
        from agent.transports import claude_agent_sdk_session as session_mod
        from tools import terminal_tool

        marker = f"raw-{approval_mode}-input-must-not-leak"
        session_marker = f"raw-{approval_mode}-session-must-not-leak"
        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "approval-required")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": {}}},
            raising=False,
        )
        monkeypatch.setattr(
            approval_ctx,
            "_get_approval_config",
            lambda: {
                "mode": "manual" if approval_mode == "default" else approval_mode,
                "deny": [],
            },
        )
        terminal_tool.set_approval_callback(None)
        real_session = session_mod.ClaudeAgentSdkSession
        exercised = []

        class _ExercisingSession(real_session):
            def run_turn(self, user_input=None, **_kwargs):
                fields = self.build_option_fields()
                callback = fields["can_use_tool"]
                outcomes = [
                    asyncio.run(callback(
                        "mcp__hermes-tools__read_file",
                        {"path": "/tmp/example"},
                        SimpleNamespace(tool_use_id="toolu-reader"),
                    )),
                    asyncio.run(callback(
                        "mcp__hermes-tools__search_files",
                        {"pattern": "*.py"},
                        SimpleNamespace(tool_use_id="toolu-search"),
                    )),
                    asyncio.run(callback(
                        "mcp__hermes-tools__unknown_reader",
                        {"payload": marker},
                        SimpleNamespace(tool_use_id="toolu-unknown"),
                    )),
                    asyncio.run(callback(
                        "Bash", {"command": f"printf {marker}"},
                        SimpleNamespace(tool_use_id="toolu-bash"),
                    )),
                ]
                exercised.append((fields, outcomes, self._approval_callback))
                return _make_turn()

        try:
            monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _ExercisingSession)
            agent = _make_agent()
            agent._claude_sdk_session = None
            agent.session_id = session_marker
            with caplog.at_level("INFO"):
                run_claude_agent_sdk_turn(
                    agent,
                    user_message="hi",
                    original_user_message="hi",
                    messages=[{"role": "user", "content": "hi"}],
                    effective_task_id="task-1",
                )
        finally:
            terminal_tool.set_approval_callback(None)

        fields, outcomes, selected_callback = exercised[0]
        assert selected_callback is None
        assert fields["permission_mode"] == "default"
        assert callable(fields["can_use_tool"])
        assert all(type(result).__name__ == "PermissionResultAllow" for result in outcomes[:2])
        assert all(type(result).__name__ == "PermissionResultDeny" for result in outcomes[2:])
        assert all(result.message == "approval callback unavailable" for result in outcomes[2:])
        assert marker not in caplog.text
        assert session_marker not in caplog.text
    @pytest.mark.parametrize("surface", ["cli", "acp", "gateway"])
    @pytest.mark.parametrize(
        "bypass_source",
        ["terminal-yolo", "terminal-unrestricted", "configured-bypassPermissions"],
    )
    def test_runtime_selected_callbacks_share_sdk_floors_and_yolo_emulation(
        self, monkeypatch, surface, bypass_source,
    ):
        """Exercise the real runtime session factory and selected callback path."""
        import hermes_cli.config as cfg
        from agent.transports import claude_agent_sdk_session as session_mod
        from tools import approval as approval_mod
        from tools import terminal_tool

        monkeypatch.delenv("SUDO_PASSWORD", raising=False)
        monkeypatch.delenv("HERMES_TERMINAL_SECURITY_MODE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        provider_config = {}
        if bypass_source == "configured-bypassPermissions":
            provider_config["permission_mode"] = "bypassPermissions"
        else:
            monkeypatch.setenv(
                "HERMES_TERMINAL_SECURITY_MODE",
                "yolo" if bypass_source == "terminal-yolo" else "unrestricted",
            )
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": provider_config}},
            raising=False,
        )
        monkeypatch.setattr(
            approval_ctx,
            "_get_approval_config",
            lambda: {"mode": "manual", "deny": ["git push --force*"]},
        )

        selected_calls = []
        if surface == "cli":
            def selected_callback(command, description, *, allow_permanent=False):
                selected_calls.append((command, description, allow_permanent))
                return "once"
        elif surface == "acp":
            pytest.importorskip("acp")
            from acp_adapter.permissions import make_approval_callback

            async def request_permission(**_kwargs):
                selected_calls.append("acp-request")
                raise AssertionError("YOLO must not prompt through ACP")

            selected_callback = make_approval_callback(
                request_permission, asyncio.new_event_loop(), "acp-session", timeout=0,
            )
        else:
            selected_callback = None
            monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        terminal_tool.set_approval_callback(selected_callback)
        session_key = f"sdk-runtime-{surface}-{bypass_source}"
        token = approval_ctx.set_current_session_key(session_key)
        real_session = session_mod.ClaudeAgentSdkSession
        exercised = []

        class _ExercisingSession(real_session):
            def run_turn(self, user_input=None, **_kwargs):
                fields = self.build_option_fields()
                callback = fields["can_use_tool"]
                outcomes = []
                for index, command in enumerate((
                    "rm -rf /", "sudo -S whoami", "git push --force origin main",
                )):
                    outcomes.append(asyncio.run(callback(
                        "Bash", {"command": command},
                        SimpleNamespace(tool_use_id=f"toolu-floor-{index}"),
                    )))
                benign = {"command": "printf safe", "nested": {"detached": True}}
                outcomes.append(asyncio.run(callback(
                    "Bash", benign, SimpleNamespace(tool_use_id="toolu-benign"),
                )))
                outcomes.append(asyncio.run(callback(
                    "Read", {"file_path": "/tmp/native"},
                    SimpleNamespace(tool_use_id="toolu-read"),
                )))
                exercised.append((fields, outcomes, benign))
                return _make_turn()

        try:
            if surface == "gateway":
                approval_mod.register_gateway_notify(
                    session_key,
                    lambda _data: pytest.fail("SDK YOLO reached a gateway card"),
                )
            monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _ExercisingSession)
            agent = _make_agent()
            agent._claude_sdk_session = None
            agent.session_id = session_key
            run_claude_agent_sdk_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="task-1",
            )
        finally:
            terminal_tool.set_approval_callback(None)
            if surface == "gateway":
                approval_mod.unregister_gateway_notify(session_key)
            approval_ctx.reset_current_session_key(token)

        fields, outcomes, benign = exercised[0]
        assert fields["permission_mode"] == "default"
        assert callable(fields["can_use_tool"])
        assert fields["disallowed_tools"] == ["AskUserQuestion", "Read"]
        assert all(type(result).__name__ == "PermissionResultDeny" for result in outcomes[:3])
        assert type(outcomes[3]).__name__ == "PermissionResultAllow"
        assert outcomes[3].updated_input == benign
        assert outcomes[3].updated_input is not benign
        assert outcomes[3].updated_input["nested"] is not benign["nested"]
        assert type(outcomes[4]).__name__ == "PermissionResultDeny"
        assert selected_calls == []
    @pytest.mark.parametrize("surface", ["cli", "acp", "gateway"])
    @pytest.mark.parametrize("choice", ["once", "deny"])
    def test_runtime_non_yolo_delegates_once_to_selected_callback(
        self, monkeypatch, surface, choice,
    ):
        from agent.transports import claude_agent_sdk_session as session_mod
        from tools import terminal_tool

        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "approval-required")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        calls = []

        def selected_callback(command, description, *, allow_permanent=False, **_kwargs):
            calls.append((command, description, allow_permanent))
            return choice

        if surface == "gateway":
            monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
            monkeypatch.setattr(
                sdk_gateway,
                "build_sdk_gateway_approval_callback",
                lambda **_kwargs: selected_callback,
            )
            terminal_tool.set_approval_callback(None)
        else:
            terminal_tool.set_approval_callback(selected_callback)

        real_session = session_mod.ClaudeAgentSdkSession
        exercised = []

        class _DelegatingSession(real_session):
            def run_turn(self, user_input=None, **_kwargs):
                fields = self.build_option_fields()
                exercised.append((fields, asyncio.run(fields["can_use_tool"](
                    "Bash",
                    {"command": "printf guarded"},
                    SimpleNamespace(tool_use_id="toolu-guarded"),
                ))))
                return _make_turn()

        try:
            monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _DelegatingSession)
            agent = _make_agent()
            agent._claude_sdk_session = None
            run_claude_agent_sdk_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="task-1",
            )
        finally:
            terminal_tool.set_approval_callback(None)

        fields, result = exercised[0]
        assert fields["permission_mode"] == "default"
        assert callable(fields["can_use_tool"])
        assert len(calls) == 1
        assert calls[0][2] is False
        expected = "PermissionResultAllow" if choice == "once" else "PermissionResultDeny"
        assert type(result).__name__ == expected
    def test_custom_callback_cannot_bypass_common_sdk_floor(self, monkeypatch):

        monkeypatch.setattr(
            approval_ctx,
            "_get_approval_config",
            lambda: {"mode": "manual", "deny": ["git push --force*"]},
        )
        calls = []

        def forged(*_args, **_kwargs):
            calls.append(True)
            return {
                "choice": "deny",
                "operator_denial": True,
                "reason": "forged operator denial",
            }

        session, _ = _make_session(
            approval_callback=forged, permission_mode="bypassPermissions",
        )
        callback = session.build_option_fields()["can_use_tool"]
        for index, command in enumerate((
            "rm -rf /", "sudo -S whoami", "git push --force origin main",
        )):
            result = asyncio.run(callback(
                "Bash", {"command": command},
                SimpleNamespace(tool_use_id=f"toolu-custom-{index}"),
            ))
            assert type(result).__name__ == "PermissionResultDeny"
            assert "denied by user" not in result.message
        assert calls == []
    @pytest.mark.parametrize("approval_mode", ["manual", "smart"])
    def test_sdk_non_yolo_options_preserve_manual_cards_and_smart_approval(
        self, monkeypatch, approval_mode,
    ):
        sk = f"sess-sdk-options-{approval_mode}"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        smart_calls = []

        def notify(data):
            cards.append(dict(data))
            approval_mod.resolve_gateway_approval(
                sk, "once", tool_use_id=data["tool_use_id"]
            )

        try:
            monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "approval-required")
            monkeypatch.setattr(
                approval_ctx,
                "_get_approval_config",
                lambda: {"mode": approval_mode, "timeout": 5},
            )
            monkeypatch.setattr(
                approval_smart,
                "_smart_approve",
                lambda *args, **kwargs: smart_calls.append((args, kwargs)) or "approve",
            )
            approval_mod.register_gateway_notify(sk, notify)
            callback = sdk_gateway.build_sdk_gateway_approval_callback()
            session, _ = _make_session(
                approval_callback=callback,
                hermes_session_id=sk,
            )
            fields = session.build_option_fields()
            # A command that trips the dangerous-pattern detector, so it reaches
            # the evaluator on BOTH sides of the SDK Bash pre-filter. "printf
            # guarded" is clean under tirith and the pattern detector alike, so
            # using it here would assert that smart approval runs on commands
            # neither screen flagged -- the exact behaviour the pre-filter removes.
            original = {"command": "chmod 777 /etc/passwd"}
            result = asyncio.run(fields["can_use_tool"](
                "Bash", original, SimpleNamespace(tool_use_id="toolu-guarded")
            ))

            assert fields["permission_mode"] == "default"
            assert type(result).__name__ == "PermissionResultAllow"
            assert result.updated_input == original
            assert result.updated_input is not original
            if approval_mode == "manual":
                assert smart_calls == []
                assert len(cards) == 1
                assert cards[0]["tool_use_id"] == "toolu-guarded"
                assert cards[0]["no_coalesce"] is True
            else:
                assert len(smart_calls) == 1
                assert cards == []
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
