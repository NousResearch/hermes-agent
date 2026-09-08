"""Session core — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

import logging
from unittest.mock import MagicMock

import pytest

from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
)
from tests.agent.claude_sdk_fakes import (
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    RateLimitInfo,
    RateLimitEvent,
    ResultMessage,
    _make_session,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


class TestSession:
    def test_reported_api_key_source_fails_closed(self):
        session, holder = _make_session(
            script=[
                SystemMessage(
                    data={"apiKeySource": "ANTHROPIC_API_KEY"},
                    session_id="sdk-keyed",
                ),
                ResultMessage(result="must not be delivered"),
            ]
        )
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert "API-key source" in turn.error
        assert turn.final_text == ""
        assert turn.should_retire is True
        assert turn.fatal_reason == "startup"
        assert holder["client"].interrupted is True

    def test_enabled_subscription_overage_fails_closed_before_fallback(self):
        session, holder = _make_session(
            script=[
                RateLimitEvent(
                    RateLimitInfo(
                        overage_status="allowed",
                        raw={"isUsingOverage": False},
                    )
                ),
                ResultMessage(result="must not be delivered"),
            ]
        )
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert "extra usage is enabled" in turn.error
        assert turn.final_text == ""
        assert turn.should_retire is True
        assert turn.fatal_reason == "startup"
        assert holder["client"].interrupted is True

    def test_disabled_overage_and_oauth_source_verify_subscription_lane(self):
        session, _holder = _make_session(
            script=[
                SystemMessage(
                    data={"apiKeySource": "none"},
                    session_id="sdk-subscription",
                ),
                RateLimitEvent(
                    RateLimitInfo(
                        overage_status="rejected",
                        raw={"isUsingOverage": False},
                    )
                ),
                ResultMessage(result="included"),
            ]
        )
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "included"
        assert turn.billing_mode == "subscription_included"
        assert turn.billing_evidence == {
            "api_key_source": "none",
            "is_using_overage": False,
            "overage_status": "rejected",
            "rate_limit_type": "five_hour",
        }

    def test_explicit_metered_opt_in_allows_and_labels_reported_lane(
        self, monkeypatch
    ):
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"allow_metered_key": True}}
            },
            raising=False,
        )
        session, _holder = _make_session(
            script=[
                SystemMessage(data={"apiKeySource": "ANTHROPIC_API_KEY"}),
                RateLimitEvent(
                    RateLimitInfo(
                        rate_limit_type="overage",
                        overage_status="allowed",
                        raw={"isUsingOverage": True},
                    )
                ),
                ResultMessage(result="metered", total_cost_usd=0.25),
            ]
        )
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "metered"
        assert turn.billing_mode == "sdk_reported_metered"
        assert turn.total_cost_usd == 0.25

    def test_empty_turn_rejects_before_start_and_following_turn_works(self):
        session, holder = _make_session(script=[ResultMessage(result="usable")])
        session.request_interrupt()
        try:
            for content in ([], "", "   "):
                turn = session.run_turn(content)
                assert turn.error == turn.final_text
                assert turn.api_call_made is False
                assert not turn.should_retire
            assert holder == {}
            usable = session.run_turn("normal turn")
        finally:
            session.close()
        assert usable.error is None
        assert holder["client"].queried == ["normal turn"]

    def test_successful_mcp_use_does_not_mask_terminal_401(self):
        session, _ = _make_session(script=[
            AssistantMessage(content=[
                ToolUseBlock(
                    id="mcp-1",
                    name="mcp__calendar__list_events",
                    input={},
                )
            ]),
            ResultMessage(
                result="",
                errors=["HTTP 401 Unauthorized"],
                is_error=True,
                subtype="error_during_execution",
                api_error_status=401,
            ),
        ])
        try:
            turn = session.run_turn("list events")
        finally:
            session.close()
        assert "HTTP 401" in (turn.error or "")
        assert turn.should_retire
        assert turn.fatal_reason == "auth"

    @pytest.mark.parametrize("mcp_tool", [True, False])
    def test_interrupted_auth_result_respects_mcp_provenance(self, mcp_tool):
        script = []
        if mcp_tool:
            script.append(AssistantMessage(content=[
                ToolUseBlock(
                    id="mcp-1",
                    name="mcp__calendar__list_events",
                    input={},
                )
            ]))
        script.append(ResultMessage(
            result="",
            errors=["HTTP 401 Unauthorized"],
            is_error=True,
            subtype="error_during_execution",
            api_error_status=401,
        ))
        session, _ = _make_session(script=script)
        original_factory = session._client_factory

        def interrupting_factory(*args, **kwargs):
            client = original_factory(*args, **kwargs)
            original_query = client.query

            async def query_then_interrupt(text):
                await original_query(text)
                session._interrupt_event.set()

            client.query = query_then_interrupt
            return client

        session._client_factory = interrupting_factory
        try:
            turn = session.run_turn("list events")
        finally:
            session.close()
        assert turn.interrupted
        # A successful MCP call is not provenance for a later terminal Claude
        # auth failure, including when the user interrupts the turn.
        assert "setup-token" in (turn.error or "")
        assert turn.should_retire
        assert turn.fatal_reason == "auth"

    def test_quota_429_contradictory_success_surfaces_result_for_fallback(self):
        session, _ = _make_session(script=[ResultMessage(
            result="You've hit your session limit", is_error=True,
            subtype="success", api_error_status=429,
        )])
        try:
            turn = session.run_turn("continue")
        finally:
            session.close()
        assert "HTTP 429" in (turn.error or "")
        assert "session limit" in (turn.error or "").lower()
        # Terminal SDK diagnostics must not become assistant history before the
        # provider-routing layer switches to its fallback.
        assert not turn.final_text

    def test_happy_turn(self):
        script = [
            AssistantMessage(
                content=[ToolUseBlock(id="t1", name="Read", input={"file_path": "/x"})]
            ),
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="data")]),
            AssistantMessage(content=[TextBlock("done reading")]),
            ResultMessage(
                result="done reading",
                usage={"input_tokens": 10, "output_tokens": 5},
            ),
        ]
        session, holder = _make_session(script=script)
        try:
            turn = session.run_turn("read /x please")
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "done reading"
        assert turn.tool_iterations == 1
        assert turn.token_usage_last == {"input_tokens": 10, "output_tokens": 5}
        assert turn.thread_id == "sdk-session-1"
        # assistant(tool_call) + tool + assistant(text)
        assert [m["role"] for m in turn.projected_messages] == [
            "assistant", "tool", "assistant",
        ]
        assert holder["client"].queried == ["read /x please"]
        assert not turn.should_retire

    def test_mixed_text_and_data_image_reaches_sdk_as_native_content(self):
        session, holder = _make_session(script=[ResultMessage(result="a diagram")])
        try:
            turn = session.run_turn([
                {"type": "text", "text": "Inspect this diagram."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aGVsbG8=",
                    },
                },
            ])
        finally:
            session.close()

        assert turn.error is None
        assert holder["client"].queried == [[{
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this diagram."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8=",
                        },
                    },
                ],
            },
            "parent_tool_use_id": None,
        }]]

    def test_image_only_url_reaches_sdk_without_fabricated_prompt(self):
        session, holder = _make_session(script=[ResultMessage(result="a photo")])
        try:
            turn = session.run_turn([{
                "type": "image_url",
                "image_url": {"url": "https://example.test/photo.png"},
            }])
        finally:
            session.close()

        assert turn.error is None
        (query,), = holder["client"].queried
        assert query["message"]["content"] == [{
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.test/photo.png",
            },
        }]

    def test_malformed_image_is_disclosed_instead_of_claimed_attached(
        self, caplog
    ):
        session, holder = _make_session(script=[ResultMessage(result="cannot inspect")])
        with caplog.at_level(logging.WARNING):
            try:
                turn = session.run_turn([{
                    "type": "image_url",
                    "image_url": {"url": "not-an-image-source"},
                }])
            finally:
                session.close()

        assert turn.error is None
        (query,) = holder["client"].queried
        assert "image attachment unavailable" in query.lower()
        assert "what do you see" not in query.lower()
        assert "image attachment" in caplog.text.lower()

    def test_sdk_error_result_surfaces(self):
        script = [
            ResultMessage(
                result="RAW terminal diagnostic",
                subtype="error_max_turns",
                is_error=False,
            )
        ]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert "error_max_turns" in turn.error
        assert not turn.final_text

    def test_contradictory_success_envelope_not_an_error(self):
        # 2026-08-11 incident: the CLI emitted is_error=True with
        # subtype="success" and an empty errors list; the fabricated
        # "SDK result error (subtype=success): success" killed the cron
        # job that had already produced its answer. The contradictory
        # envelope loses to its own subtype. A genuine failure keeps the
        # honest path: non-empty errors (see test_auth_error_marks_retire,
        # same subtype) or a non-success subtype.
        script = [
            AssistantMessage(content=[TextBlock("nightly summary saved")]),
            ResultMessage(
                result="nightly summary saved", is_error=True, subtype="success"
            ),
        ]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("consolidate")
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "nightly summary saved"
        assert not turn.should_retire

    def test_auth_error_marks_retire(self):
        script = [
            ResultMessage(
                subtype="success",
                is_error=True,
                errors=["401 unauthorized: invalid bearer token"],
            )
        ]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.should_retire
        assert "setup-token" in (turn.error or "")

    def test_connect_failure_fails_closed(self):
        session, _ = _make_session(connect_exc=RuntimeError("not logged in"))
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.should_retire
        assert turn.error is not None

    def test_option_fields_shape(self):
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        options = holder["client"].options
        assert options["model"] == "claude-opus-4-8"
        assert options["system_prompt"]["preset"] == "claude_code"
        assert "hermes-tools" in options["mcp_servers"]
        mcp = options["mcp_servers"]["hermes-tools"]
        assert mcp["args"] == [
            "-m",
            "agent.transports.hermes_tools_mcp_server",
            "--profile",
            "claude-agent-sdk",
        ]
        # Hard rule: a metered key never reaches any child of this runtime.
        assert "ANTHROPIC_API_KEY" not in (mcp.get("env") or {})
        # Agent SDK defaults fail closed to `default`: this is the only mode
        # that installs Hermes' approval bridge. The terminal `auto` posture
        # maps here too, so green-field deployments retain that bridge.
        assert options["permission_mode"] == "default"
        # Explicit SDK isolation: None would load ALL of ~/.claude and
        # .claude/settings*, letting ambient settings shadow the gateway's
        # approval posture. The empty list is the SDK's isolation mode.
        assert options["setting_sources"] == []

    def test_native_read_is_disallowed_in_favor_of_bounded_mcp_read(self):
        # Native Read must remain behind Hermes's protected-path-aware bounded
        # MCP read surface under every supported SDK permission mode.
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        fields = session.build_option_fields()
        assert fields["disallowed_tools"] == ["AskUserQuestion", "Read"]
        assert "Bash" not in fields["disallowed_tools"]
        assert "Edit" not in fields["disallowed_tools"]
        assert "Write" not in fields["disallowed_tools"]
        assert "mcp__hermes-tools__read_file" not in fields["disallowed_tools"]

    def test_config_permission_mode_overrides_env_mapping(self, monkeypatch):
        # agent.claude_agent_sdk.permission_mode (an SDK literal) wins over
        # the HERMES_TERMINAL_SECURITY_MODE mapping; explicit constructor
        # arg still wins over both.
        import hermes_cli.config as cfg

        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "unrestricted")
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"permission_mode": "plan"}}
            },
            raising=False,
        )
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["permission_mode"] == "plan"

        explicit = ClaudeAgentSdkSession(
            cwd="/tmp", permission_mode="default", client_factory=MagicMock()
        )
        assert explicit.build_option_fields()["permission_mode"] == "default"

    def test_invalid_config_permission_mode_falls_back(self, monkeypatch):
        # A typo must never silently loosen the posture — it falls back to the
        # SDK's safe default, which retains the Hermes approval bridge.
        import hermes_cli.config as cfg

        monkeypatch.delenv("HERMES_TERMINAL_SECURITY_MODE", raising=False)
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"permission_mode": "yolo"}}
            },
            raising=False,
        )
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["permission_mode"] == "default"

    def test_unknown_terminal_security_mode_falls_back_to_default(self, monkeypatch):
        """Unknown terminal modes must retain the approval-bridge posture."""
        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "unexpected-mode")
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["permission_mode"] == "default"

    def test_empty_config_permission_mode_keeps_env_mapping(self, monkeypatch):
        # "" (the canonical default) = current behavior: the
        # HERMES_TERMINAL_SECURITY_MODE mapping stands.
        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "approval-required")
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["permission_mode"] == "default"

    def test_max_buffer_size_set_above_sdk_default(self):
        # The SDK's 1 MiB default kills the turn outright when one CLI stdout
        # message clears it (2026-08-17 21:03 EDT production kill). The option
        # must always be present — falling back to the SDK default silently
        # reintroduces that failure.
        from agent.transports.claude_agent_sdk_session import (
            _DEFAULT_MAX_BUFFER_SIZE,
        )

        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["max_buffer_size"] == (
            _DEFAULT_MAX_BUFFER_SIZE
        )
        assert _DEFAULT_MAX_BUFFER_SIZE > 1024 * 1024

    def test_config_max_buffer_size_overrides_default(self, monkeypatch):
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"max_buffer_size": 4194304}}
            },
            raising=False,
        )
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["max_buffer_size"] == 4194304

    @pytest.mark.parametrize("bad", [True, "big", 0, -1, None, 1.5, float("inf")])
    def test_invalid_max_buffer_size_falls_back(self, monkeypatch, bad):
        # A typo must not disable the only backstop against an unterminated
        # line growing until the host OOMs, nor silently drop to 1 MiB.
        import hermes_cli.config as cfg
        from agent.transports.claude_agent_sdk_session import (
            _DEFAULT_MAX_BUFFER_SIZE,
        )

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"max_buffer_size": bad}}
            },
            raising=False,
        )
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["max_buffer_size"] == (
            _DEFAULT_MAX_BUFFER_SIZE
        )

    def test_metered_key_scrubbed_from_mcp_env(self, monkeypatch):
        # RED-first: with the ambient var set, the builder must scrub it.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        fields = session.build_option_fields()
        assert "ANTHROPIC_API_KEY" not in fields["mcp_servers"]["hermes-tools"]["env"]

    def test_metered_vectors_neutralized_in_cli_env(self, monkeypatch):
        # The SDK spawns the claude CLI with the FULL parent env and merges
        # options.env ON TOP ({**os.environ, **options.env}), so the scrub
        # must override each present metered vector with "" — a filtered
        # copy could never remove an inherited key. Simulate the SDK merge
        # to prove the neutralization end-to-end.
        import os as _os

        metered = {
            "ANTHROPIC_API_KEY": "sk-ant-api03-fake",
            "ANTHROPIC_AUTH_TOKEN": "fake-bearer",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_USE_VERTEX": "1",
            "AWS_ACCESS_KEY_ID": "AKIAFAKE",
            "AWS_SECRET_ACCESS_KEY": "fake-secret",
            "AWS_SESSION_TOKEN": "fake-session",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/fake-sa.json",
        }
        for key, value in metered.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-subscription")

        session, _ = _make_session(script=[ResultMessage(result="ok")])
        fields = session.build_option_fields()

        # Every present metered vector is overridden to "" (empty = unset
        # for the CLI and the AWS/GCP credential chains).
        for key in metered:
            assert fields["env"][key] == "", key

        # The SDK-side merge — the actual child env — sees them neutralized.
        merged = {**_os.environ, **fields["env"]}
        for key in metered:
            assert merged[key] == "", key

        # Benign keys and the subscription token flow are NOT overridden:
        # absent from options.env, so the inherited values survive the merge.
        for benign in ("HOME", "PATH", "CLAUDE_CODE_OAUTH_TOKEN"):
            assert benign not in fields["env"], benign
        assert merged["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-subscription"

    def test_absent_metered_vectors_are_not_invented(self, monkeypatch):
        # Only PRESENT vectors are overridden — writing "" for absent ones
        # would hand the child empty vars it never had (an empty
        # AWS_ACCESS_KEY_ID can itself break AWS credential chains).
        for key in (
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS",
        ):
            monkeypatch.delenv(key, raising=False)
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["env"] == {}

    def test_allow_metered_key_disables_the_scrub(self, monkeypatch):
        # allow_metered_key: true is the operator's explicit metered opt-in
        # (the startup guard honors it); the scrub must honor it too, or the
        # documented escape hatch would hand the CLI a blanked key.
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"allow_metered_key": True}}
            },
            raising=False,
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["env"] == {}

    def test_metered_key_refuses_startup_fail_closed(self, monkeypatch):
        # The hard rule enforced at the front door: a present metered key
        # must abort the REAL runtime startup path, never silently rebill.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        session = ClaudeAgentSdkSession(cwd="/tmp")  # no factory → real path
        turn = session.run_turn("hi")
        assert turn.should_retire
        assert "ANTHROPIC_API_KEY" in (turn.error or "")
