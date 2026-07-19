"""Unit tests for the claude_cli runtime (Phase 1 + 2a MCP + 2b multi-turn + 2c).

Covers:
  (a) stream-json parser golden test (assistant text + result/usage + is_error)
  (b) clean-env builder (pollution stripped, token injected, HOME/PATH kept)
  (c) api_mode resolution (anthropic_runtime / HERMES_ANTHROPIC_RUNTIME)
  (d) Phase 2a MCP config generation (hermes-tools server + allowed/disallowed)
  (e) Phase 2a tool-event parsing (tool_use + tool_result stream-json sample)
  (f) Phase 2a permission model (no dangerously-skip-permissions; native blocked)
  (g) Phase 2b multi-turn: --session-id create, --resume reuse, mapping
      persistence, missing-session fallback, history seed note

Phase 2c concurrency/aux unit tests live in test_claude_cli_concurrency.py
and test_claude_cli_aux.py.

No live `claude` / network calls.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pytest


@pytest.fixture(autouse=True)
def _claude_cli_slot_dir_for_session_tests(tmp_path, monkeypatch):
    """Redirect host-global claude_cli slots into tmp (never touch ~/.hermes/shared)."""
    monkeypatch.setenv(
        "HERMES_CLAUDE_CLI_SLOT_DIR", str(tmp_path / "claude_cli_slots")
    )

from agent.transports.claude_cli import (
    CLAUDE_CLI_CLEAR_ENV_NAMES,
    CLAUDE_CLI_PERMISSION_MODE,
    CLAUDE_NATIVE_FS_EXEC_TOOLS,
    HERMES_MCP_ALLOWED_TOOLS_GLOB,
    HERMES_MCP_TOOL_PREFIX,
    MCP_SERVER_NAME,
    ClaudeCliClient,
    ClaudeCliError,
    ClaudeCliSpawnConfig,
    build_claude_cli_clean_env,
    build_hermes_mcp_config,
    build_hermes_tools_mcp_server_entry,
    claude_native_disallowed_tools,
    hermes_mcp_allowed_tools,
    parse_stream_json_line,
    strip_hermes_mcp_tool_prefix,
    write_hermes_mcp_config_file,
)
from agent.transports.claude_event_projector import (
    ClaudeEventProjector,
    extract_text_delta,
    extract_tool_result_blocks,
    extract_tool_use_blocks,
    is_hermes_mcp_tool_name,
)
from agent.transports.claude_cli_session import (
    ClaudeCliSession,
    ClaudeCliTurnResult,
    _usage_to_codex_shape,
    hermes_history_has_prior_turns,
    is_resume_missing_error,
    new_claude_session_id,
)
from agent.transports.hermes_tools_mcp_server import (
    AGENT_LOOP_TOOLS_EXCLUDED,
    CLAUDE_EXPOSED_TOOLS,
    CODEX_EXPOSED_TOOLS,
    EXPOSED_TOOLS,
    get_exposed_tools,
)
from hermes_cli import runtime_provider as rp


# ---------------------------------------------------------------------------
# (a) stream-json parser golden test
# ---------------------------------------------------------------------------

# Captured-style JSONL sample approximating `claude -p --output-format
# stream-json --include-partial-messages --verbose` for a pure-text turn.
# Shapes mirror the confirmed result event from the 2026-07-19 step-0 test
# plus the stream_event / assistant envelopes documented by Claude Code.
_GOLDEN_STREAM_JSONL = """
{"type":"system","subtype":"init","session_id":"sess-golden-001","model":"claude-opus-4-8"}
{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}},"session_id":"sess-golden-001"}
{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}},"session_id":"sess-golden-001"}
{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world"}},"session_id":"sess-golden-001"}
{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}},"session_id":"sess-golden-001"}
{"type":"assistant","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"Hello, world!"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":4}},"session_id":"sess-golden-001"}
{"type":"result","subtype":"success","is_error":false,"result":"Hello, world!","session_id":"sess-golden-001","usage":{"input_tokens":12,"output_tokens":4,"cache_read_input_tokens":0},"total_cost_usd":0.0}
""".strip()

_GOLDEN_ERROR_JSONL = """
{"type":"system","subtype":"init","session_id":"sess-err-001"}
{"type":"result","subtype":"error","is_error":true,"result":"OAuth session expired and could not be refreshed","session_id":"sess-err-001","usage":{"input_tokens":0,"output_tokens":0}}
""".strip()


def test_parse_stream_json_line_skips_noise():
    assert parse_stream_json_line("") is None
    assert parse_stream_json_line("   ") is None
    assert parse_stream_json_line("not json") is None
    assert parse_stream_json_line("[1,2,3]") is None  # non-dict
    obj = parse_stream_json_line('{"type":"result","is_error":false}')
    assert obj is not None
    assert obj["type"] == "result"


def test_stream_json_golden_extracts_text_usage_and_success():
    projector = ClaudeEventProjector()
    deltas: list[str] = []
    for line in _GOLDEN_STREAM_JSONL.splitlines():
        state = projector.consume_line(line)
        deltas.extend(state.last_text_deltas)

    state = projector.state
    assert state.finished is True
    assert state.is_error is False
    assert state.final_text == "Hello, world!"
    assert state.session_id == "sess-golden-001"
    assert state.usage is not None
    assert state.usage["input_tokens"] == 12
    assert state.usage["output_tokens"] == 4
    assert state.total_cost_usd == 0.0
    # Streaming path assembled the same text.
    assert "".join(deltas) == "Hello, world!"
    assert state.streamed_text == "Hello, world!"


def test_stream_json_golden_is_error_path():
    projector = ClaudeEventProjector()
    for line in _GOLDEN_ERROR_JSONL.splitlines():
        projector.consume_line(line)
    state = projector.state
    assert state.finished is True
    assert state.is_error is True
    assert "OAuth session expired" in (state.result_text or "")


def test_extract_text_delta_handles_nested_and_flat():
    nested = {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hi"},
        },
    }
    assert extract_text_delta(nested) == "Hi"
    flat = {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "Yo"},
    }
    assert extract_text_delta(flat) == "Yo"
    assert extract_text_delta({"type": "system"}) is None


def test_turn_result_to_normalized_response():
    turn = ClaudeCliTurnResult(
        final_text="ok",
        usage={"input_tokens": 3, "output_tokens": 2, "cache_read_input_tokens": 1},
        session_id="s1",
        is_error=False,
    )
    nr = turn.to_normalized_response()
    assert nr.content == "ok"
    assert nr.finish_reason == "stop"
    assert nr.tool_calls is None
    assert nr.usage is not None
    assert nr.usage.prompt_tokens == 4  # 3 + 1 cache
    assert nr.usage.completion_tokens == 2
    assert nr.provider_data["claude_cli_session_id"] == "s1"


def test_usage_to_codex_shape():
    shaped = _usage_to_codex_shape(
        {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 2}
    )
    assert shaped["inputTokens"] == 10
    assert shaped["outputTokens"] == 5
    assert shaped["cachedInputTokens"] == 2
    assert shaped["totalTokens"] == 17


# ---------------------------------------------------------------------------
# (b) clean-env builder
# ---------------------------------------------------------------------------


def test_clean_env_strips_pollution_injects_token_keeps_home_path():
    polluted = {
        "HOME": "/Users/tester",
        "PATH": "/usr/bin:/bin",
        "USER": "tester",
        "LANG": "en_US.UTF-8",
        "TERM": "xterm-256color",
        "TMPDIR": "/tmp",
        # Pollution that must be stripped (Claude-Code / Anthropic).
        "ANTHROPIC_BASE_URL": "https://evil.example/v1",
        "ANTHROPIC_API_KEY": "sk-ant-api-should-not-leak",
        "ANTHROPIC_TOKEN": "should-not-leak",
        "CLAUDE_CODE_OAUTH_TOKEN": "stale-rotating-or-parent-token",
        "CLAUDECODE": "1",
        "CLAUDE_CODE_ENTRYPOINT": "cli",
        "CLAUDE_AGENT_SDK_VERSION": "9.9.9",
        "CLAUDE_EFFORT": "max",
        "CLAUDE_PREVIEW_FEATURE": "1",
        # Unrelated secret that must also NOT be copied (env -i style).
        "OPENAI_API_KEY": "sk-openai-secret",
        "HERMES_DASHBOARD_SESSION_TOKEN": "dash-secret",
    }
    setup_token = "sk-ant-oat01-TEST_SETUP_TOKEN_NOT_REAL"
    clean = build_claude_cli_clean_env(
        oauth_token=setup_token,
        base_env=polluted,
    )

    assert clean["CLAUDE_CODE_OAUTH_TOKEN"] == setup_token
    assert clean["HOME"] == "/Users/tester"
    assert clean["PATH"] == "/usr/bin:/bin"
    assert clean["USER"] == "tester"
    assert clean["LANG"] == "en_US.UTF-8"
    assert clean["TERM"] == "xterm-256color"
    assert clean["TMPDIR"] == "/tmp"

    for name in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_AGENT_SDK_VERSION",
        "CLAUDE_EFFORT",
        "CLAUDE_PREVIEW_FEATURE",
        "OPENAI_API_KEY",
        "HERMES_DASHBOARD_SESSION_TOKEN",
    ):
        assert name not in clean, f"{name} leaked into clean env"

    # Named clear-set must not appear (except the re-injected oauth token).
    for name in CLAUDE_CLI_CLEAR_ENV_NAMES:
        if name == "CLAUDE_CODE_OAUTH_TOKEN":
            continue
        assert name not in clean


def test_clean_env_rejects_empty_token():
    with pytest.raises(ValueError, match="setup token"):
        build_claude_cli_clean_env(oauth_token="")


def test_build_argv_includes_stream_json_and_model():
    client = ClaudeCliClient(
        oauth_token="sk-ant-oat01-TEST",
        env=build_claude_cli_clean_env(oauth_token="sk-ant-oat01-TEST"),
        claude_bin="/fake/claude",
    )
    cfg = ClaudeCliSpawnConfig(
        model="claude-opus-4-8",
        prompt="ping",
        claude_bin="/fake/claude",
        system_prompt=None,
        enable_hermes_mcp=True,
    )
    argv = client.build_argv(cfg)
    assert argv[0] == "/fake/claude"
    assert "-p" in argv
    assert "--output-format" in argv
    assert "stream-json" in argv
    assert "--include-partial-messages" in argv
    assert "--verbose" in argv
    assert "--setting-sources" in argv
    assert "user" in argv
    # Phase 2a: no unrestricted skip-all; MCP allowlist + permission mode.
    assert "--dangerously-skip-permissions" not in argv
    assert "--mcp-config" in argv
    assert "--allowedTools" in argv
    assert "--disallowedTools" in argv
    assert "--permission-mode" in argv
    assert CLAUDE_CLI_PERMISSION_MODE in argv
    assert "--model" in argv
    assert "claude-opus-4-8" in argv
    assert argv[-1] == "ping"
    client.close()


def test_build_argv_appends_system_prompt_file(tmp_path):
    client = ClaudeCliClient(
        oauth_token="sk-ant-oat01-TEST",
        env=build_claude_cli_clean_env(oauth_token="sk-ant-oat01-TEST"),
        claude_bin="/fake/claude",
    )
    cfg = ClaudeCliSpawnConfig(
        model="claude-opus-4-8",
        prompt="hi",
        system_prompt="You are Hermes.",
        claude_bin="/fake/claude",
        enable_hermes_mcp=True,
    )
    argv = client.build_argv(cfg)
    assert "--append-system-prompt-file" in argv
    idx = argv.index("--append-system-prompt-file")
    path = argv[idx + 1]
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == "You are Hermes."
    client.close()


# ---------------------------------------------------------------------------
# (c) api_mode resolution
# ---------------------------------------------------------------------------


def test_maybe_apply_claude_cli_runtime_from_config(monkeypatch):
    # Explicit enable never depends on auto-eligibility.
    monkeypatch.setattr(rp, "_claude_cli_auto_eligible", lambda **kw: False)
    mode = rp._maybe_apply_claude_cli_runtime(
        provider="anthropic",
        api_mode="anthropic_messages",
        model_cfg={"anthropic_runtime": "claude_cli"},
        model="claude-opus-4-8",
    )
    assert mode == "claude_cli"

    # Unset + not auto-eligible → stay HTTP.
    mode_off = rp._maybe_apply_claude_cli_runtime(
        provider="anthropic",
        api_mode="anthropic_messages",
        model_cfg={},
        model="claude-opus-4-8",
    )
    assert mode_off == "anthropic_messages"

    # Non-anthropic providers never rewrite.
    mode_or = rp._maybe_apply_claude_cli_runtime(
        provider="openrouter",
        api_mode="chat_completions",
        model_cfg={"anthropic_runtime": "claude_cli"},
        model="claude-opus-4-8",
    )
    assert mode_or == "chat_completions"


def test_maybe_apply_claude_cli_runtime_from_env(monkeypatch):
    monkeypatch.setattr(rp, "_claude_cli_auto_eligible", lambda **kw: False)
    monkeypatch.setenv("HERMES_ANTHROPIC_RUNTIME", "claude_cli")
    mode = rp._maybe_apply_claude_cli_runtime(
        provider="anthropic",
        api_mode="anthropic_messages",
        model_cfg={},
        model="claude-opus-4-8",
    )
    assert mode == "claude_cli"

    monkeypatch.setenv("HERMES_ANTHROPIC_RUNTIME", "off")
    mode = rp._maybe_apply_claude_cli_runtime(
        provider="anthropic",
        api_mode="anthropic_messages",
        model_cfg={"anthropic_runtime": "claude_cli"},
        model="claude-opus-4-8",
    )
    assert mode == "anthropic_messages"


def test_maybe_apply_claude_cli_runtime_default_when_token(monkeypatch):
    """UNSET/auto → claude_cli when token+binary available; else HTTP."""
    monkeypatch.delenv("HERMES_ANTHROPIC_RUNTIME", raising=False)

    # Eligible auto → default claude_cli without per-profile flag.
    monkeypatch.setattr(rp, "_claude_cli_auto_eligible", lambda **kw: True)
    mode = rp._maybe_apply_claude_cli_runtime(
        provider="anthropic",
        api_mode="anthropic_messages",
        model_cfg={},  # no anthropic_runtime (e.g. profile og)
        model="claude-opus-4-8",
    )
    assert mode == "claude_cli"

    # auto string same as unset.
    mode_auto = rp._maybe_apply_claude_cli_runtime(
        provider="anthropic",
        api_mode="anthropic_messages",
        model_cfg={"anthropic_runtime": "auto"},
        model="claude-sonnet-4-6",
    )
    assert mode_auto == "claude_cli"

    # Uses model_cfg.default when model kw omitted.
    mode_cfg_default = rp._maybe_apply_claude_cli_runtime(
        provider="anthropic",
        api_mode="anthropic_messages",
        model_cfg={"default": "claude-opus-4-8"},
    )
    assert mode_cfg_default == "claude_cli"

    # Not eligible → HTTP unchanged.
    monkeypatch.setattr(rp, "_claude_cli_auto_eligible", lambda **kw: False)
    mode_http = rp._maybe_apply_claude_cli_runtime(
        provider="anthropic",
        api_mode="anthropic_messages",
        model_cfg={},
        model="claude-opus-4-8",
    )
    assert mode_http == "anthropic_messages"


def test_maybe_apply_claude_cli_runtime_explicit_opt_out(monkeypatch):
    """anthropic_runtime: anthropic_messages / http / api forces HTTP."""
    monkeypatch.delenv("HERMES_ANTHROPIC_RUNTIME", raising=False)
    # Even when auto would qualify, explicit opt-out wins.
    monkeypatch.setattr(rp, "_claude_cli_auto_eligible", lambda **kw: True)
    for opt_out in ("anthropic_messages", "http", "api", "messages", "off"):
        mode = rp._maybe_apply_claude_cli_runtime(
            provider="anthropic",
            api_mode="anthropic_messages",
            model_cfg={"anthropic_runtime": opt_out},
            model="claude-opus-4-8",
        )
        assert mode == "anthropic_messages", opt_out


def test_maybe_apply_claude_cli_runtime_non_claude_model_stays_http(monkeypatch):
    """Auto path requires a Claude model name; non-Claude stays HTTP."""
    monkeypatch.delenv("HERMES_ANTHROPIC_RUNTIME", raising=False)
    # Real eligibility helper — mock only token/bin so model check is live.
    monkeypatch.setattr(rp, "_claude_cli_binary_available", lambda: True)
    monkeypatch.setattr(rp, "_claude_cli_token_resolvable", lambda: True)
    mode = rp._maybe_apply_claude_cli_runtime(
        provider="anthropic",
        api_mode="anthropic_messages",
        model_cfg={},
        model="some-other-model",
    )
    assert mode == "anthropic_messages"


def test_claude_cli_auto_eligible_checks_token_and_binary(monkeypatch):
    """Auto eligibility is Claude model + binary + resolvable setup token."""
    monkeypatch.setattr(rp, "_is_claude_model_name", lambda m: "claude" in (m or "").lower())
    monkeypatch.setattr(rp, "_claude_cli_binary_available", lambda: True)
    monkeypatch.setattr(rp, "_claude_cli_token_resolvable", lambda: True)
    assert rp._claude_cli_auto_eligible(model="claude-opus-4-8") is True

    monkeypatch.setattr(rp, "_claude_cli_token_resolvable", lambda: False)
    assert rp._claude_cli_auto_eligible(model="claude-opus-4-8") is False

    monkeypatch.setattr(rp, "_claude_cli_token_resolvable", lambda: True)
    monkeypatch.setattr(rp, "_claude_cli_binary_available", lambda: False)
    assert rp._claude_cli_auto_eligible(model="claude-opus-4-8") is False

    monkeypatch.setattr(rp, "_claude_cli_binary_available", lambda: True)
    assert rp._claude_cli_auto_eligible(model="") is False


def test_claude_cli_in_valid_api_modes():
    assert "claude_cli" in rp._VALID_API_MODES
    assert rp._parse_api_mode("claude_cli") == "claude_cli"
    assert rp._parse_api_mode("bogus") is None


def test_resolve_runtime_from_pool_entry_applies_claude_cli(monkeypatch):
    """Pool path for anthropic rewrites api_mode when anthropic_runtime set."""
    monkeypatch.delenv("HERMES_ANTHROPIC_RUNTIME", raising=False)
    # Isolate auto path so explicit-enable / opt-out cases are deterministic.
    monkeypatch.setattr(rp, "_claude_cli_auto_eligible", lambda **kw: False)

    class _Entry:
        runtime_api_key = "sk-ant-oat01-TEST"
        access_token = "sk-ant-oat01-TEST"
        runtime_base_url = None
        base_url = "https://api.anthropic.com"
        source = "env:CLAUDE_CODE_OAUTH_TOKEN"

    resolved = rp._resolve_runtime_from_pool_entry(
        provider="anthropic",
        entry=_Entry(),
        requested_provider="anthropic",
        model_cfg={
            "provider": "anthropic",
            "anthropic_runtime": "claude_cli",
            "default": "claude-opus-4-8",
        },
        pool=None,
        target_model="claude-opus-4-8",
    )
    assert resolved["provider"] == "anthropic"
    assert resolved["api_mode"] == "claude_cli"
    assert resolved["api_key"] == "sk-ant-oat01-TEST"

    # Absent anthropic_runtime + not auto-eligible → anthropic_messages.
    resolved2 = rp._resolve_runtime_from_pool_entry(
        provider="anthropic",
        entry=_Entry(),
        requested_provider="anthropic",
        model_cfg={"provider": "anthropic", "default": "claude-opus-4-8"},
        pool=None,
        target_model="claude-opus-4-8",
    )
    assert resolved2["api_mode"] == "anthropic_messages"

    # Absent anthropic_runtime + auto-eligible → claude_cli (desktop / og case).
    monkeypatch.setattr(rp, "_claude_cli_auto_eligible", lambda **kw: True)
    resolved3 = rp._resolve_runtime_from_pool_entry(
        provider="anthropic",
        entry=_Entry(),
        requested_provider="anthropic",
        model_cfg={"provider": "anthropic", "default": "claude-opus-4-8"},
        pool=None,
        target_model="claude-opus-4-8",
    )
    assert resolved3["api_mode"] == "claude_cli"

    # Explicit HTTP opt-out wins over auto-eligible.
    resolved4 = rp._resolve_runtime_from_pool_entry(
        provider="anthropic",
        entry=_Entry(),
        requested_provider="anthropic",
        model_cfg={
            "provider": "anthropic",
            "default": "claude-opus-4-8",
            "anthropic_runtime": "anthropic_messages",
        },
        pool=None,
        target_model="claude-opus-4-8",
    )
    assert resolved4["api_mode"] == "anthropic_messages"


def test_resolve_runtime_provider_override_model_default_on(monkeypatch):
    """Per-call anthropic + Claude override → claude_cli even when config is non-Claude.

    Reproduces the og gap: profile has openai-codex/gpt default and NO
    anthropic_runtime, but ``--provider anthropic --model claude-opus-4-8``
    (or an equivalent desktop runtime override) must still hit default-on.
    """
    monkeypatch.delenv("HERMES_ANTHROPIC_RUNTIME", raising=False)

    # og-style config: non-anthropic default, no anthropic_runtime.
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "default": "gpt-5.6-sol",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
    # Force the env/credential path (not pool) so we exercise the
    # target_model-aware anthropic branch that previously only read config.default.
    monkeypatch.setattr(rp, "load_pool", lambda provider: None)

    seen_models = []

    def _eligible(*, model=None):
        seen_models.append(model)
        return bool(model and "claude" in str(model).lower())

    monkeypatch.setattr(rp, "_claude_cli_auto_eligible", _eligible)

    # Avoid real Anthropic token resolution / network.
    import agent.anthropic_adapter as anth

    monkeypatch.setattr(anth, "resolve_anthropic_token", lambda: "sk-ant-oat01-TEST")

    resolved = rp.resolve_runtime_provider(
        requested="anthropic",
        target_model="claude-opus-4-8",
    )
    assert resolved["provider"] == "anthropic"
    assert resolved["api_mode"] == "claude_cli"
    # Default-on must evaluate the *override* model, not config.default (gpt).
    assert any(m and "claude" in str(m).lower() for m in seen_models)

    # Without target_model, config.default is gpt → auto ineligible → HTTP.
    seen_models.clear()
    resolved_no_override = rp.resolve_runtime_provider(requested="anthropic")
    assert resolved_no_override["api_mode"] == "anthropic_messages"

    # Explicit opt-out still wins even with Claude override + eligible auto.
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "default": "gpt-5.6-sol",
            "provider": "openai-codex",
            "anthropic_runtime": "anthropic_messages",
        },
    )
    resolved_opt_out = rp.resolve_runtime_provider(
        requested="anthropic",
        target_model="claude-opus-4-8",
    )
    assert resolved_opt_out["api_mode"] == "anthropic_messages"


def test_resolve_explicit_runtime_override_model_default_on(monkeypatch):
    """Explicit api_key/base_url path also honors target_model for default-on."""
    monkeypatch.delenv("HERMES_ANTHROPIC_RUNTIME", raising=False)
    monkeypatch.setattr(
        rp,
        "_claude_cli_auto_eligible",
        lambda *, model=None: bool(model and "claude" in str(model).lower()),
    )
    # Config default is non-Claude; only the override is Claude.
    model_cfg = {"default": "gpt-5.6-sol", "provider": "openai-codex"}
    resolved = rp._resolve_explicit_runtime(
        provider="anthropic",
        requested_provider="anthropic",
        model_cfg=model_cfg,
        explicit_api_key="sk-ant-oat01-TEST",
        explicit_base_url="https://api.anthropic.com",
        target_model="claude-opus-4-8",
    )
    assert resolved is not None
    assert resolved["api_mode"] == "claude_cli"

    # Same path without target_model → HTTP (config default not Claude).
    resolved2 = rp._resolve_explicit_runtime(
        provider="anthropic",
        requested_provider="anthropic",
        model_cfg=model_cfg,
        explicit_api_key="sk-ant-oat01-TEST",
        explicit_base_url="https://api.anthropic.com",
    )
    assert resolved2["api_mode"] == "anthropic_messages"


def test_agent_init_accepts_claude_cli_api_mode():
    """AIAgent / agent_init allowlist includes claude_cli."""
    # Read the source of the allowlist check via a minimal object that
    # exercises the same set used in agent_init.
    allowed = {
        "chat_completions",
        "codex_responses",
        "anthropic_messages",
        "bedrock_converse",
        "codex_app_server",
        "claude_cli",
    }
    assert "claude_cli" in allowed


def test_claude_cli_error_str_includes_stderr_tail():
    err = ClaudeCliError(
        message="boom",
        exit_code=1,
        stderr_tail="line1\nline2",
    )
    text = str(err)
    assert "boom" in text
    assert "exit=1" in text
    assert "line1" in text


def test_session_raises_on_is_error_result():
    """ClaudeCliSession.run_turn raises ClaudeCliError when result is_error."""

    class _FakeClient:
        def __init__(self, *a, **k):
            self._lines = [
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "error",
                        "is_error": True,
                        "result": "rate limited",
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                )
                + "\n"
            ]

        def spawn(self, cfg):
            return None

        def iter_stdout_lines(self, timeout=600.0):
            yield from self._lines

        def wait(self, timeout=5.0):
            return 1

        def stderr_tail(self, n=20):
            return ["stderr noise"]

        def close(self):
            pass

        def kill(self):
            pass

    session = ClaudeCliSession(
        oauth_token="sk-ant-oat01-TEST",
        model="claude-opus-4-8",
        client_factory=_FakeClient,
    )
    with pytest.raises(ClaudeCliError, match="rate limited"):
        session.run_turn("hello")
    session.close()


def test_session_success_returns_final_text():
    class _FakeClient:
        def __init__(self, *a, **k):
            self._lines = list(_GOLDEN_STREAM_JSONL.splitlines(keepends=True))

        def spawn(self, cfg):
            return None

        def iter_stdout_lines(self, timeout=600.0):
            yield from self._lines

        def wait(self, timeout=5.0):
            return 0

        def stderr_tail(self, n=20):
            return []

        def close(self):
            pass

        def kill(self):
            pass

    deltas: list[str] = []

    def on_event(note):
        if note.get("method") == "claude/text_delta":
            deltas.append(note["params"]["delta"])

    session = ClaudeCliSession(
        oauth_token="sk-ant-oat01-TEST",
        model="claude-opus-4-8",
        client_factory=_FakeClient,
        on_event=on_event,
    )
    turn = session.run_turn("hello")
    assert turn.final_text == "Hello, world!"
    assert turn.is_error is False
    assert turn.session_id == "sess-golden-001"
    assert turn.usage["input_tokens"] == 12
    assert "".join(deltas) == "Hello, world!"
    nr = turn.to_normalized_response()
    assert nr.content == "Hello, world!"
    assert nr.finish_reason == "stop"
    session.close()


# ---------------------------------------------------------------------------
# (d) Phase 2a MCP config generation
# ---------------------------------------------------------------------------


def test_build_hermes_mcp_config_has_hermes_tools_server():
    cfg = build_hermes_mcp_config(profile="claude", hermes_home="/tmp/hermes-fake-home")
    assert "mcpServers" in cfg
    assert MCP_SERVER_NAME in cfg["mcpServers"]
    entry = cfg["mcpServers"][MCP_SERVER_NAME]
    assert entry["command"]  # sys.executable
    assert entry["args"] == ["-m", "agent.transports.hermes_tools_mcp_server"]
    assert entry["env"]["HERMES_TOOLS_MCP_PROFILE"] == "claude"
    assert entry["env"]["HERMES_QUIET"] == "1"
    assert entry["env"]["HERMES_REDACT_SECRETS"] == "true"
    assert "PYTHONPATH" in entry["env"]
    # Worktree root must be on PYTHONPATH so the module resolves.
    assert "hermes-agent" in entry["env"]["PYTHONPATH"] or entry["env"]["PYTHONPATH"]


def test_build_hermes_tools_mcp_server_entry_skips_pytest_tempdir_hermes_home(tmp_path):
    # Simulate a pytest tempdir HERMES_HOME that must not be baked in.
    entry = build_hermes_tools_mcp_server_entry(
        profile="claude",
        hermes_home=str(tmp_path / "pytest-of-user" / "pytest-0" / "h"),
    )
    assert "HERMES_HOME" not in entry.get("env", {})


def test_write_hermes_mcp_config_file_roundtrip(tmp_path):
    path = write_hermes_mcp_config_file(
        profile="claude",
        hermes_home="/Users/tester/.hermes",
        path=str(tmp_path / "mcp.json"),
    )
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["mcpServers"][MCP_SERVER_NAME]["env"]["HERMES_HOME"] == "/Users/tester/.hermes"


def test_hermes_mcp_allowed_tools_glob_and_explicit():
    globs = hermes_mcp_allowed_tools(use_glob=True)
    assert globs == [HERMES_MCP_ALLOWED_TOOLS_GLOB]
    assert globs[0].startswith("mcp__hermes-tools__")
    explicit = hermes_mcp_allowed_tools(use_glob=False, tool_names=("terminal", "read_file"))
    assert explicit == [
        "mcp__hermes-tools__terminal",
        "mcp__hermes-tools__read_file",
    ]


def test_claude_profile_exposes_terminal_and_excludes_agent_loop_tools():
    claude_tools = get_exposed_tools("claude")
    assert "terminal" in claude_tools
    assert "read_file" in claude_tools
    assert "write_file" in claude_tools
    assert "search_files" in claude_tools
    assert "web_search" in claude_tools
    for excluded in AGENT_LOOP_TOOLS_EXCLUDED:
        assert excluded not in claude_tools
    # Codex profile still omits terminal/fs.
    codex_tools = get_exposed_tools("codex")
    assert "terminal" not in codex_tools
    assert "read_file" not in codex_tools
    assert set(CODEX_EXPOSED_TOOLS) == set(EXPOSED_TOOLS)
    assert set(CLAUDE_EXPOSED_TOOLS) == set(claude_tools)


def test_build_argv_mcp_wiring_permission_model():
    """(d)+(f) argv must wire MCP config, allow Hermes MCP, disallow native fs/exec."""
    client = ClaudeCliClient(
        oauth_token="sk-ant-oat01-TEST",
        env=build_claude_cli_clean_env(oauth_token="sk-ant-oat01-TEST"),
        claude_bin="/fake/claude",
    )
    cfg = ClaudeCliSpawnConfig(
        model="claude-opus-4-8",
        prompt="run terminal",
        claude_bin="/fake/claude",
        enable_hermes_mcp=True,
    )
    argv = client.build_argv(cfg)

    assert "--dangerously-skip-permissions" not in argv
    assert "--mcp-config" in argv
    mcp_idx = argv.index("--mcp-config")
    mcp_path = argv[mcp_idx + 1]
    assert os.path.isfile(mcp_path)
    with open(mcp_path, encoding="utf-8") as fh:
        mcp_cfg = json.load(fh)
    assert MCP_SERVER_NAME in mcp_cfg["mcpServers"]
    server = mcp_cfg["mcpServers"][MCP_SERVER_NAME]
    assert server["args"] == ["-m", "agent.transports.hermes_tools_mcp_server"]
    assert server["env"]["HERMES_TOOLS_MCP_PROFILE"] == "claude"
    # PYTHONPATH must include worktree + (when available) venv site-packages
    # so Claude's MCP child can import the mcp package.
    assert "PYTHONPATH" in server["env"]
    assert server["env"]["PYTHONPATH"]

    assert "--strict-mcp-config" in argv
    assert "--allowedTools" in argv
    allowed_idx = argv.index("--allowedTools")
    # Next tokens until the next flag are allowed tool specs.
    allowed_vals = []
    for tok in argv[allowed_idx + 1 :]:
        if tok.startswith("--"):
            break
        allowed_vals.append(tok)
    assert HERMES_MCP_ALLOWED_TOOLS_GLOB in allowed_vals

    assert "--disallowedTools" in argv
    dis_idx = argv.index("--disallowedTools")
    disallowed_vals = []
    for tok in argv[dis_idx + 1 :]:
        if tok.startswith("--"):
            break
        disallowed_vals.append(tok)
    for native in ("Bash", "Edit", "Write", "Read"):
        assert native in disallowed_vals, f"{native} must be disallowed"

    # Critical: --tools "" drops MCP tools too. Default must omit --tools.
    assert "--tools" not in argv

    assert "--permission-mode" in argv
    pm_idx = argv.index("--permission-mode")
    assert argv[pm_idx + 1] == CLAUDE_CLI_PERMISSION_MODE

    assert "--max-turns" in argv
    client.close()


def test_build_argv_tools_override_only_when_explicit():
    client = ClaudeCliClient(
        oauth_token="sk-ant-oat01-TEST",
        env=build_claude_cli_clean_env(oauth_token="sk-ant-oat01-TEST"),
        claude_bin="/fake/claude",
    )
    cfg = ClaudeCliSpawnConfig(
        model="claude-opus-4-8",
        prompt="x",
        claude_bin="/fake/claude",
        enable_hermes_mcp=True,
        tools_override="",  # explicit opt-in (not recommended)
    )
    argv = client.build_argv(cfg)
    assert "--tools" in argv
    assert argv[argv.index("--tools") + 1] == ""
    client.close()


def test_build_argv_mcp_off_no_dangerously_skip_by_default():
    client = ClaudeCliClient(
        oauth_token="sk-ant-oat01-TEST",
        env=build_claude_cli_clean_env(oauth_token="sk-ant-oat01-TEST"),
        claude_bin="/fake/claude",
    )
    cfg = ClaudeCliSpawnConfig(
        model="claude-opus-4-8",
        prompt="ping",
        claude_bin="/fake/claude",
        enable_hermes_mcp=False,
        dangerously_skip_permissions=False,
    )
    argv = client.build_argv(cfg)
    assert "--dangerously-skip-permissions" not in argv
    assert "--mcp-config" not in argv
    client.close()


def test_claude_native_disallowed_tools_covers_fs_exec():
    denied = set(claude_native_disallowed_tools())
    for name in CLAUDE_NATIVE_FS_EXEC_TOOLS:
        assert name in denied
    assert "Bash" in denied
    assert "Read" in denied
    assert "Write" in denied
    assert "Edit" in denied


# ---------------------------------------------------------------------------
# (e) Phase 2a tool-event parsing — REAL round-trip golden (from live capture)
# ---------------------------------------------------------------------------

# Captured-style stream-json for a completed Hermes MCP tool round-trip:
# message_start → tool_use mcp__hermes-tools__terminal → tool_result with
# real date +%s output → final text_delta + result. Includes the noise that
# used to garble projection (thinking_delta, signature_delta, input_json_delta,
# intermediate assistant snapshots) so the projector must stay clean.
_GOLDEN_TOOL_STREAM_JSONL = """
{"type":"system","subtype":"init","session_id":"sess-tool-001","model":"claude-opus-4-8","mcp_servers":[{"name":"hermes-tools","status":"pending"}],"tools":["ToolSearch"]}
{"type":"stream_event","event":{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[]}},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I should call the terminal tool.","estimated_tokens":12}},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EsICCokBCA8YAipAglG4fTlima5fEBb4ZyhJxz5T"}},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"content_block_stop","index":0},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_term_1","name":"mcp__hermes-tools__terminal","input":{}}},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"date +%s\\"}"}},"session_id":"sess-tool-001"}
{"type":"assistant","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"tool_use","id":"toolu_term_1","name":"mcp__hermes-tools__terminal","input":{"command":"date +%s"}}],"stop_reason":"tool_use","usage":{"input_tokens":40,"output_tokens":20}},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"message_stop"},"session_id":"sess-tool-001"}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_term_1","content":"{\\"result\\":\\"{\\\\\\"output\\\\\\": \\\\\\"1784434336\\\\\\", \\\\\\"exit_code\\\\\\": 0, \\\\\\"error\\\\\\": null}\\"}"}]},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"message_start","message":{"id":"msg_2","role":"assistant","content":[]}},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"1784"}},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"434336"}},"session_id":"sess-tool-001"}
{"type":"assistant","message":{"id":"msg_2","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"1784434336"}],"stop_reason":"end_turn","usage":{"input_tokens":60,"output_tokens":8}},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"content_block_stop","index":0},"session_id":"sess-tool-001"}
{"type":"stream_event","event":{"type":"message_stop"},"session_id":"sess-tool-001"}
{"type":"result","subtype":"success","is_error":false,"result":"1784434336","session_id":"sess-tool-001","usage":{"input_tokens":60,"output_tokens":8,"cache_read_input_tokens":0},"total_cost_usd":0.01,"num_turns":2}
""".strip()


def test_tool_use_and_result_extractors():
    assistant_evt = None
    user_evt = None
    for line in _GOLDEN_TOOL_STREAM_JSONL.splitlines():
        obj = json.loads(line)
        if obj.get("type") == "assistant" and assistant_evt is None:
            # First assistant with tool_use
            content = (obj.get("message") or {}).get("content") or []
            if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content):
                assistant_evt = obj
        if obj.get("type") == "user" and user_evt is None:
            user_evt = obj
    assert assistant_evt is not None and user_evt is not None
    uses = extract_tool_use_blocks(assistant_evt)
    assert len(uses) == 1
    assert uses[0]["name"] == "mcp__hermes-tools__terminal"
    assert uses[0]["input"]["command"] == "date +%s"
    assert is_hermes_mcp_tool_name(uses[0]["name"])
    assert strip_hermes_mcp_tool_prefix(uses[0]["name"]) == "terminal"

    results = extract_tool_result_blocks(user_evt)
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "toolu_term_1"
    assert "1784434336" in results[0]["content"]


def test_stream_json_tool_use_result_projection():
    """Real tool round-trip: clean final text + tool call surfaced."""
    projector = ClaudeEventProjector()
    started: list[str] = []
    completed: list[str] = []
    streamed_deltas: list[str] = []
    for line in _GOLDEN_TOOL_STREAM_JSONL.splitlines():
        state = projector.consume_line(line)
        for rec in state.last_tool_started:
            started.append(rec.name)
        for rec in state.last_tool_completed:
            completed.append(rec.name)
        streamed_deltas.extend(state.last_text_deltas)

    state = projector.state
    assert state.finished is True
    assert state.is_error is False
    # (a) clean projected text — ONLY the timestamp, no thinking/json junk
    assert state.final_text == "1784434336"
    assert "thinking" not in state.final_text.lower()
    assert "signature" not in state.final_text.lower()
    assert "command" not in state.final_text
    assert "<br>" not in state.final_text
    # Live stream deltas assembled the same clean number (after message reset)
    assert "".join(streamed_deltas) == "1784434336"
    # (b) tool call surfaced end-to-end
    assert state.tool_iterations == 1
    assert started == ["terminal"]
    assert completed == ["terminal"]
    assert len(state.tool_calls) == 1
    rec = state.tool_calls[0]
    assert rec.raw_name == "mcp__hermes-tools__terminal"
    assert rec.name == "terminal"
    assert rec.input["command"] == "date +%s"
    assert rec.completed is True
    assert "1784434336" in (rec.result or "")
    assert rec.id == "toolu_term_1"
    assert rec.is_error is False


def test_projector_ignores_thinking_and_json_deltas():
    """thinking_delta / signature_delta / input_json_delta must not garble text."""
    projector = ClaudeEventProjector()
    for line in _GOLDEN_TOOL_STREAM_JSONL.splitlines():
        projector.consume_line(line)
    # streamed_text is only the final message's text_deltas
    assert projector.state.streamed_text == "1784434336"
    assert projector.state.assistant_text == "1784434336"
    assert projector.state.result_text == "1784434336"


def test_projector_resets_stream_on_message_start():
    """Intermediate narration must not concatenate onto the final answer."""
    projector = ClaudeEventProjector()
    events = [
        {
            "type": "stream_event",
            "event": {"type": "message_start", "message": {"role": "assistant"}},
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "I'll use the terminal tool"},
            },
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "I'll use the terminal tool"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "mcp__hermes-tools__terminal",
                        "input": {"command": "date +%s"},
                    },
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "1784434999\n",
                    }
                ]
            },
        },
        {
            "type": "stream_event",
            "event": {"type": "message_start", "message": {"role": "assistant"}},
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "1784434999"},
            },
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "1784434999"}]},
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "1784434999",
        },
    ]
    for ev in events:
        projector.consume(ev)
    assert projector.state.final_text == "1784434999"
    # No double-join of narration + number
    assert "I'll use" not in projector.state.final_text
    assert projector.state.tool_iterations == 1


def test_stream_event_content_block_start_tool_use():
    projector = ClaudeEventProjector()
    line = json.dumps(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_ws_1",
                    "name": "mcp__hermes-tools__web_search",
                    "input": {},
                },
            },
            "session_id": "sess-s",
        }
    )
    state = projector.consume_line(line)
    assert len(state.last_tool_started) == 1
    assert state.last_tool_started[0].name == "web_search"
    assert state.tool_calls[0].raw_name.startswith(HERMES_MCP_TOOL_PREFIX)


def test_session_tool_turn_fires_tool_events_and_counts():
    class _FakeClient:
        def __init__(self, *a, **k):
            self._lines = list(_GOLDEN_TOOL_STREAM_JSONL.splitlines(keepends=True))

        def spawn(self, cfg):
            # Capture that MCP was requested.
            self.spawn_cfg = cfg
            return None

        def iter_stdout_lines(self, timeout=600.0):
            yield from self._lines

        def wait(self, timeout=5.0):
            return 0

        def stderr_tail(self, n=20):
            return []

        def close(self):
            pass

        def kill(self):
            pass

    events: list[dict] = []

    def on_event(note):
        events.append(note)

    session = ClaudeCliSession(
        oauth_token="sk-ant-oat01-TEST",
        model="claude-opus-4-8",
        client_factory=_FakeClient,
        on_event=on_event,
        enable_hermes_mcp=True,
    )
    turn = session.run_turn(
        "Use the terminal tool to run exactly: date +%s — then reply with ONLY the number."
    )
    # Clean final text is the real tool output timestamp
    assert turn.final_text == "1784434336"
    assert turn.tool_iterations == 1
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0]["name"] == "terminal"
    assert turn.tool_calls[0]["raw_name"] == "mcp__hermes-tools__terminal"
    assert turn.tool_calls[0]["arguments"]["command"] == "date +%s"
    assert "1784434336" in (turn.tool_calls[0].get("result") or "")
    methods = [e.get("method") for e in events]
    assert "claude/tool_started" in methods
    assert "claude/tool_completed" in methods
    assert "claude/text_delta" in methods
    assert "claude/assistant_completed" in methods
    started = next(e for e in events if e["method"] == "claude/tool_started")
    assert started["params"]["name"] == "terminal"
    assert started["params"]["raw_name"] == "mcp__hermes-tools__terminal"
    completed = next(e for e in events if e["method"] == "claude/tool_completed")
    assert "1784434336" in (completed["params"].get("result") or "")
    done = next(e for e in events if e["method"] == "claude/assistant_completed")
    assert done["params"]["text"] == "1784434336"
    assert done["params"].get("replace_stream") is True
    # Streamed deltas join to clean number (no thinking/json leakage)
    deltas = [
        e["params"]["delta"]
        for e in events
        if e.get("method") == "claude/text_delta"
    ]
    assert "".join(deltas) == "1784434336"
    session.close()


def test_strip_hermes_mcp_tool_prefix_helpers():
    assert strip_hermes_mcp_tool_prefix("mcp__hermes-tools__terminal") == "terminal"
    assert strip_hermes_mcp_tool_prefix("terminal") == "terminal"
    assert strip_hermes_mcp_tool_prefix("mcp__other__foo") == "foo"
    assert is_hermes_mcp_tool_name("mcp__hermes-tools__read_file")
    assert not is_hermes_mcp_tool_name("Bash")


# ---------------------------------------------------------------------------
# (g) Phase 2b multi-turn session create / resume
# ---------------------------------------------------------------------------


def test_new_claude_session_id_is_uuid():
    sid = new_claude_session_id()
    # Claude requires a valid UUID for --session-id.
    parts = sid.split("-")
    assert len(parts) == 5
    assert len(sid) == 36


def test_build_argv_session_id_on_create():
    client = ClaudeCliClient(
        oauth_token="sk-ant-oat01-TEST",
        env=build_claude_cli_clean_env(oauth_token="sk-ant-oat01-TEST"),
        claude_bin="/fake/claude",
    )
    sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    cfg = ClaudeCliSpawnConfig(
        model="claude-opus-4-8",
        prompt="Remember FALCON-42",
        claude_bin="/fake/claude",
        enable_hermes_mcp=True,
        session_id=sid,
    )
    argv = client.build_argv(cfg)
    assert "--session-id" in argv
    assert argv[argv.index("--session-id") + 1] == sid
    assert "--resume" not in argv
    assert "--no-session-persistence" not in argv
    # MCP + clean surface still present on create.
    assert "--mcp-config" in argv
    assert "--allowedTools" in argv
    assert argv[-1] == "Remember FALCON-42"
    client.close()


def test_build_argv_resume_on_subsequent_turn():
    client = ClaudeCliClient(
        oauth_token="sk-ant-oat01-TEST",
        env=build_claude_cli_clean_env(oauth_token="sk-ant-oat01-TEST"),
        claude_bin="/fake/claude",
    )
    sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    cfg = ClaudeCliSpawnConfig(
        model="claude-opus-4-8",
        prompt="What was the codeword?",
        claude_bin="/fake/claude",
        enable_hermes_mcp=True,
        resume=sid,
        session_id="should-be-ignored-when-resume-set",
    )
    argv = client.build_argv(cfg)
    assert "--resume" in argv
    assert argv[argv.index("--resume") + 1] == sid
    # Resume wins over session_id (mutually exclusive).
    assert "--session-id" not in argv
    assert "--no-session-persistence" not in argv
    assert "--mcp-config" in argv
    assert argv[-1] == "What was the codeword?"
    client.close()


class _CapturingFakeClient:
    """Fake ClaudeCliClient that records spawn configs and yields JSONL."""

    last_cfg: Optional[ClaudeCliSpawnConfig] = None
    configs: list = []
    lines_by_call: list = []

    def __init__(self, *a, **k):
        self._call = len(type(self).configs)

    def spawn(self, cfg):
        type(self).last_cfg = cfg
        type(self).configs.append(cfg)
        return None

    def iter_stdout_lines(self, timeout=600.0):
        idx = len(type(self).configs) - 1
        lines = type(self).lines_by_call[idx] if idx < len(type(self).lines_by_call) else []
        yield from lines

    def wait(self, timeout=5.0):
        return 0

    def stderr_tail(self, n=20):
        return []

    def close(self):
        pass

    def kill(self):
        pass

    @classmethod
    def reset(cls, lines_by_call):
        cls.last_cfg = None
        cls.configs = []
        cls.lines_by_call = list(lines_by_call)


def _success_jsonl(session_id: str, text: str) -> list[str]:
    return [
        json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "session_id": session_id,
                "model": "claude-opus-4-8",
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                },
                "session_id": session_id,
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": text,
                "session_id": session_id,
                "usage": {"input_tokens": 5, "output_tokens": 2},
                "total_cost_usd": 0.0,
            }
        )
        + "\n",
    ]


def test_session_turn1_creates_turn2_resumes_same_id(tmp_path):
    """Turn 1 --session-id; turn 2 --resume with the SAME Claude session id."""
    # Use a fixed returned id so we can assert mapping even if create uuid differs.
    returned_id = "11111111-2222-3333-4444-555555555555"
    _CapturingFakeClient.reset(
        [
            _success_jsonl(returned_id, "OK"),
            _success_jsonl(returned_id, "FALCON-42"),
        ]
    )
    session = ClaudeCliSession(
        oauth_token="sk-ant-oat01-TEST",
        model="claude-opus-4-8",
        cwd=str(tmp_path),
        client_factory=_CapturingFakeClient,
        hermes_conversation_id="hermes-chat-abc",
    )
    t1 = session.run_turn("Remember this codeword: FALCON-42. Reply OK.")
    assert t1.final_text == "OK"
    assert t1.created_session is True
    assert t1.resumed is False
    assert t1.session_id == returned_id
    assert t1.should_retire is False
    assert session.claude_session_id == returned_id
    assert session.turn_index == 1

    cfg1 = _CapturingFakeClient.configs[0]
    assert cfg1.session_id is not None  # generated UUID on create
    assert cfg1.resume is None
    assert cfg1.prompt.startswith("Remember this codeword")
    # System prompt path: create may carry system_prompt (None in this test).
    assert cfg1.system_prompt is None

    t2 = session.run_turn("What was the codeword? Reply with only the codeword.")
    assert t2.final_text == "FALCON-42"
    assert t2.resumed is True
    assert t2.created_session is False
    assert t2.session_id == returned_id
    assert session.claude_session_id == returned_id
    assert session.turn_index == 2

    cfg2 = _CapturingFakeClient.configs[1]
    assert cfg2.resume == returned_id
    assert cfg2.session_id is None
    assert cfg2.prompt.startswith("What was the codeword")
    # Resume must NOT re-append system prompt.
    assert cfg2.system_prompt is None
    # Stable cwd across turns (session file resolution).
    assert cfg1.cwd == cfg2.cwd == str(tmp_path)
    session.close()


def test_session_mapping_persists_on_session_object(tmp_path):
    returned_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    _CapturingFakeClient.reset([_success_jsonl(returned_id, "hi")])
    session = ClaudeCliSession(
        oauth_token="sk-ant-oat01-TEST",
        model="claude-opus-4-8",
        cwd=str(tmp_path),
        client_factory=_CapturingFakeClient,
        hermes_conversation_id="hermes-xyz",
    )
    assert session.claude_session_id is None
    session.run_turn("hello")
    assert session.claude_session_id == returned_id
    assert session.hermes_conversation_id == "hermes-xyz"
    # Mapping survives without re-create until reset.
    assert session.claude_session_id == returned_id
    session.reset_claude_session()
    assert session.claude_session_id is None
    session.close()


def test_resume_missing_session_falls_back_to_create(tmp_path):
    """Missing/expired --resume → clear mapping, create fresh --session-id."""
    missing_id = "deadbeef-dead-beef-dead-beefdeadbeef"
    fresh_id = "feedface-feed-face-feed-facefeedface"

    class _ResumeThenCreateClient:
        calls = 0
        configs: list = []

        def __init__(self, *a, **k):
            pass

        def spawn(self, cfg):
            type(self).configs.append(cfg)
            type(self).calls += 1
            return None

        def iter_stdout_lines(self, timeout=600.0):
            if type(self).calls == 1:
                # Resume attempt → missing session error.
                yield json.dumps(
                    {
                        "type": "result",
                        "subtype": "error",
                        "is_error": True,
                        "result": "No conversation found with session id",
                        "session_id": missing_id,
                    }
                ) + "\n"
            else:
                for line in _success_jsonl(fresh_id, "fresh start"):
                    yield line

        def wait(self, timeout=5.0):
            return 1 if type(self).calls == 1 else 0

        def stderr_tail(self, n=20):
            return ["session not found"] if type(self).calls == 1 else []

        def close(self):
            pass

        def kill(self):
            pass

    _ResumeThenCreateClient.calls = 0
    _ResumeThenCreateClient.configs = []

    session = ClaudeCliSession(
        oauth_token="sk-ant-oat01-TEST",
        model="claude-opus-4-8",
        cwd=str(tmp_path),
        client_factory=_ResumeThenCreateClient,
        claude_session_id=missing_id,  # force resume on first run_turn
        hermes_conversation_id="hermes-resume-miss",
    )
    turn = session.run_turn("continue please")
    assert turn.final_text == "fresh start"
    assert turn.resume_fallback is True
    assert turn.created_session is True
    assert turn.session_id == fresh_id
    assert session.claude_session_id == fresh_id
    # First spawn resumed the dead id; second created a new session.
    assert _ResumeThenCreateClient.configs[0].resume == missing_id
    assert _ResumeThenCreateClient.configs[1].session_id is not None
    assert _ResumeThenCreateClient.configs[1].resume is None
    session.close()


def test_is_resume_missing_error_hints():
    assert is_resume_missing_error("No conversation found with that id")
    assert is_resume_missing_error("session not found")
    assert is_resume_missing_error("Unable to resume session")
    assert not is_resume_missing_error("rate limited")
    assert not is_resume_missing_error("")


def test_hermes_history_has_prior_turns():
    assert hermes_history_has_prior_turns(None) is False
    assert hermes_history_has_prior_turns([]) is False
    assert hermes_history_has_prior_turns([{"role": "system", "content": "x"}]) is False
    assert hermes_history_has_prior_turns([{"role": "user", "content": "hi"}]) is False
    assert (
        hermes_history_has_prior_turns(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ]
        )
        is True
    )


def test_history_seed_note_on_preexisting_history(tmp_path):
    returned_id = "99999999-8888-7777-6666-555555555555"
    _CapturingFakeClient.reset([_success_jsonl(returned_id, "ok")])
    session = ClaudeCliSession(
        oauth_token="sk-ant-oat01-TEST",
        model="claude-opus-4-8",
        cwd=str(tmp_path),
        client_factory=_CapturingFakeClient,
    )
    prior = [
        {"role": "user", "content": "old1"},
        {"role": "assistant", "content": "old2"},
        {"role": "user", "content": "latest"},
    ]
    turn = session.run_turn("latest", messages=prior)
    assert turn.history_seed_note is not None
    assert "pre-seed" in turn.history_seed_note.lower() or "prior" in turn.history_seed_note.lower()
    # Still only sent the latest prompt string (MVP).
    assert _CapturingFakeClient.configs[0].prompt == "latest"
    session.close()


def test_runtime_reuses_agent_session_across_turns(tmp_path, monkeypatch):
    """run_claude_cli_turn keeps agent._claude_cli_session like codex does."""
    from agent import claude_runtime as cr

    returned_id = "abcdef01-2345-6789-abcd-ef0123456789"
    _CapturingFakeClient.reset(
        [
            _success_jsonl(returned_id, "OK"),
            _success_jsonl(returned_id, "FALCON-42"),
        ]
    )

    class _Agent:
        api_key = "sk-ant-oat01-TEST"
        model = "claude-opus-4-8"
        provider = "anthropic"
        base_url = "https://api.anthropic.com"
        session_id = "hermes-sess-1"
        session_cwd = str(tmp_path)
        show_commentary = False
        tool_progress_callback = None
        _session_db = None
        _session_db_created = False
        session_api_calls = 0
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_input_tokens = 0
        session_output_tokens = 0
        session_cache_read_tokens = 0
        session_reasoning_tokens = 0
        session_estimated_cost_usd = 0.0
        session_cost_status = None
        session_cost_source = None
        context_compressor = None
        _claude_cli_session = None

        def _fire_stream_delta(self, *a, **k):
            pass

        def _emit_interim_assistant_message(self, *a, **k):
            pass

        def _sync_external_memory_for_turn(self, **k):
            pass

        def _spawn_background_review(self, **k):
            pass

        def _flush_messages_to_session_db(self, *a, **k):
            pass

        def _ensure_db_session(self):
            pass

    # Force session to use capturing client.
    real_init = ClaudeCliSession.__init__

    def _patched_init(self, *a, **k):
        k = dict(k)
        k["client_factory"] = _CapturingFakeClient
        k.setdefault("cwd", str(tmp_path))
        real_init(self, *a, **k)

    monkeypatch.setattr(ClaudeCliSession, "__init__", _patched_init)
    # Patch the token resolver used by run_claude_cli_turn / ClaudeCliSession.
    monkeypatch.setattr(
        "agent.transports.claude_cli_session.resolve_claude_cli_oauth_token",
        lambda **kw: "sk-ant-oat01-TEST",
    )

    agent = _Agent()
    r1 = cr.run_claude_cli_turn(
        agent,
        user_message="Remember this codeword: FALCON-42. Reply OK.",
        original_user_message="Remember this codeword: FALCON-42. Reply OK.",
        messages=[{"role": "user", "content": "Remember this codeword: FALCON-42. Reply OK."}],
        effective_task_id="t1",
    )
    assert r1["completed"] is True
    assert r1["final_response"] == "OK"
    assert r1["claude_cli_session_id"] == returned_id
    assert agent._claude_cli_session is not None
    assert agent._claude_cli_session.claude_session_id == returned_id

    r2 = cr.run_claude_cli_turn(
        agent,
        user_message="What was the codeword? Reply with only the codeword.",
        original_user_message="What was the codeword? Reply with only the codeword.",
        messages=[
            {"role": "user", "content": "Remember this codeword: FALCON-42. Reply OK."},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "What was the codeword? Reply with only the codeword."},
        ],
        effective_task_id="t1",
    )
    assert r2["completed"] is True
    assert r2["final_response"] == "FALCON-42"
    assert r2["claude_cli_resumed"] is True
    assert r2["claude_cli_session_id"] == returned_id
    # Same session object reused.
    assert agent._claude_cli_session.claude_session_id == returned_id
    assert len(_CapturingFakeClient.configs) == 2
    assert _CapturingFakeClient.configs[0].session_id is not None
    assert _CapturingFakeClient.configs[1].resume == returned_id
    agent._claude_cli_session.close()
