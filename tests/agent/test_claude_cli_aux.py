"""Phase 2c/2d aux handling for claude_cli — compression/title skip + no HTTP aux.

Confirms:
  * Hermes HTTP context compression is skipped for api_mode=claude_cli
    (Claude owns native compaction via --resume)
  * Title generation skips the failing Anthropic HTTP aux path
  * Failures do not error the main turn (skip returns cleanly)
  * Central invariant: claude_cli main runtime never constructs an HTTP
    Anthropic aux client (extra-usage 400 path) — auto + explicit provider
  * model_switch / switch_model preserve api_mode=claude_cli when
    anthropic_runtime is opted in

No live network / claude calls.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import conversation_compression as cc
from agent import title_generator as tg
from agent import auxiliary_client as aux


class _AgentStub:
    api_mode = "claude_cli"
    model = "claude-opus-4-8"
    provider = "anthropic"
    session_id = "sess-test"
    _cached_system_prompt = "sys-prompt"
    compression_enabled = False
    context_compressor = SimpleNamespace(
        should_compress=lambda *_a, **_k: True,
        threshold_tokens=1000,
        context_length=200_000,
        compression_count=0,
    )

    def _build_system_prompt(self, system_message=None):
        return system_message or self._cached_system_prompt


def test_compress_context_skips_for_claude_cli(caplog):
    agent = _AgentStub()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "more context " * 50},
    ]
    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        out_msgs, out_prompt = cc.compress_context(
            agent, messages, system_message="sys"
        )
    assert out_msgs is messages  # unchanged identity
    assert out_prompt == "sys-prompt"
    assert any(
        "skipping Hermes HTTP context compression" in r.message for r in caplog.records
    )


def test_compress_context_skip_is_api_mode_gated(caplog):
    """claude_cli skip only fires when api_mode is claude_cli."""
    agent = _AgentStub()
    agent.api_mode = "claude_cli"
    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        cc.compress_context(agent, [{"role": "user", "content": "x"}], None)
    assert any("claude_cli" in r.message for r in caplog.records)

    # Different api_mode must not emit the claude_cli skip line.
    caplog.clear()
    agent.api_mode = "anthropic_messages"
    # Intercept before any aux work: replace compress_context body after the
    # claude_cli guard by short-circuiting on a custom compressor that
    # raises if entered — we only need the skip log absence.
    class _BoomCompressor:
        @staticmethod
        def _automatic_compression_blocked(_self):
            # Returning True makes compress_context return early without aux.
            return True

    agent.context_compressor = _BoomCompressor()
    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        out_msgs, _ = cc.compress_context(
            agent, [{"role": "user", "content": "x"}], None
        )
    assert len(out_msgs) == 1
    assert not any(
        "skipping Hermes HTTP context compression" in r.message for r in caplog.records
    )


def test_generate_title_skips_claude_cli_runtime(caplog, monkeypatch):
    # If skip fails, call_llm would be invoked — make it explode.
    def _boom(**_kw):
        raise AssertionError("call_llm must not run for claude_cli title")

    monkeypatch.setattr(tg, "call_llm", _boom)
    monkeypatch.setattr(tg, "_auto_title_enabled", lambda: True)

    with caplog.at_level(logging.INFO, logger="agent.title_generator"):
        title = tg.generate_title(
            "user hello",
            "assistant world",
            main_runtime={
                "model": "claude-opus-4-8",
                "provider": "anthropic",
                "api_mode": "claude_cli",
            },
        )
    assert title is None
    assert any("claude_cli runtime" in r.message for r in caplog.records)


def test_generate_title_still_calls_llm_for_other_modes(monkeypatch):
    class _Resp:
        class choices:
            pass

    class _Choice:
        class message:
            content = "My Title"

    _Resp.choices = [type("C", (), {"message": type("M", (), {"content": "My Title"})()})()]

    called = {}

    def _fake_call_llm(**kw):
        called["yes"] = True
        return _Resp()

    monkeypatch.setattr(tg, "call_llm", _fake_call_llm)
    monkeypatch.setattr(tg, "_auto_title_enabled", lambda: True)
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.strip_think_blocks",
        lambda _self, content: content,
    )

    title = tg.generate_title(
        "user hello",
        "assistant world",
        main_runtime={
            "model": "grok-4.3",
            "provider": "xai-oauth",
            "api_mode": "chat_completions",
        },
    )
    assert called.get("yes") is True
    assert title == "My Title"


def test_claude_cli_runtime_turn_does_not_raise_on_aux_skip(tmp_path, monkeypatch):
    """Main turn completes even when title/compression would have failed.

    Integration-ish: run_claude_cli_turn with a fake session succeeds;
    generate_title skip is independent and non-fatal.
    """
    from agent import claude_runtime as cr
    from agent.transports.claude_cli import ClaudeCliSpawnConfig
    from agent.transports.claude_cli_session import ClaudeCliSession

    monkeypatch.setenv(
        "HERMES_CLAUDE_CLI_SLOT_DIR", str(tmp_path / "claude_cli_slots")
    )

    class _FakeClient:
        def __init__(self, **kw):
            pass

        def spawn(self, cfg: ClaudeCliSpawnConfig):
            return None

        def iter_stdout_lines(self, timeout=None):
            sid = "22222222-2222-2222-2222-222222222222"
            yield '{"type":"system","subtype":"init","session_id":"%s"}' % sid
            yield (
                '{"type":"result","subtype":"success","is_error":false,'
                '"result":"done","session_id":"%s",'
                '"usage":{"input_tokens":2,"output_tokens":1}}' % sid
            )

        def wait(self, timeout=None):
            return 0

        def stderr_tail(self, n=20):
            return []

        def close(self):
            pass

    real_init = ClaudeCliSession.__init__

    def _patched_init(self, *a, **k):
        k = dict(k)
        k["client_factory"] = lambda **kw: _FakeClient(**kw)
        k.setdefault("cwd", str(tmp_path))
        real_init(self, *a, **k)

    monkeypatch.setattr(ClaudeCliSession, "__init__", _patched_init)
    monkeypatch.setattr(
        "agent.transports.claude_cli_session.resolve_claude_cli_oauth_token",
        lambda **kw: "sk-ant-oat01-TEST",
    )

    agent = SimpleNamespace(
        api_mode="claude_cli",
        model="claude-opus-4-8",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_key="sk-ant-oat01-TEST",
        session_id="s1",
        session_cwd=str(tmp_path),
        system_prompt="sys",
        show_commentary=True,
        tool_progress_callback=None,
        _session_db=None,
        _session_db_created=False,
        session_api_calls=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0,
        session_cost_status=None,
        session_cost_source=None,
        context_compressor=None,
        log_prefix="",
        quiet_mode=True,
    )

    def _sync(**k):
        pass

    def _spawn(**k):
        pass

    def _fire(*a, **k):
        pass

    agent._sync_external_memory_for_turn = _sync
    agent._spawn_background_review = _spawn
    agent._fire_stream_delta = _fire
    agent._emit_interim_assistant_message = _fire
    agent._flush_messages_to_session_db = _fire
    agent._ensure_db_session = _fire

    result = cr.run_claude_cli_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
        effective_task_id="t1",
    )
    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert result.get("error") is None

    # Title skip remains non-fatal.
    title = tg.generate_title(
        "hello",
        "done",
        main_runtime={"api_mode": "claude_cli", "provider": "anthropic"},
    )
    assert title is None


# ── Phase 2d: central no-HTTP-anthropic-aux invariant ──────────────────────


def test_is_claude_cli_runtime_active_from_main_runtime():
    assert aux._is_claude_cli_runtime_active(
        {"api_mode": "claude_cli", "provider": "anthropic"}
    )
    assert not aux._is_claude_cli_runtime_active(
        {"api_mode": "anthropic_messages", "provider": "anthropic"}
    )


def test_try_anthropic_refuses_http_when_claude_cli_main_runtime(monkeypatch):
    """Spy: build_anthropic_client must never run for claude_cli main."""
    built = {"n": 0}

    def _boom(*_a, **_k):
        built["n"] += 1
        raise AssertionError("HTTP Anthropic client must not be built")

    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        _boom,
        raising=False,
    )
    # Also prevent config/env from interfering: only main_runtime matters.
    monkeypatch.delenv("HERMES_ANTHROPIC_RUNTIME", raising=False)
    monkeypatch.setattr(
        aux,
        "_is_claude_cli_runtime_active",
        lambda main_runtime=None: (
            isinstance(main_runtime, dict)
            and str(main_runtime.get("api_mode") or "").lower() == "claude_cli"
        ),
    )

    client, model = aux._try_anthropic(
        explicit_api_key="sk-ant-oat01-TEST",
        main_runtime={"api_mode": "claude_cli", "provider": "anthropic"},
    )
    assert client is None
    assert model is None
    assert built["n"] == 0


def test_resolve_provider_client_anthropic_refuses_claude_cli(monkeypatch):
    """Explicit provider=anthropic still refuses HTTP under claude_cli."""
    built = {"n": 0}

    def _boom_build(*_a, **_k):
        built["n"] += 1
        raise AssertionError("must not build HTTP anthropic")

    monkeypatch.setenv("HERMES_ANTHROPIC_RUNTIME", "claude_cli")

    with patch(
        "agent.anthropic_adapter.build_anthropic_client",
        side_effect=_boom_build,
    ):
        client, model = aux.resolve_provider_client(
            "anthropic",
            model="claude-haiku-4-5-20251001",
            main_runtime={
                "api_mode": "claude_cli",
                "provider": "anthropic",
                "model": "claude-opus-4-8",
            },
        )
    assert client is None
    assert model is None
    assert built["n"] == 0


def test_resolve_auto_skips_http_anthropic_for_claude_cli(monkeypatch, caplog):
    """auto aux with claude_cli main_runtime must not land on Anthropic HTTP."""
    # Force Step-1 skip path; make fallback chain empty so we see the skip.
    monkeypatch.setattr(aux, "_try_configured_fallback_chain", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(aux, "_try_main_fallback_chain", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(aux, "_get_provider_chain", lambda: [])
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda *_a, **_k: False)

    def _must_not_try_anthropic(*_a, **_k):
        raise AssertionError("_try_anthropic must not be called from auto Step-1")

    monkeypatch.setattr(aux, "_try_anthropic", _must_not_try_anthropic)

    with caplog.at_level(logging.INFO, logger="agent.auxiliary_client"):
        client, model = aux._resolve_auto(
            main_runtime={
                "provider": "anthropic",
                "model": "claude-opus-4-8",
                "api_mode": "claude_cli",
                "api_key": "sk-ant-oat01-TEST",
                "base_url": "https://api.anthropic.com",
            }
        )
    assert client is None
    assert model is None
    assert any(
        "skipping HTTP Anthropic main provider" in r.message for r in caplog.records
    )


def test_switch_model_preserves_claude_cli_api_mode(monkeypatch):
    """In-place model switch must not drop claude_cli → anthropic_messages."""
    from agent import agent_runtime_helpers as arh

    monkeypatch.setenv("HERMES_ANTHROPIC_RUNTIME", "claude_cli")

    agent = SimpleNamespace(
        model="claude-opus-4-8",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_mode="claude_cli",
        api_key="sk-ant-oat01-TEST",
        client=None,
        _anthropic_client=None,
        _anthropic_api_key="sk-ant-oat01-TEST",
        _anthropic_base_url="https://api.anthropic.com",
        _is_anthropic_oauth=True,
        _config_context_length=None,
        _client_kwargs={},
        _credential_pool=None,
        _claude_cli_session=None,
        compression_enabled=False,
        context_compressor=None,
        _primary_runtime={},
        max_tokens=None,
        request_overrides={},
        quiet_mode=True,
        _use_prompt_caching=False,
        _use_native_cache_layout=False,
        reasoning_config=None,
    )
    agent._anthropic_prompt_cache_policy = lambda **_k: (False, False)
    agent._ensure_lmstudio_runtime_loaded = lambda: None

    monkeypatch.setattr(
        arh,
        "get_provider_request_timeout",
        lambda *_a, **_k: None,
        raising=False,
    )
    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        lambda *_a, **_k: None,
        raising=False,
    )

    # switch_model imports determine_api_mode which returns anthropic_messages;
    # our re-apply of _maybe_apply_claude_cli_runtime must restore claude_cli.
    arh.switch_model(
        agent,
        new_model="claude-sonnet-4-6",
        new_provider="anthropic",
        api_key="sk-ant-oat01-TEST",
        base_url="https://api.anthropic.com",
        api_mode="",  # force determine_api_mode path
    )
    assert agent.api_mode == "claude_cli", (
        f"switch_model dropped claude_cli → {agent.api_mode!r} "
        "(HTTP anthropic extra-usage path)"
    )
    assert agent.client is None
    assert agent._anthropic_client is None
    assert agent.model == "claude-sonnet-4-6"


def test_model_switch_resolve_preserves_claude_cli(monkeypatch):
    """hermes_cli.model_switch host_mandated must not strip claude_cli."""
    from hermes_cli.runtime_provider import _maybe_apply_claude_cli_runtime

    # Simulate the post-host_mandated re-apply that model_switch now does.
    api_mode = "anthropic_messages"  # what host_mandated returns for api.anthropic.com
    monkeypatch.setenv("HERMES_ANTHROPIC_RUNTIME", "claude_cli")
    restored = _maybe_apply_claude_cli_runtime(
        provider="anthropic",
        api_mode=api_mode,
        model_cfg={"anthropic_runtime": "claude_cli"},
    )
    assert restored == "claude_cli"


def test_claude_cli_conversation_makes_no_http_anthropic_aux_call(monkeypatch, tmp_path):
    """Regression: full-lifecycle multi-turn claude_cli → zero HTTP anthropic aux.

    Constructs a lightweight AIAgent-shaped object, runs title + compression
    + resolve_provider_client as interactive would, and spies that
    build_anthropic_client is never invoked for aux.
    """
    built = {"n": 0, "urls": []}

    def _spy_build(token, base_url=None, **_k):
        built["n"] += 1
        built["urls"].append(str(base_url or ""))
        raise AssertionError(
            f"HTTP Anthropic aux client constructed (base_url={base_url!r}) "
            "— claude_cli must never use api.anthropic.com for side-LLM"
        )

    monkeypatch.setenv("HERMES_ANTHROPIC_RUNTIME", "claude_cli")
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        _spy_build,
    )

    main_runtime = {
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "api_mode": "claude_cli",
        "api_key": "sk-ant-oat01-TEST",
        "base_url": "https://api.anthropic.com",
    }

    # Title
    monkeypatch.setattr(tg, "_auto_title_enabled", lambda: True)
    title = tg.generate_title("u", "a", main_runtime=main_runtime)
    assert title is None

    # Compression
    agent = _AgentStub()
    msgs, _ = cc.compress_context(agent, [{"role": "user", "content": "x"}], None)
    assert len(msgs) == 1

    # Explicit + auto aux resolution
    c1, _ = aux.resolve_provider_client(
        "anthropic", model="claude-haiku-4-5-20251001", main_runtime=main_runtime
    )
    assert c1 is None
    monkeypatch.setattr(aux, "_try_configured_fallback_chain", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(aux, "_try_main_fallback_chain", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(aux, "_get_provider_chain", lambda: [])
    c2, _ = aux._resolve_auto(main_runtime=main_runtime)
    assert c2 is None

    assert built["n"] == 0, f"HTTP anthropic built {built['n']} times: {built['urls']}"
