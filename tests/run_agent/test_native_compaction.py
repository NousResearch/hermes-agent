"""Tests for native OpenAI Responses server-side compaction (gpt-5.6 only).

Live behavior verified 2026-08-08 against api.openai.com: gpt-5.6 and
gpt-5.3-codex accept ``context_management`` and emit compaction items;
gpt-5.1/gpt-5.2 fail server-side (HTTP 500 / stream stall) with no
structured rejection — hence the hard model-family gate these tests pin.
"""

from types import SimpleNamespace

import pytest

from agent.native_compaction import (
    DEFAULT_COMPACT_THRESHOLD,
    is_direct_openai_route,
    is_native_compaction_model,
    is_native_compaction_rejection,
    native_compaction_context_management,
    resolve_compact_threshold,
)


def _agent(
    model="gpt-5.6",
    base_url="https://api.openai.com/v1",
    enabled=True,
    compression_enabled=True,
    threshold=DEFAULT_COMPACT_THRESHOLD,
    compressor=None,
):
    return SimpleNamespace(
        model=model,
        base_url=base_url,
        codex_responses_native_compaction=enabled,
        compression_enabled=compression_enabled,
        codex_responses_compact_threshold=threshold,
        context_compressor=compressor,
    )


class TestModelGate:
    def test_gpt56_family_eligible(self):
        assert is_native_compaction_model("gpt-5.6")
        assert is_native_compaction_model("gpt-5.6-mini")
        assert is_native_compaction_model("GPT-5.6-2026-07-15")

    def test_other_models_ineligible(self):
        # gpt-5.1/5.2 fail server-side on context_management (live-verified);
        # gpt-5.3-codex works upstream but is outside the supported set.
        for model in ("gpt-5.1", "gpt-5.2", "gpt-5.3-codex", "gpt-4o", "o3", ""):
            assert not is_native_compaction_model(model)
        assert not is_native_compaction_model(None)


class TestRouteGate:
    def test_direct_openai_api(self):
        assert is_direct_openai_route("https://api.openai.com/v1")

    def test_codex_backend_flag(self):
        assert is_direct_openai_route(
            "https://chatgpt.com/backend-api/codex", is_codex_backend=True
        )

    def test_everything_else_rejected(self):
        for url in (
            "https://openrouter.ai/api/v1",
            "https://api.x.ai/v1",
            "https://models.github.ai/inference",
            "http://localhost:1234/v1",
            "https://api.openai.com.evil.com/v1",  # suffix spoof
            "",
            None,
        ):
            assert not is_direct_openai_route(url), url


class TestRequestGate:
    def test_eligible_route_gets_payload(self):
        payload = native_compaction_context_management(
            _agent(), is_codex_backend=False
        )
        assert payload == [
            {"type": "compaction", "compact_threshold": DEFAULT_COMPACT_THRESHOLD}
        ]

    def test_codex_backend_gets_payload(self):
        payload = native_compaction_context_management(
            _agent(base_url="https://chatgpt.com/backend-api/codex"),
            is_codex_backend=True,
        )
        assert payload is not None

    def test_disabled_by_default_config_value(self):
        assert (
            native_compaction_context_management(
                _agent(enabled=False), is_codex_backend=False
            )
            is None
        )

    def test_compression_disabled_disables_native(self):
        assert (
            native_compaction_context_management(
                _agent(compression_enabled=False), is_codex_backend=False
            )
            is None
        )

    def test_wrong_model_never_sends(self):
        assert (
            native_compaction_context_management(
                _agent(model="gpt-5.1"), is_codex_backend=False
            )
            is None
        )

    def test_xai_and_github_surfaces_never_send(self):
        agent = _agent()
        assert (
            native_compaction_context_management(
                agent, is_codex_backend=False, is_xai_responses=True
            )
            is None
        )
        assert (
            native_compaction_context_management(
                agent, is_codex_backend=False, is_github_responses=True
            )
            is None
        )

    def test_non_openai_route_never_sends(self):
        assert (
            native_compaction_context_management(
                _agent(base_url="https://openrouter.ai/api/v1"),
                is_codex_backend=False,
            )
            is None
        )

    def test_threshold_clamped_below_local_compressor(self):
        compressor = SimpleNamespace(threshold_tokens=100_000)
        payload = native_compaction_context_management(
            _agent(compressor=compressor), is_codex_backend=False
        )
        assert payload[0]["compact_threshold"] < 100_000


class TestThresholdClamp:
    def test_clamps_below_local_trigger(self):
        assert resolve_compact_threshold(200_000, 100_000) == 100_000 - 8_192

    def test_no_local_trigger_uses_configured(self):
        assert resolve_compact_threshold(200_000, None) == 200_000

    def test_garbage_configured_falls_back_to_default(self):
        assert resolve_compact_threshold("garbage", None) == DEFAULT_COMPACT_THRESHOLD
        assert resolve_compact_threshold(True, None) == DEFAULT_COMPACT_THRESHOLD
        assert resolve_compact_threshold(-5, None) == DEFAULT_COMPACT_THRESHOLD

    def test_tiny_local_trigger_stays_positive(self):
        assert resolve_compact_threshold(200_000, 4_000) >= 1_024


class TestRejectionMatcher:
    def test_structured_param_rejection_matches(self):
        assert is_native_compaction_rejection(
            "Error code: 400 - Unknown parameter: 'context_management'"
        )
        assert is_native_compaction_rejection(
            "invalid value for compact_threshold"
        )

    def test_generic_errors_do_not_match(self):
        # Generic failures must take the normal retry path — a transient
        # 500 or timeout must never permanently disable native compaction.
        for err in (
            "An error occurred while processing your request",
            "Broken pipe",
            "Rate limit exceeded",
            "",
            None,
        ):
            assert not is_native_compaction_rejection(err)


class TestWirePlumbing:
    """context_management flows through build_kwargs and both preflights."""

    def test_transport_build_kwargs_includes_field(self):
        from agent.transports.codex import ResponsesApiTransport

        transport = ResponsesApiTransport()
        kwargs = transport.build_kwargs(
            model="gpt-5.6",
            messages=[{"role": "user", "content": "hi"}],
            context_management=[{"type": "compaction", "compact_threshold": 4000}],
        )
        assert kwargs["context_management"] == [
            {"type": "compaction", "compact_threshold": 4000}
        ]

    def test_transport_build_kwargs_omits_field_when_none(self):
        from agent.transports.codex import ResponsesApiTransport

        transport = ResponsesApiTransport()
        kwargs = transport.build_kwargs(
            model="gpt-5.6",
            messages=[{"role": "user", "content": "hi"}],
            context_management=None,
        )
        assert "context_management" not in kwargs

    def test_preflight_preserves_field(self):
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs

        normalized = _preflight_codex_api_kwargs(
            {
                "model": "gpt-5.6",
                "instructions": "You are a test.",
                "input": [{"role": "user", "content": "hi"}],
                "store": False,
                "context_management": [
                    {"type": "compaction", "compact_threshold": 4000}
                ],
            }
        )
        assert normalized["context_management"] == [
            {"type": "compaction", "compact_threshold": 4000}
        ]

    def test_preflight_accepts_replayed_compaction_input_item(self):
        from agent.codex_responses_adapter import _preflight_codex_input_items

        items = _preflight_codex_input_items(
            [
                {"type": "compaction", "encrypted_content": "opaque-blob"},
                {"role": "user", "content": "hi"},
            ]
        )
        assert items[0] == {"type": "compaction", "encrypted_content": "opaque-blob"}

    def test_preflight_drops_empty_compaction_item(self):
        from agent.codex_responses_adapter import _preflight_codex_input_items

        items = _preflight_codex_input_items(
            [
                {"type": "compaction", "encrypted_content": ""},
                {"role": "user", "content": "hi"},
            ]
        )
        assert all(item.get("type") != "compaction" for item in items)


class TestResponseCapture:
    def test_compaction_output_item_lands_in_reasoning_sidecar(self):
        from agent.codex_responses_adapter import _normalize_codex_response

        response = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="compaction",
                    encrypted_content="blob123",
                    status="completed",
                ),
                SimpleNamespace(
                    type="message",
                    status="completed",
                    phase="final_answer",
                    content=[SimpleNamespace(type="output_text", text="OK")],
                    id="msg_1",
                ),
            ],
        )
        msg, finish_reason = _normalize_codex_response(
            response, issuer_kind="other:https://api.openai.com/v1"
        )
        assert finish_reason == "stop"
        compaction_items = [
            item
            for item in (msg.codex_reasoning_items or [])
            if item.get("type") == "compaction"
        ]
        assert len(compaction_items) == 1
        assert compaction_items[0]["encrypted_content"] == "blob123"
        assert compaction_items[0]["_issuer_kind"] == "other:https://api.openai.com/v1"

    def test_compaction_item_replayed_on_next_turn(self):
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        items = _chat_messages_to_responses_input(
            [
                {
                    "role": "assistant",
                    "content": "OK",
                    "codex_reasoning_items": [
                        {
                            "type": "compaction",
                            "encrypted_content": "blob123",
                            "_issuer_kind": "codex_backend",
                        }
                    ],
                },
                {"role": "user", "content": "next"},
            ],
            current_issuer_kind="codex_backend",
        )
        replayed = [item for item in items if item.get("type") == "compaction"]
        assert len(replayed) == 1
        assert replayed[0]["encrypted_content"] == "blob123"
        # Internal stamp must not go over the wire.
        assert "_issuer_kind" not in replayed[0]

    def test_foreign_issuer_compaction_item_dropped(self):
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        items = _chat_messages_to_responses_input(
            [
                {
                    "role": "assistant",
                    "content": "OK",
                    "codex_reasoning_items": [
                        {
                            "type": "compaction",
                            "encrypted_content": "blob123",
                            "_issuer_kind": "codex_backend",
                        }
                    ],
                },
                {"role": "user", "content": "next"},
            ],
            current_issuer_kind="xai_responses",
        )
        assert all(item.get("type") != "compaction" for item in items)


class TestAgentInitConfig:
    def test_defaults_off_and_threshold(self, monkeypatch):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            api_mode="codex_responses",
            model="gpt-5.6",
            provider="openai-api",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )
        assert agent.codex_responses_native_compaction is False
        assert agent.codex_responses_compact_threshold == 200_000

    def test_kwargs_have_no_context_management_by_default(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            api_mode="codex_responses",
            model="gpt-5.6",
            provider="openai-api",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "context_management" not in kwargs

    def test_kwargs_include_field_when_enabled_on_eligible_route(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            api_mode="codex_responses",
            model="gpt-5.6",
            provider="openai-api",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )
        agent.codex_responses_native_compaction = True
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert isinstance(kwargs.get("context_management"), list)

    def test_kwargs_omit_field_for_ineligible_model_even_when_enabled(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            api_mode="codex_responses",
            model="gpt-5.1",
            provider="openai-api",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )
        agent.codex_responses_native_compaction = True
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "context_management" not in kwargs


class TestPrunePreCheckpointItems:
    """Wire restructure around a replayed checkpoint (live-verified Aug 2026:
    the server renders nothing placed before a compaction input item)."""

    def _items(self):
        return [
            {"role": "user", "content": "goal: build the runner"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "constraint: no sleep-polling"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "out"},
            {"type": "compaction", "encrypted_content": "blobA"},
            {"role": "assistant", "content": "post-cp answer"},
            {"role": "user", "content": "next ask"},
        ]

    def test_no_checkpoint_is_identity(self):
        from agent.native_compaction import prune_pre_checkpoint_items

        items = [i for i in self._items() if i.get("type") != "compaction"]
        assert prune_pre_checkpoint_items(list(items)) == items

    def test_checkpoint_leads_and_pre_history_dropped(self):
        from agent.native_compaction import prune_pre_checkpoint_items

        out = prune_pre_checkpoint_items(self._items())
        assert out[0] == {"type": "compaction", "encrypted_content": "blobA"}
        # Pre-checkpoint assistant + tool traffic is gone (server never saw it
        # anyway); post-checkpoint tail is intact and ordered.
        assert {"role": "assistant", "content": "ack"} not in out
        assert all(i.get("call_id") != "c1" for i in out if isinstance(i, dict))
        assert out[-2:] == self._items()[-2:]

    def test_pre_checkpoint_user_messages_retained_in_order(self):
        from agent.native_compaction import prune_pre_checkpoint_items

        out = prune_pre_checkpoint_items(self._items())
        users = [i["content"] for i in out if i.get("role") == "user"]
        assert users == [
            "goal: build the runner",
            "constraint: no sleep-polling",
            "next ask",
        ]
        # Retained users sit between the checkpoint and the post tail.
        assert out.index({"role": "user", "content": "goal: build the runner"}) == 1

    def test_newest_checkpoint_run_wins(self):
        from agent.native_compaction import prune_pre_checkpoint_items

        items = [
            {"type": "compaction", "encrypted_content": "old"},
            {"role": "user", "content": "mid ask"},
            {"type": "compaction", "encrypted_content": "newA"},
            {"type": "compaction", "encrypted_content": "newB"},
            {"role": "user", "content": "tail ask"},
        ]
        out = prune_pre_checkpoint_items(items)
        blobs = [i["encrypted_content"] for i in out if i.get("type") == "compaction"]
        assert blobs == ["newA", "newB"]
        assert [i["content"] for i in out if i.get("role") == "user"] == [
            "mid ask",
            "tail ask",
        ]

    def test_retention_budget_newest_first_with_truncation(self):
        from agent.native_compaction import prune_pre_checkpoint_items

        old = {"role": "user", "content": "x" * 4000}   # ~1000 tokens
        newer = {"role": "user", "content": "y" * 2000}  # ~500 tokens
        items = [old, newer, {"type": "compaction", "encrypted_content": "b"}]
        out = prune_pre_checkpoint_items(items, retained_user_token_budget=600)
        users = [i["content"] for i in out if i.get("role") == "user"]
        # Newest kept whole; boundary (older) head-truncated to remaining budget.
        assert users[-1] == "y" * 2000
        assert users[0] == "x" * 400  # (600-500)*4 chars
        assert out[0]["type"] == "compaction"

    def test_zero_budget_keeps_only_post_tail(self):
        from agent.native_compaction import prune_pre_checkpoint_items

        out = prune_pre_checkpoint_items(self._items(), retained_user_token_budget=0)
        users = [i["content"] for i in out if i.get("role") == "user"]
        assert users == ["next ask"]

    def test_adapter_applies_prune_end_to_end(self):
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        msgs = [
            {"role": "user", "content": "the goal"},
            {
                "role": "assistant",
                "content": "ok",
                "codex_reasoning_items": [
                    {"type": "compaction", "encrypted_content": "blob"}
                ],
            },
            {"role": "user", "content": "follow-up"},
        ]
        items = _chat_messages_to_responses_input(msgs)
        assert items[0] == {"type": "compaction", "encrypted_content": "blob"}
        users = [i["content"] for i in items if i.get("role") == "user"]
        assert users == ["the goal", "follow-up"]
        # The checkpoint's own turn content is emitted AFTER the checkpoint
        # in wire order (sidecar items lead the assistant branch), so it
        # survives in the post tail — call pairing for that turn is intact.
        assert {"role": "assistant", "content": "ok"} in items
        assert items.index({"role": "assistant", "content": "ok"}) > 0

    def test_adapter_without_checkpoint_unchanged_shape(self):
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]
        items = _chat_messages_to_responses_input(msgs)
        assert [i.get("role") for i in items] == ["user", "assistant", "user"]
