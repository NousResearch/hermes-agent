"""Tests for native OpenAI Responses server-side compaction (gpt-5.6 only).

Live behavior verified 2026-08-08 against api.openai.com: gpt-5.6 and
gpt-5.3-codex accept ``context_management`` and emit compaction items;
gpt-5.1/gpt-5.2 fail server-side (HTTP 500 / stream stall) with no
structured rejection — hence the hard model-family gate these tests pin.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.native_compaction import (
    ASTRA_COMPACTION_METADATA_KEY,
    AstraCompactionResult,
    astra_compaction_content_digest,
    astra_compaction_prefix_with_row_ids,
    DEFAULT_COMPACT_THRESHOLD,
    is_direct_openai_route,
    is_astra_native_compaction_eligible,
    is_native_compaction_model,
    is_native_compaction_rejection,
    native_compaction_context_management,
    persist_astra_compaction_result,
    request_astra_compaction,
    resolve_compact_threshold,
    restore_astra_compaction_state,
)
from agent.transports.codex import ResponsesApiTransport


def _agent(
    model="gpt-5.6",
    base_url="https://api.openai.com/v1",
    enabled=True,
    compression_enabled=True,
    threshold: object = DEFAULT_COMPACT_THRESHOLD,
    compressor=None,
    capabilities=None,
):
    return SimpleNamespace(
        model=model,
        base_url=base_url,
        codex_responses_native_compaction=enabled,
        compression_enabled=compression_enabled,
        codex_responses_compact_threshold=threshold,
        context_compressor=compressor,
        capabilities=capabilities or {},
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

    def test_direct_astra_uses_explicit_trigger_instead_of_context_management(self):
        agent = _agent(model="gpt-6-astra")
        agent.api_mode = "codex_responses"
        agent.api_key = "fixture-key"
        agent.auth_mode = "api_key"
        agent.provider = "openai"
        agent.is_subagent = False
        agent.platform = "cli"
        agent._delegate_depth = 0
        assert is_astra_native_compaction_eligible(agent)
        assert native_compaction_context_management(agent, is_codex_backend=False) is None

    def test_direct_astra_api_provider_alias_is_eligible(self):
        agent = _agent(model="gpt-6-astra")
        agent.api_mode = "codex_responses"
        agent.api_key = "fixture-key"
        agent.auth_mode = "api_key"
        agent.provider = "openai-api"
        assert is_astra_native_compaction_eligible(agent)

    def test_explicit_trigger_request_persists_canonical_window_before_activation(self):
        sent = {}

        class Responses:
            def create(self, **kwargs):
                sent.update(kwargs)
                return SimpleNamespace(
                    status="completed",
                    output=[
                        {"type": "compaction", "encrypted_content": "opaque-checkpoint"},
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "kept"}]},
                    ],
                )

        client = Mock()
        client.responses = Responses()

        class DB:
            def __init__(self):
                self.metadata = None

            def merge_message_display_metadata(self, session_id, row_id, metadata):
                assert (session_id, row_id) == ("fixture-session", 4)
                self.metadata = metadata
                return 1

        db = DB()
        agent = SimpleNamespace(
            model="gpt-6-astra", base_url="https://api.openai.com/v1", api_mode="codex_responses",
            api_key="fixture-key", auth_mode="api_key", provider="openai", is_subagent=False,
            platform="cli", _delegate_depth=0, compression_checkpoint_required=False,
            codex_responses_native_compaction=True, compression_enabled=True,
            _codex_reasoning_replay_enabled=True, _astra_reasoning_state={}, session_id="fixture-session",
            reasoning_config={"enabled": True, "effort": "high"},
            _session_db=db,
        )
        from agent.client_lifecycle import ClientLifecycleMixin
        lifecycle = ClientLifecycleMixin()
        lifecycle.client = client
        lifecycle.provider = "openai"

        def create_request_client(*, reason, api_kwargs):
            assert reason == "astra_compaction_maintenance"
            assert api_kwargs["input"][-1] == {"type": "compaction_trigger"}
            return lifecycle._create_request_openai_client(reason=reason, api_kwargs=api_kwargs)

        agent._create_request_openai_client = create_request_client
        agent._close_request_openai_client = lambda _client, *, reason: None
        prefix = [
            {"role": "user", "content": "first", "_row_id": 2},
            {"role": "assistant", "content": "done", "_row_id": 4},
        ]
        result = request_astra_compaction(agent, prefix, system_message="fixture system", tools=[])
        assert result is not None
        assert sent["extra_body"]["input"][-1] == {"type": "compaction_trigger"}
        assert sent["reasoning"]["effort"] == "high"
        assert "tools" not in sent
        assert persist_astra_compaction_result(agent, prefix, result)
        assert db.metadata[ASTRA_COMPACTION_METADATA_KEY]["window"] == result.window
        assert agent._astra_native_compaction["covered_boundary"]["message_count"] == 2

    def test_invalid_maintenance_output_never_activates(self):
        class Responses:
            def create(self, **_kwargs):
                return SimpleNamespace(status="completed", output=[
                    {"type": "compaction", "encrypted_content": ""},
                ])

        agent = _agent(model="gpt-6-astra")
        agent.__dict__.update(
            api_mode="codex_responses", api_key="fixture-key", auth_mode="api_key", provider="openai",
            _astra_reasoning_state={}, _codex_reasoning_replay_enabled=True,
        )
        client = SimpleNamespace(responses=Responses())
        agent._create_request_openai_client = lambda *, reason, api_kwargs: client
        agent._close_request_openai_client = lambda _client, *, reason: None
        assert request_astra_compaction(agent, [{"role": "assistant", "content": "done", "_row_id": 4}]) is None
        assert not hasattr(agent, "_astra_native_compaction")

    def test_structured_trigger_rejection_disables_session_attempts(self):
        class Rejected(Exception):
            status_code = 400

            def __str__(self):
                return "Unknown parameter: compaction_trigger"

        class Responses:
            def create(self, **_kwargs):
                raise Rejected()

        agent = _agent(model="gpt-6-astra")
        agent.__dict__.update(
            api_mode="codex_responses", api_key="fixture-key", auth_mode="api_key", provider="openai",
            _astra_reasoning_state={}, _codex_reasoning_replay_enabled=True,
        )
        client = SimpleNamespace(responses=Responses())
        agent._create_request_openai_client = lambda *, reason, api_kwargs: client
        agent._close_request_openai_client = lambda _client, *, reason: None
        assert request_astra_compaction(agent, [{"role": "assistant", "content": "done", "_row_id": 4}]) is None
        assert agent._astra_native_compaction_disabled is True

    def test_usage_accounting_is_called_once_for_invisible_request(self, monkeypatch):
        calls = []

        def record(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr("agent.turn_usage.record_response_usage", record)

        class Responses:
            def create(self, **_kwargs):
                return SimpleNamespace(
                    status="completed", usage={"input_tokens": 10},
                    output=[{"type": "compaction", "encrypted_content": "opaque"}],
                )

        agent = _agent(model="gpt-6-astra")
        agent.__dict__.update(
            api_mode="codex_responses", api_key="fixture-key", auth_mode="api_key", provider="openai",
            _astra_reasoning_state={}, _codex_reasoning_replay_enabled=True, session_api_calls=0,
        )
        client = SimpleNamespace(responses=Responses())
        agent._create_request_openai_client = lambda *, reason, api_kwargs: client
        agent._close_request_openai_client = lambda _client, *, reason: None
        assert request_astra_compaction(agent, [{"role": "assistant", "content": "done", "_row_id": 4}])
        assert len(calls) == 1

    def test_persistence_requires_carrier_row_and_handles_db_error(self):
        result = AstraCompactionResult(
            window=[{"type": "compaction", "encrypted_content": "opaque"}],
            covered_boundary={"message_count": 1, "last_row_id": None},
        )

        class DB:
            def latest_message_row_id(self, *_args, **_kwargs):
                raise AssertionError("latest-row fallback must not run")

            def merge_message_display_metadata(self, *_args, **_kwargs):
                raise RuntimeError("fixture persistence failure")

        agent = SimpleNamespace(session_id="fixture-session", _session_db=DB())
        assert not persist_astra_compaction_result(
            agent, [{"role": "assistant", "content": "done"}], result,
        )
        assert not hasattr(agent, "_astra_native_compaction")
        assert not persist_astra_compaction_result(
            agent, [{"role": "assistant", "content": "done", "_row_id": 4}], result,
        )
        assert not hasattr(agent, "_astra_native_compaction")

    def test_turn_prefix_uses_ordered_durable_rows_and_rejects_mismatch(self):
        class DB:
            def get_messages_as_conversation(self, session_id, *, include_row_ids):
                assert (session_id, include_row_ids) == ("fixture-session", True)
                return [
                    {"role": "user", "content": "same", "_row_id": 11},
                    {"role": "assistant", "content": "done", "_row_id": 12},
                ]

        agent = SimpleNamespace(session_id="fixture-session", _session_db=DB())
        prefix = [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "done"},
        ]
        resolved = astra_compaction_prefix_with_row_ids(agent, prefix)
        assert [item["_row_id"] for item in resolved] == [11, 12]
        mismatch = [prefix[0], {"role": "assistant", "content": "duplicate"}]
        assert astra_compaction_prefix_with_row_ids(agent, mismatch) == []

    def test_durable_checkpoint_reloads_and_replays_only_live_tail(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(tmp_path / "state.db")
        try:
            db.create_session(session_id="astra-replay", source="cli", model="gpt-6-astra")
            user_id = db.append_message("astra-replay", "user", "old")
            assistant_id = db.append_message("astra-replay", "assistant", "done")
            history = db.get_messages_as_conversation("astra-replay", include_row_ids=True)
            window = [{"type": "compaction", "encrypted_content": "opaque"}]
            assert db.merge_message_display_metadata(
                "astra-replay", assistant_id,
                {ASTRA_COMPACTION_METADATA_KEY: {
                    "version": 1, "window": window,
                    "covered_boundary": {
                        "message_count": 2, "last_row_id": assistant_id, "last_role": "assistant",
                        "covered_digest": astra_compaction_content_digest(history),
                    },
                    "route": "direct_openai",
                }},
            ) == 1
            history = db.get_messages_as_conversation("astra-replay", include_row_ids=True)
            agent = SimpleNamespace()
            state = restore_astra_compaction_state(agent, history)
            assert state["window"] == window
            transport = ResponsesApiTransport()
            request = transport.build_kwargs(
                model="gpt-6-astra", messages=history + [{"role": "user", "content": "new"}],
                reasoning_config={"effort": "low"}, api_mode="codex_responses", api_key="fixture-key",
                auth_mode="api_key", provider="openai", base_url="https://api.openai.com/v1",
                session_id="astra-replay", astra_state={}, astra_compaction_state=state,
                astra_compaction_enabled=True,
            )
            assert request["input"] == window + [{"role": "user", "content": "new"}]
            assert len(db.get_messages("astra-replay")) == 2
            assert user_id != assistant_id
        finally:
            db.close()

    def test_checkpoint_replay_is_disabled_when_native_setting_closes(self):
        from agent.native_compaction import astra_compaction_content_digest

        history = [
            {"role": "user", "content": "old", "_row_id": 1},
            {"role": "assistant", "content": "done", "_row_id": 2},
        ]
        state = {
            "version": 1, "route": "direct_openai",
            "window": [{"type": "compaction", "encrypted_content": "opaque"}],
            "covered_boundary": {
                "message_count": 2, "last_row_id": 2, "last_role": "assistant",
                "covered_digest": astra_compaction_content_digest(history),
            },
        }
        request = ResponsesApiTransport().build_kwargs(
            model="gpt-6-astra", messages=history + [{"role": "user", "content": "new"}],
            api_mode="codex_responses", api_key="fixture-key", auth_mode="api_key",
            provider="openai", base_url="https://api.openai.com/v1", astra_state={},
            astra_compaction_state=state, astra_compaction_enabled=False,
        )
        assert all(item.get("type") != "compaction" for item in request["input"])
        assert {"role": "assistant", "content": "done"} in request["input"]

    def test_accepted_steer_invalidates_covered_prefix_replay(self):
        from agent.turn_iteration_prep import _inject_steer_into_newest_tool_result

        history = [
            {"role": "user", "content": "old", "_row_id": 1},
            {"role": "assistant", "content": "", "_row_id": 2,
             "tool_calls": [{"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result", "_row_id": 3},
            {"role": "assistant", "content": "finished", "_row_id": 4},
        ]
        state = {
            "version": 1, "route": "direct_openai",
            "window": [{"type": "compaction", "encrypted_content": "opaque"}],
            "covered_boundary": {
                "message_count": 4, "last_row_id": 4, "last_role": "assistant",
                "covered_digest": astra_compaction_content_digest(history),
            },
        }
        _inject_steer_into_newest_tool_result(SimpleNamespace(), history, "accepted steer")
        request = ResponsesApiTransport().build_kwargs(
            model="gpt-6-astra", messages=history + [{"role": "user", "content": "new"}],
            api_mode="codex_responses", api_key="fixture-key", auth_mode="api_key",
            provider="openai", base_url="https://api.openai.com/v1", astra_state={},
            astra_compaction_state=state, astra_compaction_enabled=True,
        )
        assert all(item.get("type") != "compaction" for item in request["input"])
        assert "accepted steer" in str(request["input"])

    def test_turn_start_restores_checkpoint_before_duplicate_boundary_check(self, monkeypatch):
        from agent.turn_context_compaction import run_turn_start_compaction

        prefix = [
            {"role": "user", "content": "old", "_row_id": 1},
            {"role": "assistant", "content": "done", "_row_id": 2},
        ]
        state = {
            "version": 1, "route": "direct_openai",
            "window": [{"type": "compaction", "encrypted_content": "opaque"}],
            "covered_boundary": {
                "message_count": 2, "last_row_id": 2, "last_role": "assistant",
                "covered_digest": astra_compaction_content_digest(prefix),
            },
        }
        prefix[1]["display_metadata"] = {ASTRA_COMPACTION_METADATA_KEY: state}

        class Compressor:
            protect_first_n = 0
            protect_last_n = 0
            threshold_tokens = 10
            last_real_prompt_tokens = 0

            @staticmethod
            def should_defer_preflight_to_real_usage(_tokens):
                return False

            @staticmethod
            def should_compress(_tokens):
                return False

        agent = SimpleNamespace(
            model="gpt-6-astra", base_url="https://api.openai.com/v1",
            api_mode="codex_responses", api_key="fixture-key", auth_mode="api_key",
            provider="openai", is_subagent=False, platform="cli", _delegate_depth=0,
            compression_checkpoint_required=False, codex_responses_native_compaction=True,
            compression_enabled=True, codex_responses_compact_threshold=10,
            context_compressor=Compressor(), _astra_native_compaction_disabled=False,
        )
        monkeypatch.setattr("agent.turn_context._preflight_request_tokens", lambda *_args: 100)
        monkeypatch.setattr("agent.turn_context_compaction._engine_preflight_maintenance", lambda *_args: None)
        calls = []
        monkeypatch.setattr("agent.native_compaction.request_astra_compaction", lambda *_args, **_kwargs: calls.append(1))

        run_turn_start_compaction(
            agent, messages=prefix + [{"role": "user", "content": "new"}], system_message=None,
            active_system_prompt="", conversation_history=None, current_turn_user_idx=2,
            user_message="new", effective_task_id="fixture",
        )
        assert calls == []
        assert agent._astra_native_compaction["covered_boundary"]["message_count"] == 2

    def test_codex_backend_gets_payload(self):
        payload = native_compaction_context_management(
            _agent(base_url="https://chatgpt.com/backend-api/codex"),
            is_codex_backend=True,
        )
        assert payload is not None

    def test_trusted_proxy_capability_gets_payload(self):
        payload = native_compaction_context_management(
            _agent(
                base_url="https://trusted-proxy.example/v1",
                capabilities={"openai_native_compaction": True},
            ),
            is_codex_backend=False,
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

    def test_omitted_threshold_tracks_resolved_local_trigger(self):
        compressor = SimpleNamespace(threshold_tokens=765_000)
        payload = native_compaction_context_management(
            _agent(threshold=None, compressor=compressor), is_codex_backend=False
        )
        assert payload == [{"type": "compaction", "compact_threshold": 756_808}]


class TestThresholdClamp:
    def test_omitted_threshold_derives_from_local_trigger(self):
        assert resolve_compact_threshold(None, 765_000) == 756_808

    def test_clamps_below_local_trigger(self):
        assert resolve_compact_threshold(200_000, 100_000) == 100_000 - 8_192

    def test_explicit_threshold_remains_absolute_for_larger_window(self):
        assert resolve_compact_threshold(200_000, 765_000) == 200_000

    def test_no_local_trigger_uses_configured(self):
        assert resolve_compact_threshold(200_000, None) == 200_000

    @pytest.mark.parametrize("configured", ["garbage", True, -5, 1.5])
    def test_invalid_configured_uses_automatic_local_threshold(self, configured):
        assert resolve_compact_threshold(configured, 765_000) == 756_808

    def test_automatic_without_local_trigger_uses_documented_fallback(self):
        assert resolve_compact_threshold(None, None) == DEFAULT_COMPACT_THRESHOLD

    @pytest.mark.parametrize(
        ("local_trigger", "expected"),
        [
            (8_193, 1_024),
            (8_192, 6_553),
            (4_000, 3_200),
            (1_024, 1_024),
            (1_000, 1_024),
            (1, 1_024),
        ],
    )
    def test_automatic_tiny_local_trigger_respects_provider_floor(
        self, local_trigger, expected
    ):
        assert resolve_compact_threshold(None, local_trigger) == expected

    def test_explicit_tiny_local_trigger_stays_positive(self):
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

    def test_field_echo_without_rejection_language_does_not_match(self):
        # A transient failure body that merely echoes the request (and so
        # contains the field name) must not downgrade the session (#82777).
        assert not is_native_compaction_rejection(
            "upstream timeout while processing request with "
            "context_management=[{...}]"
        )
        assert not is_native_compaction_rejection(
            "connection reset; last request included compact_threshold=200000"
        )

    def test_non_400_status_does_not_match(self):
        msg = "Unknown parameter: 'context_management'"
        assert not is_native_compaction_rejection(msg, 500)
        assert not is_native_compaction_rejection(msg, 503)
        assert not is_native_compaction_rejection(msg, 429)

    def test_400_status_with_rejection_language_matches(self):
        msg = "Unknown parameter: 'context_management'"
        assert is_native_compaction_rejection(msg, 400)

    def test_unknown_status_preserves_message_only_matching(self):
        # Transports that surface only a string keep working.
        assert is_native_compaction_rejection(
            "Error code: 400 - Unknown parameter: 'context_management'", None
        )
        assert is_native_compaction_rejection(
            "unsupported field compact_threshold", "not-a-number"
        )


class TestConfigCoercion:
    def test_false_like_strings_stay_disabled(self, monkeypatch):
        from utils import is_truthy_value

        for raw in ("false", "off", "no", "0", "", "FALSE", " Off "):
            assert not is_truthy_value(raw, False), raw

    def test_true_like_strings_enable(self):
        from utils import is_truthy_value

        for raw in ("true", "1", "yes", "on", "TRUE"):
            assert is_truthy_value(raw, False), raw


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

    def test_preflight_accepts_compaction_trigger_only_at_final_input_position(self):
        from agent.codex_responses_adapter import _preflight_codex_input_items

        items = _preflight_codex_input_items(
            [{"role": "assistant", "content": "done"}, {"type": "compaction_trigger"}]
        )
        assert items[-1] == {"type": "compaction_trigger"}
        with pytest.raises(ValueError, match="final input item"):
            _preflight_codex_input_items(
                [{"type": "compaction_trigger"}, {"role": "user", "content": "next"}]
            )

    def test_preflight_rejects_extra_compaction_trigger_fields(self):
        from agent.codex_responses_adapter import _preflight_codex_input_items

        with pytest.raises(ValueError, match="unsupported fields"):
            _preflight_codex_input_items([{"type": "compaction_trigger", "id": "x"}])


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
            native_compaction_eligible=True,
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
            native_compaction_eligible=True,
        )
        assert all(item.get("type") != "compaction" for item in items)


class TestAgentInitConfig:
    def test_defaults_off_and_automatic_threshold(self, monkeypatch):
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
        assert agent.codex_responses_compact_threshold is None

    def test_public_config_default_selects_automatic_threshold(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert (
            DEFAULT_CONFIG["compression"]["codex_responses_compact_threshold"] is None
        )

    @pytest.mark.parametrize(
        ("threshold_yaml", "configured", "resolved"),
        [
            (None, None, 756_808),
            ("null", None, 756_808),
            ("200000", 200_000, 200_000),
            ("true", None, 756_808),
            ("-5", None, 756_808),
            ("1.5", None, 756_808),
            ('"bad"', None, 756_808),
        ],
    )
    def test_loaded_config_reaches_request_threshold(
        self, tmp_path, monkeypatch, threshold_yaml, configured, resolved
    ):
        from run_agent import AIAgent

        home = tmp_path / ".hermes"
        home.mkdir()
        lines = ["compression:", "  codex_responses_native: true"]
        if threshold_yaml is not None:
            lines.append(f"  codex_responses_compact_threshold: {threshold_yaml}")
        (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))

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
        compressor = getattr(agent, "context_compressor")
        compressor.threshold_tokens = 765_000
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])

        assert getattr(agent, "codex_responses_compact_threshold") == configured
        assert kwargs["context_management"] == [
            {"type": "compaction", "compact_threshold": resolved}
        ]

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
        items = _chat_messages_to_responses_input(msgs, native_compaction_eligible=True)
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


class TestCheckpointGatedOnCurrentEligibility:
    """A captured checkpoint must not outlive the native gate.

    The checkpoint is persisted in the ``codex_reasoning_items`` sidecar, so
    it survives a mid-session model swap, ``compression.enabled: false``, the
    rejection kill switch and a resumed session. Every one of those closes the
    gate; if the wire kept being restructured around the stale checkpoint,
    pre-checkpoint history would be deleted from requests that were never
    natively compacted — on a model that cannot even decrypt the blob.
    """

    def _history(self):
        return [
            {"role": "user", "content": "goal: ship the migration"},
            {"role": "assistant", "content": "on it"},
            {"role": "user", "content": "detail A"},
            {
                "role": "assistant",
                "content": "checkpointed turn",
                "codex_reasoning_items": [
                    {
                        "type": "compaction",
                        "encrypted_content": "blob",
                        "_issuer_kind": "codex_backend",
                    }
                ],
            },
            {"role": "user", "content": "next ask"},
        ]

    def test_ineligible_request_keeps_pre_feature_wire(self):
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        history = self._history()
        items = _chat_messages_to_responses_input(
            history,
            current_issuer_kind="codex_backend",
            native_compaction_eligible=False,
        )
        pre_feature = _chat_messages_to_responses_input(
            [
                {k: v for k, v in msg.items() if k != "codex_reasoning_items"}
                for msg in history
            ],
        )
        assert items == pre_feature
        # Specifically: no checkpoint on the wire, no deleted history.
        assert all(i.get("type") != "compaction" for i in items)
        assert {"role": "assistant", "content": "on it"} in items

    def test_eligible_request_still_restructures(self):
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        items = _chat_messages_to_responses_input(
            self._history(),
            current_issuer_kind="codex_backend",
            native_compaction_eligible=True,
        )
        assert items[0]["type"] == "compaction"
        assert {"role": "assistant", "content": "on it"} not in items

    def test_converter_defaults_to_ineligible(self):
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        items = _chat_messages_to_responses_input(self._history())
        assert all(i.get("type") != "compaction" for i in items)
        assert {"role": "assistant", "content": "on it"} in items

    def test_build_kwargs_without_field_does_not_prune(self):
        """Model swapped out of the gpt-5.6 family / kill switch fired:
        the gate returns None, so the wire must be the pre-feature one."""
        from agent.transports.codex import ResponsesApiTransport

        kwargs = ResponsesApiTransport().build_kwargs(
            model="gpt-5.2",
            messages=self._history(),
            context_management=None,
        )
        assert "context_management" not in kwargs
        assert all(i.get("type") != "compaction" for i in kwargs["input"])
        assert {"role": "assistant", "content": "on it"} in kwargs["input"]

    def test_build_kwargs_with_field_prunes(self):
        from agent.transports.codex import ResponsesApiTransport

        kwargs = ResponsesApiTransport().build_kwargs(
            model="gpt-5.6",
            messages=self._history(),
            is_codex_backend=True,
            context_management=[{"type": "compaction", "compact_threshold": 4000}],
        )
        assert kwargs["input"][0]["type"] == "compaction"
        assert {"role": "assistant", "content": "on it"} not in kwargs["input"]

    def test_convert_messages_defaults_to_ineligible(self):
        from agent.transports.codex import ResponsesApiTransport

        items = ResponsesApiTransport().convert_messages(
            self._history(), is_codex_backend=True
        )
        assert all(i.get("type") != "compaction" for i in items)
        assert {"role": "assistant", "content": "on it"} in items

    def test_auxiliary_responses_adapter_never_prunes(self, monkeypatch):
        """Auxiliary calls (compression, flush_memories, MoA) replay real
        session history but never send ``context_management`` — so a
        checkpoint in that history must not restructure their request."""
        import agent.codex_responses_adapter as adapter
        from agent.auxiliary_client import _CodexCompletionsAdapter

        seen = {}
        real = adapter._chat_messages_to_responses_input

        def _spy(messages, **kw):
            seen.update(kw)
            return real(messages, **kw)

        monkeypatch.setattr(adapter, "_chat_messages_to_responses_input", _spy)

        class _Responses:
            def create(self, **kwargs):
                # #93650 routes the bulk input around the SDK transform via
                # extra_body; accept the payload in either wire shape.
                seen["input"] = kwargs.get("input") or (
                    kwargs.get("extra_body") or {}
                ).get("input")
                raise RuntimeError("stop before network")

        class _Client:
            base_url = "https://chatgpt.com/backend-api/codex"
            responses = _Responses()

        with pytest.raises(RuntimeError, match="stop before network"):
            _CodexCompletionsAdapter(_Client(), "gpt-5.2").create(
                messages=self._history()
            )

        assert seen.get("native_compaction_eligible") is False
        assert all(i.get("type") != "compaction" for i in seen["input"])
        assert {"role": "assistant", "content": "on it"} in seen["input"]
