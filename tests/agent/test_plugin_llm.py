"""Unit tests for the plugin LLM facade (``agent.plugin_llm``).

These tests exercise the trust gate, JSON parsing, schema validation,
image input encoding, and the auxiliary-client invocation contract.
The auxiliary client itself is stubbed via ``make_plugin_llm_for_test``
so we don't hit real providers.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any

import pytest

from agent.plugin_llm import (
    PluginLlmCompleteResult,
    PluginLlmEmbeddingResult,
    PluginLlmImageInput,
    PluginLlmStructuredResult,
    PluginLlmTextInput,
    PluginLlmTrustError,
    _build_structured_messages,
    _check_overrides,
    _coerce_allowlist,
    _parse_structured_text,
    _strip_code_fences,
    _TrustPolicy,
    make_plugin_llm_for_test,
)


class TestEmbeddings:
    """Embedding requests obey the facade trust and vector contracts."""

    @staticmethod
    def _response(vectors, *, indices=None):
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=v) for i, v in zip(
                range(len(vectors)) if indices is None else indices, vectors)],
            usage=SimpleNamespace(prompt_tokens=7, total_tokens=7), model="resolved-embedding",
            _hermes_input_type_handling="shared")

    def test_order_identity_usage_and_no_input_egress(self, monkeypatch, caplog):
        import agent.embedding_client as transport
        seen = []
        response = self._response([[3, 4], [1, 2]], indices=[1, 0])
        monkeypatch.setattr(transport, "call_embeddings", lambda **kw: (
            seen.append(kw) or "custom:fixture", "resolved-embedding", response))
        llm = make_plugin_llm_for_test(plugin_id="vectors", policy=_TrustPolicy(plugin_id="vectors"))
        with caplog.at_level("INFO"):
            result = llm.embed(["private first", "private second"], input_type="query", dimensions=2)
        assert isinstance(result, PluginLlmEmbeddingResult)
        assert result.vectors == [[1.0, 2.0], [3.0, 4.0]]
        assert (result.provider, result.model, result.dimensions) == ("custom:fixture", "resolved-embedding", 2)
        assert result.usage.input_tokens == result.usage.total_tokens == 7
        assert result.audit["input_type"] == seen[0]["input_type"] == "query"
        assert "private" not in repr(result) + caplog.text
        assert "api_key" not in repr(result)

    @pytest.mark.parametrize("field,value", [("provider", "openai"), ("model", "some-model"), ("profile", "other")])
    def test_overrides_fail_before_transport(self, monkeypatch, field, value):
        import agent.embedding_client as transport
        monkeypatch.setattr(transport, "call_embeddings", lambda **kw: pytest.fail("denied request dispatched"))
        llm = make_plugin_llm_for_test(plugin_id="vectors", policy=_TrustPolicy(plugin_id="vectors"))
        with pytest.raises(PluginLlmTrustError):
            llm.embed(["synthetic"], **{field: value})

    def test_allowlists_apply_to_embeddings(self, monkeypatch):
        import agent.embedding_client as transport
        monkeypatch.setattr(transport, "call_embeddings", lambda **kw: (
            kw["provider"], kw["model"], self._response([[1, 2]])))
        policy = _trusted_policy(allowed_providers=frozenset({"openai"}), allow_any_provider=False,
                                 allowed_models=frozenset({"text-embedding-3-small"}), allow_any_model=False)
        llm = make_plugin_llm_for_test(plugin_id="vectors", policy=policy)
        assert llm.embed(["synthetic"], provider="openai", model="text-embedding-3-small").dimensions == 2
        with pytest.raises(PluginLlmTrustError):
            llm.embed(["synthetic"], provider="openrouter")
        with pytest.raises(PluginLlmTrustError):
            llm.embed(["synthetic"], model="denied-model")

    @pytest.mark.parametrize("inputs,kw", [
        ([], {}), ([""], {}), ("text", {}), ([3], {}), (["x"] * 129, {}),
        (["x" * (1024 * 1024 + 1)], {}), (["x"], {"input_type": "other"}),
        (["x"], {"dimensions": True}), (["x"], {"dimensions": 0}),
        (["x"], {"dimensions": 65537}), (["x"], {"timeout": float("nan")}),
        (["x"], {"timeout": 0}), (["x"], {"timeout": 301}),
    ])
    def test_request_bounds(self, inputs, kw):
        llm = make_plugin_llm_for_test(plugin_id="vectors", policy=_TrustPolicy(plugin_id="vectors"))
        with pytest.raises(ValueError):
            llm.embed(inputs, **kw)

    @pytest.mark.parametrize("vectors,indices,dimensions", [
        ([], [], None), ([[1]], [1], None), ([[1], [2]], [0, 0], None),
        ([[1], [2, 3]], [0, 1], None), ([[float("nan")]], [0], None),
        ([[float("inf")]], [0], None), ([[True]], [0], None),
        ([["1"]], [0], None), ([[]], [0], None), ([[1, 2]], [0], 3),
    ])
    def test_invalid_vectors(self, monkeypatch, vectors, indices, dimensions):
        import agent.embedding_client as transport
        monkeypatch.setattr(transport, "call_embeddings", lambda **kw: (
            "openai", "embedding", self._response(vectors, indices=indices)))
        llm = make_plugin_llm_for_test(plugin_id="vectors", policy=_TrustPolicy(plugin_id="vectors"))
        with pytest.raises(transport.EmbeddingError) as exc:
            llm.embed(["synthetic"] * max(1, len(vectors)), dimensions=dimensions)
        assert exc.value.code == "embedding_invalid_response"

    def test_async_and_cancellation(self, monkeypatch):
        import agent.embedding_client as transport
        seen = []

        async def call(**kw):
            seen.append(kw)
            return "openai", "embedding", self._response([[1, 2]])

        monkeypatch.setattr(transport, "async_call_embeddings", call)
        llm = make_plugin_llm_for_test(plugin_id="vectors", policy=_TrustPolicy(plugin_id="vectors"))
        result = asyncio.run(llm.aembed(["synthetic"]))
        assert result.vectors == [[1, 2]]
        assert seen[0]["input_type"] == "document"

        async def cancelled(**kw):
            raise asyncio.CancelledError

        monkeypatch.setattr(transport, "async_call_embeddings", cancelled)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(llm.aembed(["synthetic"]))

    @pytest.mark.parametrize("handling", [None, "unknown", {"unexpected": "metadata"}, "x" * 4096])
    def test_audit_rejects_unknown_semantics(self, monkeypatch, handling):
        import agent.embedding_client as transport
        response = self._response([[1, 2]])
        response._hermes_input_type_handling = handling
        monkeypatch.setattr(transport, "call_embeddings", lambda **kw: ("openai", "embedding", response))
        llm = make_plugin_llm_for_test(plugin_id="vectors", policy=_TrustPolicy(plugin_id="vectors"))
        with pytest.raises(transport.EmbeddingError):
            llm.embed(["synthetic"])

    def test_embedding_slot_cannot_be_used_as_a_chat_route(self):
        from agent.auxiliary_client import get_text_auxiliary_client
        from agent.embedding_client import EmbeddingError
        with pytest.raises(EmbeddingError) as exc:
            get_text_auxiliary_client("embedding")
        assert exc.value.code == "embedding_unsupported"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(text: str, *, prompt: int = 4, completion: int = 6) -> SimpleNamespace:
    """Build an OpenAI-shaped response with the given text + token usage."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )

def _trusted_policy(plugin_id: str = "trusted-plugin", **overrides: Any) -> _TrustPolicy:
    defaults = dict(
        allow_provider_override=True,
        allowed_providers=None,
        allow_any_provider=True,
        allow_model_override=True,
        allowed_models=None,
        allow_any_model=True,
        allow_agent_id_override=True,
        allow_profile_override=True,
    )
    defaults.update(overrides)
    return _TrustPolicy(plugin_id=plugin_id, **defaults)

# ---------------------------------------------------------------------------
# Trust gate
# ---------------------------------------------------------------------------

class TestTrustGate:

    def test_overrides_independent(self):
        """Each override is gated independently — turning on
        ``allow_model_override`` does NOT also grant provider override."""
        policy = _TrustPolicy(
            plugin_id="model-only",
            allow_model_override=True,
            allow_any_model=True,
        )
        # model alone passes
        _, m, _, _ = _check_overrides(
            policy,
            requested_provider=None,
            requested_model="gpt-4o",
            requested_agent_id=None,
            requested_profile=None,
        )
        assert m == "gpt-4o"
        # provider alone is still denied
        with pytest.raises(PluginLlmTrustError, match="cannot override the provider"):
            _check_overrides(
                policy,
                requested_provider="anthropic",
                requested_model=None,
                requested_agent_id=None,
                requested_profile=None,
            )

    def test_provider_allowlist_accepts_listed_case_insensitively(self):
        policy = _TrustPolicy(
            plugin_id="restricted",
            allow_provider_override=True,
            allowed_providers=frozenset({"openrouter"}),
            allow_any_provider=False,
        )
        p, _, _, _ = _check_overrides(
            policy,
            requested_provider="OpenRouter",
            requested_model=None,
            requested_agent_id=None,
            requested_profile=None,
        )
        assert p == "OpenRouter"

    def test_no_overrides_passes_through(self):
        policy = _TrustPolicy(plugin_id="locked")
        result = _check_overrides(
            policy,
            requested_provider=None,
            requested_model=None,
            requested_agent_id=None,
            requested_profile=None,
        )
        assert result == (None, None, None, None)

    def test_all_overrides_when_fully_trusted(self):
        policy = _trusted_policy()
        result = _check_overrides(
            policy,
            requested_provider="openrouter",
            requested_model="anthropic/claude-3-5-sonnet",
            requested_agent_id="ada",
            requested_profile="work",
        )
        assert result == ("openrouter", "anthropic/claude-3-5-sonnet", "ada", "work")

class TestAllowlistCoercion:

    def test_list_of_strings(self):
        ranges, allow_any = _coerce_allowlist(["A", "B"])
        assert ranges == frozenset({"a", "b"})
        assert allow_any is False

    def test_star_alone_means_any(self):
        ranges, allow_any = _coerce_allowlist(["*"])
        assert ranges == frozenset()
        assert allow_any is True

# ---------------------------------------------------------------------------
# Structured message building
# ---------------------------------------------------------------------------

class TestStructuredMessageBuilding:
    def test_text_only_input(self):
        messages = _build_structured_messages(
            instructions="Extract the action items",
            inputs=[PluginLlmTextInput(text="meeting notes go here")],
            json_mode=False,
            json_schema=None,
            schema_name=None,
            system_prompt=None,
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        parts = messages[0]["content"]
        assert parts[0]["type"] == "text"
        assert "Extract the action items" in parts[0]["text"]
        assert parts[1] == {"type": "text", "text": "meeting notes go here"}

    def test_image_bytes_encoded_as_data_url(self):
        png_bytes = b"\x89PNG\r\n\x1a\nfake"
        messages = _build_structured_messages(
            instructions="Read the image",
            inputs=[
                PluginLlmImageInput(data=png_bytes, mime_type="image/png"),
                PluginLlmTextInput(text="prefer printed text"),
            ],
            json_mode=False,
            json_schema=None,
            schema_name=None,
            system_prompt=None,
        )
        parts = messages[0]["content"]
        assert parts[1]["type"] == "image_url"
        url = parts[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        decoded = base64.b64decode(url.split(",", 1)[1])
        assert decoded == png_bytes
        assert parts[2] == {"type": "text", "text": "prefer printed text"}

    def test_image_url_passed_through(self):
        messages = _build_structured_messages(
            instructions="Caption this",
            inputs=[PluginLlmImageInput(url="https://example.com/cat.jpg")],
            json_mode=False,
            json_schema=None,
            schema_name=None,
            system_prompt=None,
        )
        img_part = messages[0]["content"][1]
        assert img_part["type"] == "image_url"
        assert img_part["image_url"]["url"] == "https://example.com/cat.jpg"

# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

class TestJsonParsing:
    def test_strip_code_fences_with_json_label(self):
        assert _strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'

    def test_parse_valid_json_with_json_mode(self):
        parsed, ct = _parse_structured_text(
            text='{"language": "French", "is_question": true}',
            json_mode=True,
            json_schema=None,
        )
        assert parsed == {"language": "French", "is_question": True}
        assert ct == "json"

    def test_schema_validation_accepts_match(self):
        pytest.importorskip("jsonschema")
        schema = {
            "type": "object",
            "properties": {"language": {"type": "string"}},
            "required": ["language"],
        }
        parsed, ct = _parse_structured_text(
            text='{"language": "French"}',
            json_mode=False,
            json_schema=schema,
        )
        assert parsed == {"language": "French"}
        assert ct == "json"

    def test_a_reply_whose_strings_hold_code_fences_parses_as_a_whole(self):
        """A valid JSON reply is parsed as it is, not from a ``` block inside one of its strings;
        a fenced or prose-wrapped reply still parses from its fence."""
        whole = '{"code": "```hello```"}'
        assert _parse_structured_text(text=whole, json_mode=True, json_schema=None) == (
            {"code": "```hello```"}, "json")
        wrapped = 'Here it is:\n```json\n{"code": "x"}\n```'
        assert _parse_structured_text(text=wrapped, json_mode=True, json_schema=None) == ({"code": "x"}, "json")

    def test_a_fenced_block_that_parses_stays_the_answer_and_the_whole_reply_is_the_fallback(self):
        """The released order first: a fenced block that parses (and fits the schema) is the answer,
        even inside a JSON string. The whole reply is tried only when that fails; when both fail,
        the error is the fenced block's, as before."""
        assert _parse_structured_text(text='{"a": "```1```"}', json_mode=True, json_schema=None) == (1, "json")
        schema = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
        assert _parse_structured_text(text='{"code": "```1```"}', json_mode=False, json_schema=schema) == (
            {"code": "```1```"}, "json")
        with pytest.raises(ValueError, match="is not of type 'object'"):
            _parse_structured_text(text='{"other": "```1```"}', json_mode=False, json_schema=schema)
        assert _parse_structured_text(text='{"code": 1, "x": "```hello```"}', json_mode=False,
                                      json_schema=schema) == (None, "text")

    def test_a_whole_reply_fallback_that_exhausts_recursion_keeps_the_released_answer(self):
        """A recursive schema over a deeply nested reply makes validation hit the recursion limit.
        The released parser answered text here; the whole-reply fallback must not raise instead."""
        text = "[" * 300 + '"```hello```"' + "]" * 300
        schema = {"type": ["array", "string"], "items": {"$ref": "#"}}
        assert _parse_structured_text(text=text, json_mode=False, json_schema=schema) == (None, "text")

    def test_a_whole_reply_fallback_never_adds_an_error_of_its_own(self, monkeypatch):
        """The fenced block does not parse, and the whole reply cannot be used either: parsing it hits
        the recursion limit, or the plugin's schema is invalid or holds a ``$ref`` that does not
        resolve. The released parser answered text in each case; the fallback must not raise instead."""
        import json

        text = '{"a": "```hello```"}'
        loads = json.loads

        def nested_too_deep(candidate, *args, **kwargs):
            if candidate == text:
                raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")
            return loads(candidate, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(json, "loads", nested_too_deep)
            assert _parse_structured_text(text=text, json_mode=True, json_schema=None) == (None, "text")
        pytest.importorskip("jsonschema")
        for schema in ({"type": "nonsense"}, {"$ref": "#/missing"}):
            assert _parse_structured_text(text=text, json_mode=False, json_schema=schema) == (None, "text")

    def test_a_reply_that_breaks_the_schema_is_an_error(self):
        """With ``jsonschema`` installed, parsed JSON that does not match ``json_schema`` raises."""
        pytest.importorskip("jsonschema")
        schema = {"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]}
        with pytest.raises(ValueError, match="did not match schema"):
            _parse_structured_text(text='{"kind": 1}', json_mode=False, json_schema=schema)

# ---------------------------------------------------------------------------
# End-to-end facade
# ---------------------------------------------------------------------------

class TestPluginLlmFacade:
    def test_complete_uses_active_model_by_default(self):
        captured: dict = {}

        def fake_caller(**kwargs):
            captured.update(kwargs)
            return "auto", "default", _fake_response("Hello world.")

        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            sync_caller=fake_caller,
        )
        result = llm.complete([{"role": "user", "content": "hi"}])
        assert isinstance(result, PluginLlmCompleteResult)
        assert result.text == "Hello world."
        assert captured["provider_override"] is None
        assert captured["model_override"] is None
        assert captured["profile_override"] is None
        assert result.usage.input_tokens == 4
        assert result.usage.total_tokens == 10

    def test_complete_passes_through_trusted_overrides(self):
        captured: dict = {}

        def fake_caller(**kwargs):
            captured.update(kwargs)
            return "anthropic", "claude-3-opus", _fake_response("ok")

        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_trusted_policy("my-plugin"),
            sync_caller=fake_caller,
        )
        result = llm.complete(
            [{"role": "user", "content": "hi"}],
            provider="anthropic",
            model="claude-3-opus",
            profile="work",
            agent_id="ada",
            temperature=0.0,
            max_tokens=128,
            timeout=10.0,
            purpose="extract",
        )
        # The recorded provider/model in the result come from the override,
        # since the stub caller echoed those values.
        assert result.provider == "anthropic"
        assert result.model == "claude-3-opus"
        assert captured["provider_override"] == "anthropic"
        assert captured["model_override"] == "claude-3-opus"
        assert captured["profile_override"] == "work"
        assert captured["temperature"] == 0.0
        assert captured["max_tokens"] == 128
        assert captured["timeout"] == 10.0

    def test_complete_structured_returns_parsed_json(self):
        def fake_caller(**_kwargs):
            return "openai", "gpt-4o", _fake_response(
                '{"language": "French", "is_question": true, "confidence": 0.99}'
            )

        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            sync_caller=fake_caller,
        )
        result = llm.complete_structured(
            instructions="Detect language",
            input=[PluginLlmTextInput(text="Comment ça va?")],
            json_mode=True,
        )
        assert isinstance(result, PluginLlmStructuredResult)
        assert result.parsed == {
            "language": "French",
            "is_question": True,
            "confidence": 0.99,
        }
        assert result.content_type == "json"

    def test_complete_structured_with_image_passes_image_url_part(self):
        captured: dict = {}

        def fake_caller(**kwargs):
            captured.update(kwargs)
            return "openai", "gpt-4o", _fake_response('{"caption": "ok"}')

        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            sync_caller=fake_caller,
        )
        png = b"fake-bytes"
        llm.complete_structured(
            instructions="Caption this",
            input=[PluginLlmImageInput(data=png, mime_type="image/png")],
            json_mode=True,
        )
        msgs = captured["messages"]
        user_msg = next(m for m in msgs if m["role"] == "user")
        image_parts = [p for p in user_msg["content"] if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

# ---------------------------------------------------------------------------
# Async surface
# ---------------------------------------------------------------------------

class TestAsyncSurface:

    def test_acomplete_structured_parses_json(self):
        async def fake_async(**_kwargs):
            return "openai", "gpt-4o", _fake_response('{"x": 42}')

        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            async_caller=fake_async,
        )

        async def _run() -> PluginLlmStructuredResult:
            return await llm.acomplete_structured(
                instructions="Extract x",
                input=[PluginLlmTextInput(text="data")],
                json_mode=True,
            )

        result = asyncio.run(_run())
        assert result.parsed == {"x": 42}
        assert result.content_type == "json"

# ---------------------------------------------------------------------------
# Config-driven trust gate (round-trip via plugins.entries.<id>.llm)
# ---------------------------------------------------------------------------

class TestConfigDrivenPolicy:
    def test_policy_loaded_from_yaml(self, tmp_path, monkeypatch):
        from agent.plugin_llm import _resolve_trust_policy

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            """
plugins:
  entries:
    my-plugin:
      llm:
        allow_provider_override: true
        allowed_providers: [openrouter, anthropic]
        allow_model_override: true
        allowed_models:
          - openai/gpt-4o-mini
          - anthropic/claude-3-5-haiku
        allow_profile_override: false
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_cli import config as _config_mod
        _config_mod._config_cache = None  # type: ignore[attr-defined]

        policy = _resolve_trust_policy("my-plugin")
        assert policy.allow_provider_override is True
        assert policy.allow_model_override is True
        assert policy.allow_profile_override is False
        assert policy.allowed_providers == frozenset({"openrouter", "anthropic"})
        assert policy.allowed_models == frozenset({
            "openai/gpt-4o-mini", "anthropic/claude-3-5-haiku",
        })

    def test_missing_plugin_entry_yields_default_deny(self, tmp_path, monkeypatch):
        from agent.plugin_llm import _resolve_trust_policy

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("plugins: {}\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_cli import config as _config_mod
        _config_mod._config_cache = None  # type: ignore[attr-defined]

        policy = _resolve_trust_policy("never-configured")
        assert policy.allow_provider_override is False
        assert policy.allow_model_override is False
        assert policy.allow_profile_override is False
        assert policy.allow_agent_id_override is False

# ---------------------------------------------------------------------------
# Plugin context wiring
# ---------------------------------------------------------------------------

class TestPluginContextIntegration:

    def test_ctx_llm_uses_manifest_key_for_policy(self):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(
            name="bare-name", source="test", key="image_gen/openai"
        )
        manager = PluginManager()
        ctx = PluginContext(manifest, manager)
        assert ctx.llm._plugin_id == "image_gen/openai"  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Attribution (result.provider / result.model / audit log)
# ---------------------------------------------------------------------------

class TestAttribution:
    """Verifies that the result object and the audit log carry the real
    provider/model that ``call_llm`` ended up using, NOT the placeholder
    fallbacks ('auto', 'default') from earlier drafts."""

    def test_response_model_wins_over_model_override(self):
        """Providers often canonicalise the model name (e.g. ``gpt-4o``
        → ``gpt-4o-2024-08-06``). Whatever they actually returned wins
        for the recorded model so the audit log reflects reality."""
        from agent.plugin_llm import _resolve_attribution

        response = SimpleNamespace(model="gpt-4o-2024-08-06", choices=[])
        provider, model = _resolve_attribution(
            provider_override="openrouter",
            model_override="openai/gpt-4o",
            response=response,
        )
        assert model == "gpt-4o-2024-08-06"
        # Provider override is unaffected by response.model.
        assert provider == "openrouter"

    def test_response_model_used_even_when_no_overrides(self, monkeypatch):
        """The provider's canonical model name should still flow through
        when no overrides are set."""
        from agent import plugin_llm
        import agent.auxiliary_client as ac

        monkeypatch.setattr(ac, "_read_main_provider", lambda: "openrouter")
        monkeypatch.setattr(ac, "_read_main_model", lambda: "openai/gpt-4o")

        response = SimpleNamespace(model="openai/gpt-4o-2024-08-06", choices=[])
        provider, model = plugin_llm._resolve_attribution(
            provider_override=None,
            model_override=None,
            response=response,
        )
        assert provider == "openrouter"
        assert model == "openai/gpt-4o-2024-08-06"

# ---------------------------------------------------------------------------
# Hook-mode integration (ctx.llm called from a post_tool_call callback)
# ---------------------------------------------------------------------------

class TestHookMode:
    """The docs page promises ``ctx.llm`` works from inside lifecycle
    hooks. This exercises that path: register a ``post_tool_call``
    callback that calls ``ctx.llm.complete``, fire the hook through
    the real ``invoke_hook`` machinery, and check the call landed."""

    def test_complete_works_from_post_tool_call_hook(self):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(name="hook-plugin", source="test", key="hook-plugin")
        manager = PluginManager()
        ctx = PluginContext(manifest, manager)

        # Replace ctx.llm with a stub that records what the hook called.
        captured: list = []

        def fake_caller(**kwargs):
            captured.append(kwargs)
            return "openrouter", "openai/gpt-4o", _fake_response("rewrote it")

        ctx._llm = make_plugin_llm_for_test(  # type: ignore[attr-defined]
            plugin_id="hook-plugin",
            policy=_TrustPolicy(plugin_id="hook-plugin"),
            sync_caller=fake_caller,
        )

        # Plugin registers a hook that runs ctx.llm.complete on every tool call.
        def rewrite_error_hook(*, tool_name, args, result, **_):
            if "Traceback" in (result or ""):
                rewritten = ctx.llm.complete(
                    messages=[
                        {"role": "system", "content": "Rewrite errors plainly."},
                        {"role": "user", "content": result},
                    ],
                    max_tokens=64,
                    purpose="hook-plugin.rewrite",
                )
                # Real hook would return the rewritten text via
                # transform_tool_result; here we just capture for the assert.
                captured.append({"hook_returned": rewritten.text})

        ctx.register_hook("post_tool_call", rewrite_error_hook)

        # Fire the hook the same way the agent core does it.
        manager.invoke_hook(
            "post_tool_call",
            tool_name="terminal",
            args={"command": "boom"},
            result="Traceback (most recent call last):\n  RuntimeError",
        )

        # Verify ctx.llm.complete fired through the hook.
        assert len(captured) == 2  # one llm call + one hook return record
        llm_call = captured[0]
        assert "messages" in llm_call
        assert any("rewrite" in m.get("content", "").lower()
                   for m in llm_call["messages"] if isinstance(m, dict))
        hook_record = captured[1]
        assert hook_record["hook_returned"] == "rewrote it"
