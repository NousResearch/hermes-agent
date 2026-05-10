"""Unit tests for the plugin LLM facade (``agent.plugin_llm``).

These tests exercise the trust gate, JSON parsing, schema validation,
image input encoding, and the auxiliary-client invocation contract.
The auxiliary client itself is stubbed via ``make_plugin_llm_for_test``
so we don't hit real providers.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.plugin_llm import (
    PluginLlm,
    PluginLlmCompleteResult,
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
    def test_default_policy_blocks_provider_override(self):
        policy = _TrustPolicy(plugin_id="locked")
        with pytest.raises(PluginLlmTrustError, match="cannot override the provider"):
            _check_overrides(
                policy,
                requested_provider="anthropic",
                requested_model=None,
                requested_agent_id=None,
                requested_profile=None,
            )

    def test_default_policy_blocks_model_override(self):
        policy = _TrustPolicy(plugin_id="locked")
        with pytest.raises(PluginLlmTrustError, match="cannot override the model"):
            _check_overrides(
                policy,
                requested_provider=None,
                requested_model="claude-3-5-sonnet",
                requested_agent_id=None,
                requested_profile=None,
            )

    def test_default_policy_blocks_agent_override(self):
        policy = _TrustPolicy(plugin_id="locked")
        with pytest.raises(PluginLlmTrustError, match="non-default agent id"):
            _check_overrides(
                policy,
                requested_provider=None,
                requested_model=None,
                requested_agent_id="ada",
                requested_profile=None,
            )

    def test_default_policy_blocks_profile_override(self):
        policy = _TrustPolicy(plugin_id="locked")
        with pytest.raises(PluginLlmTrustError, match="cannot override the auth profile"):
            _check_overrides(
                policy,
                requested_provider=None,
                requested_model=None,
                requested_agent_id=None,
                requested_profile="work",
            )

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

    def test_provider_allowlist_rejects_non_listed(self):
        policy = _TrustPolicy(
            plugin_id="restricted",
            allow_provider_override=True,
            allowed_providers=frozenset({"openrouter", "anthropic"}),
            allow_any_provider=False,
        )
        with pytest.raises(PluginLlmTrustError, match="not in plugins.entries"):
            _check_overrides(
                policy,
                requested_provider="openai",
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

    def test_model_allowlist_rejects_non_listed(self):
        policy = _TrustPolicy(
            plugin_id="restricted",
            allow_model_override=True,
            allowed_models=frozenset({"openai/gpt-4o-mini"}),
            allow_any_model=False,
        )
        with pytest.raises(PluginLlmTrustError, match="not in plugins.entries"):
            _check_overrides(
                policy,
                requested_provider=None,
                requested_model="anthropic/claude-3-opus",
                requested_agent_id=None,
                requested_profile=None,
            )

    def test_model_allowlist_accepts_listed_case_insensitively(self):
        policy = _TrustPolicy(
            plugin_id="restricted",
            allow_model_override=True,
            allowed_models=frozenset({"openai/gpt-4o-mini"}),
            allow_any_model=False,
        )
        _, m, _, _ = _check_overrides(
            policy,
            requested_provider=None,
            requested_model="OpenAI/GPT-4o-mini",
            requested_agent_id=None,
            requested_profile=None,
        )
        assert m == "OpenAI/GPT-4o-mini"

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
    def test_missing_yields_none(self):
        ranges, allow_any = _coerce_allowlist(None)
        assert ranges is None
        assert allow_any is False

    def test_list_of_strings(self):
        ranges, allow_any = _coerce_allowlist(["A", "B"])
        assert ranges == frozenset({"a", "b"})
        assert allow_any is False

    def test_star_alone_means_any(self):
        ranges, allow_any = _coerce_allowlist(["*"])
        assert ranges == frozenset()
        assert allow_any is True

    def test_star_plus_specific_keeps_specifics(self):
        ranges, allow_any = _coerce_allowlist(["*", "openrouter"])
        assert ranges == frozenset({"openrouter"})
        assert allow_any is True

    def test_non_list_yields_none(self):
        ranges, allow_any = _coerce_allowlist("openrouter")
        assert ranges is None
        assert allow_any is False


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

    def test_json_mode_adds_system_directive(self):
        messages = _build_structured_messages(
            instructions="Summarise",
            inputs=[PluginLlmTextInput(text="content")],
            json_mode=True,
            json_schema=None,
            schema_name=None,
            system_prompt=None,
        )
        assert messages[0]["role"] == "system"
        assert "JSON object" in messages[0]["content"]

    def test_schema_name_appended_to_header(self):
        messages = _build_structured_messages(
            instructions="Extract fields",
            inputs=[PluginLlmTextInput(text="data")],
            json_mode=False,
            json_schema=None,
            schema_name="action.items",
            system_prompt=None,
        )
        header = messages[0]["content"][0]["text"]
        assert "Schema name: action.items" in header

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

    def test_dict_inputs_normalized(self):
        messages = _build_structured_messages(
            instructions="Test",
            inputs=[
                {"type": "text", "text": "hello"},
                {"type": "image", "url": "https://x.example/y.png"},
            ],
            json_mode=False,
            json_schema=None,
            schema_name=None,
            system_prompt=None,
        )
        parts = messages[0]["content"]
        assert parts[1]["text"] == "hello"
        assert parts[2]["image_url"]["url"] == "https://x.example/y.png"

    def test_invalid_input_block_rejected(self):
        with pytest.raises(ValueError, match="Unknown input block"):
            _build_structured_messages(
                instructions="Test",
                inputs=[{"type": "audio", "data": b""}],
                json_mode=False,
                json_schema=None,
                schema_name=None,
                system_prompt=None,
            )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestJsonParsing:
    def test_strip_code_fences_with_json_label(self):
        assert _strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'

    def test_strip_code_fences_without_label(self):
        assert _strip_code_fences("```\nfoo\n```") == "foo"

    def test_strip_code_fences_no_fence(self):
        assert _strip_code_fences('{"a":1}') == '{"a":1}'

    def test_parse_returns_text_when_not_json_mode(self):
        parsed, ct = _parse_structured_text(
            text='{"a": 1}', json_mode=False, json_schema=None
        )
        assert parsed is None
        assert ct == "text"

    def test_parse_valid_json_with_json_mode(self):
        parsed, ct = _parse_structured_text(
            text='{"language": "French", "is_question": true}',
            json_mode=True,
            json_schema=None,
        )
        assert parsed == {"language": "French", "is_question": True}
        assert ct == "json"

    def test_parse_strips_code_fences_before_loading(self):
        parsed, ct = _parse_structured_text(
            text='Here you go:\n```json\n{"ok": true}\n```',
            json_mode=True,
            json_schema=None,
        )
        assert parsed == {"ok": True}
        assert ct == "json"

    def test_parse_returns_text_on_invalid_json(self):
        parsed, ct = _parse_structured_text(
            text="not even close to json",
            json_mode=True,
            json_schema=None,
        )
        assert parsed is None
        assert ct == "text"

    def test_schema_validation_rejects_mismatch(self):
        pytest.importorskip("jsonschema")
        schema = {
            "type": "object",
            "properties": {"language": {"type": "string"}},
            "required": ["language"],
        }
        with pytest.raises(ValueError, match="did not match schema"):
            _parse_structured_text(
                text='{"is_question": true}',
                json_mode=False,
                json_schema=schema,
            )

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

    def test_complete_rejects_provider_override_without_trust(self):
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            sync_caller=lambda **_: ("x", "y", _fake_response("")),
        )
        with pytest.raises(PluginLlmTrustError, match="cannot override the provider"):
            llm.complete(
                [{"role": "user", "content": "hi"}],
                provider="openrouter",
            )

    def test_complete_rejects_model_override_without_trust(self):
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            sync_caller=lambda **_: ("x", "y", _fake_response("")),
        )
        with pytest.raises(PluginLlmTrustError, match="cannot override the model"):
            llm.complete(
                [{"role": "user", "content": "hi"}],
                model="anthropic/claude-3-opus",
            )

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

    def test_complete_structured_returns_text_on_unparseable_response(self):
        def fake_caller(**_kwargs):
            return "openai", "gpt-4o", _fake_response("Sorry, I can't help with that.")

        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            sync_caller=fake_caller,
        )
        result = llm.complete_structured(
            instructions="Detect language",
            input=[PluginLlmTextInput(text="x")],
            json_mode=True,
        )
        assert result.parsed is None
        assert result.content_type == "text"
        assert result.text.startswith("Sorry")

    def test_complete_structured_validates_against_schema(self):
        pytest.importorskip("jsonschema")

        def fake_caller(**_kwargs):
            return "openai", "gpt-4o", _fake_response('{"unrelated": "field"}')

        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            sync_caller=fake_caller,
        )
        schema = {
            "type": "object",
            "properties": {"language": {"type": "string"}},
            "required": ["language"],
        }
        with pytest.raises(ValueError, match="did not match schema"):
            llm.complete_structured(
                instructions="Detect language",
                input=[PluginLlmTextInput(text="x")],
                json_schema=schema,
            )

    def test_complete_structured_requires_instructions(self):
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            sync_caller=MagicMock(),
        )
        with pytest.raises(ValueError, match="non-empty instructions"):
            llm.complete_structured(
                instructions="   ",
                input=[PluginLlmTextInput(text="x")],
            )

    def test_complete_structured_requires_at_least_one_input(self):
        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            sync_caller=MagicMock(),
        )
        with pytest.raises(ValueError, match="at least one input"):
            llm.complete_structured(
                instructions="Extract",
                input=[],
            )

    def test_complete_structured_emits_response_format_extra_body(self):
        captured: dict = {}

        def fake_caller(**kwargs):
            captured.update(kwargs)
            return "openai", "gpt-4o", _fake_response('{"a": 1}')

        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            sync_caller=fake_caller,
        )
        schema = {"type": "object"}
        llm.complete_structured(
            instructions="Test",
            input=[PluginLlmTextInput(text="x")],
            json_schema=schema,
        )
        rf = captured["extra_body"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] == schema

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
    def test_acomplete_uses_async_caller(self):
        async def fake_async(**_kwargs):
            return "openai", "gpt-4o", _fake_response("async hello")

        llm = make_plugin_llm_for_test(
            plugin_id="my-plugin",
            policy=_TrustPolicy(plugin_id="my-plugin"),
            async_caller=fake_async,
        )

        async def _run() -> PluginLlmCompleteResult:
            return await llm.acomplete([{"role": "user", "content": "hi"}])

        result = asyncio.run(_run())
        assert result.text == "async hello"
        assert result.provider == "openai"

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
    def test_ctx_llm_is_lazy_singleton(self):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(name="test-plugin", source="test", key="test-plugin")
        manager = PluginManager()
        ctx = PluginContext(manifest, manager)
        first = ctx.llm
        second = ctx.llm
        assert first is second
        assert isinstance(first, PluginLlm)
        assert first._plugin_id == "test-plugin"  # type: ignore[attr-defined]

    def test_ctx_llm_uses_manifest_key_for_policy(self):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(
            name="bare-name", source="test", key="image_gen/openai"
        )
        manager = PluginManager()
        ctx = PluginContext(manifest, manager)
        assert ctx.llm._plugin_id == "image_gen/openai"  # type: ignore[attr-defined]
