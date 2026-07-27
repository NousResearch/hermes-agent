"""Security and contract tests for the bounded structured inference endpoint."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.plugin_llm import PluginLlmStructuredResult, PluginLlmUsage
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.structured_inference import (
    MAX_STRUCTURED_INFERENCE_PROMPT_BYTES,
    MAX_STRUCTURED_INFERENCE_REQUEST_BYTES,
    STRUCTURED_INFERENCE_BOUNDARY,
    STRUCTURED_INFERENCE_CAPABILITIES,
    StructuredInferenceValidationError,
    parse_structured_inference_request,
    structured_inference_backend_revision,
    structured_inference_enforcement,
    structured_inference_revision_quality,
)


MODEL = "gpt-5.6-terra"
AUTH_HEADERS = {
    "Authorization": "Bearer strong-api-key",
    "Content-Type": "application/json",
}
RUNTIME = {
    "provider": "openai-codex",
    "requested_provider": "openai-codex",
    "base_url": "https://chatgpt.com/backend-api/codex",
    "api_key": "provider-secret",
    "api_mode": "codex_responses",
    "extra_headers": {"X-Provider-Route": "primary"},
}
SCHEMA = {
    "type": "object",
    "properties": {
        "eligible": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["eligible", "confidence"],
    "additionalProperties": False,
}


def _payload(**overrides):
    payload = {
        "model": MODEL,
        "prompt": "Classify this public disclosure.",
        "json_schema": SCHEMA,
        "schema_name": "disclosure_classification",
        "purpose": "crypto_replay.classify",
        "max_output_tokens": 512,
    }
    payload.update(overrides)
    return payload


def _adapter(*, api_key: str = "strong-api-key") -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key}))


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/v1/inference/structured",
        adapter._handle_structured_inference,
    )
    return app


def _result(
    *,
    parsed=None,
    text: str | None = None,
    system_fingerprint: str = "fp-provider-2026",
    provider: str = "openai-codex",
    model: str = MODEL,
    input_tokens: int = 120,
    output_tokens: int = 18,
    total_tokens: int = 138,
):
    output = parsed if parsed is not None else {"eligible": True, "confidence": 0.91}
    return PluginLlmStructuredResult(
        text=text if text is not None else json.dumps(output),
        parsed=output,
        content_type="json",
        provider=provider,
        model=model,
        agent_id="default",
        usage=PluginLlmUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=40,
        ),
        system_fingerprint=system_fingerprint,
    )


@pytest.fixture(autouse=True)
def _configured_backend():
    with (
        patch("agent.auxiliary_client._read_main_model", return_value=MODEL),
        patch(
            "agent.auxiliary_client._read_main_provider",
            return_value="openai-codex",
        ),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=RUNTIME,
        ),
    ):
        yield


class TestStructuredInferenceRequestValidation:
    def test_accepts_the_exact_contract(self):
        parsed = parse_structured_inference_request(
            json.dumps(_payload()).encode(),
            active_model=MODEL,
        )

        assert parsed.model == MODEL
        assert parsed.purpose == "crypto_replay.classify"
        assert parsed.temperature is None

    @pytest.mark.parametrize(
        ("mutation", "code"),
        [
            ({"messages": []}, "unsupported_fields"),
            ({"model": "different-model"}, "model_mismatch"),
            ({"purpose": ""}, "invalid_purpose"),
            ({"max_output_tokens": True}, "invalid_max_output_tokens"),
            ({"temperature": 2.1}, "invalid_temperature"),
            (
                {
                    "json_schema": {
                        "type": "object",
                        "$ref": "https://example.com/schema",
                    }
                },
                "remote_schema_reference",
            ),
            (
                {"json_schema": {"type": "object", "properties": []}},
                "invalid_json_schema",
            ),
            (
                {
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "pattern": "^[A-Z]+$",
                            }
                        },
                    }
                },
                "unsupported_schema_keyword",
            ),
            (
                {
                    "json_schema": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                    }
                },
                "unsupported_schema_dialect",
            ),
        ],
    )
    def test_rejects_unsupported_or_malformed_fields(self, mutation, code):
        with pytest.raises(StructuredInferenceValidationError) as caught:
            parse_structured_inference_request(
                json.dumps(_payload(**mutation)).encode(),
                active_model=MODEL,
            )

        assert caught.value.code == code

    def test_schema_keyword_names_are_allowed_as_instance_property_names(self):
        schema = {
            "type": "object",
            "properties": {
                "$ref": {"type": "string"},
                "pattern": {"type": "string"},
            },
            "additionalProperties": False,
        }

        parsed = parse_structured_inference_request(
            json.dumps(_payload(json_schema=schema)).encode(),
            active_model=MODEL,
        )

        assert parsed.json_schema == schema

    def test_remote_reference_in_nested_subschema_is_rejected(self):
        schema = {
            "type": "object",
            "$defs": {
                "entry": {
                    "type": "object",
                    "$ref": "https://example.com/entry.json",
                }
            },
        }

        with pytest.raises(StructuredInferenceValidationError) as caught:
            parse_structured_inference_request(
                json.dumps(_payload(json_schema=schema)).encode(),
                active_model=MODEL,
            )

        assert caught.value.code == "remote_schema_reference"

    def test_rejects_missing_required_field(self):
        payload = _payload()
        payload.pop("purpose")

        with pytest.raises(StructuredInferenceValidationError) as caught:
            parse_structured_inference_request(
                json.dumps(payload).encode(),
                active_model=MODEL,
            )

        assert caught.value.code == "missing_fields"

    def test_rejects_oversized_prompt_by_utf8_bytes(self):
        prompt = "€" * ((MAX_STRUCTURED_INFERENCE_PROMPT_BYTES // 3) + 1)
        with pytest.raises(StructuredInferenceValidationError) as caught:
            parse_structured_inference_request(
                json.dumps(_payload(prompt=prompt), ensure_ascii=False).encode(),
                active_model=MODEL,
            )

        assert caught.value.code == "prompt_too_large"
        assert caught.value.status == 413

    def test_rejects_oversized_request_before_json_parsing(self):
        with pytest.raises(StructuredInferenceValidationError) as caught:
            parse_structured_inference_request(
                b"{" + (b"x" * MAX_STRUCTURED_INFERENCE_REQUEST_BYTES),
                active_model=MODEL,
            )

        assert caught.value.code == "body_too_large"
        assert caught.value.status == 413

    def test_rejects_duplicate_json_keys(self):
        raw = (
            json
            .dumps(_payload())
            .replace(f'"model": "{MODEL}"', f'"model": "{MODEL}", "model": "{MODEL}"')
            .encode()
        )
        with pytest.raises(StructuredInferenceValidationError) as caught:
            parse_structured_inference_request(raw, active_model=MODEL)

        assert caught.value.code == "invalid_json"

    def test_rejects_float_overflow_in_schema(self):
        raw = json.dumps(_payload()).replace(
            '"type": "number"',
            '"type": "number", "maximum": 1e999',
            1,
        )

        with pytest.raises(StructuredInferenceValidationError) as caught:
            parse_structured_inference_request(raw.encode(), active_model=MODEL)

        assert caught.value.code == "invalid_json"

    def test_backend_revision_is_stable_and_backend_sensitive(self):
        kwargs = {
            "provider": "openai-codex",
            "model": MODEL,
            "gateway_version": "0.19.0",
            "system_fingerprint": "fp-1",
        }
        first = structured_inference_backend_revision(**kwargs)
        second = structured_inference_backend_revision(**kwargs)

        assert first == second
        assert first.startswith("provider-fingerprint-sha256:")
        assert first != structured_inference_backend_revision(**{
            **kwargs,
            "system_fingerprint": "fp-2",
        })
        assert first != structured_inference_backend_revision(
            **kwargs,
            api_mode="codex_responses",
        )
        assert first != structured_inference_backend_revision(**{
            **kwargs,
            "base_url": "https://different.example/v1",
        })
        with_secret = structured_inference_backend_revision(
            **kwargs,
            base_url="https://user:secret@example.com/v1?api_key=secret-one",
        )
        rotated_secret = structured_inference_backend_revision(
            **kwargs,
            base_url="https://user:changed@example.com/v1?api_key=secret-two",
        )
        assert with_secret == rotated_secret

    def test_revision_without_provider_fingerprint_is_configuration_only(self):
        revision = structured_inference_backend_revision(
            provider="openai-codex",
            model=MODEL,
            gateway_version="0.19.0",
        )

        assert revision.startswith("configuration-sha256:")
        assert (
            structured_inference_revision_quality(system_fingerprint="")
            == "configuration_only"
        )

    def test_codex_enforcement_rejects_unsupported_temperature(self):
        with pytest.raises(StructuredInferenceValidationError) as caught:
            structured_inference_enforcement(
                provider="openai-codex",
                temperature=0.0,
            )

        assert caught.value.code == "temperature_not_supported"
        assert caught.value.status == 422

    def test_responses_api_mode_rejects_temperature_for_custom_route(self):
        with pytest.raises(StructuredInferenceValidationError) as caught:
            structured_inference_enforcement(
                provider="custom",
                api_mode="codex_responses",
                temperature=0.0,
            )

        assert caught.value.code == "temperature_not_supported"

    def test_direct_openai_responses_reports_upstream_controls(self):
        enforcement = structured_inference_enforcement(
            provider="custom",
            api_mode="codex_responses",
            base_url="https://api.openai.com/v1",
            temperature=0.25,
        )

        assert enforcement == {
            "json_schema": "posthoc_strict",
            "max_output_tokens": "provider_and_posthoc_usage_limit",
            "temperature": "provider_requested",
        }


class TestStructuredInferenceEndpoint:
    @pytest.mark.asyncio
    async def test_requires_normal_gateway_bearer_auth(self):
        adapter = _adapter()
        completion = AsyncMock(return_value=_result())
        with patch(
            "agent.plugin_llm.PluginLlm.acomplete_structured",
            completion,
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                )

        assert response.status == 401
        completion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_direct_app_wiring_without_api_key_fails_closed(self):
        adapter = _adapter(api_key="")
        completion = AsyncMock(return_value=_result())
        with patch(
            "agent.plugin_llm.PluginLlm.acomplete_structured",
            completion,
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                )
                body = await response.json()

        assert response.status == 503
        assert body["error"]["code"] == "gateway_auth_not_configured"
        completion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_only_plugin_llm_and_never_agent_or_session_surfaces(self):
        adapter = _adapter()
        adapter._run_agent = AsyncMock(side_effect=AssertionError("_run_agent called"))
        adapter._create_agent = MagicMock(side_effect=AssertionError("AIAgent created"))
        adapter._ensure_session_db = MagicMock(
            side_effect=AssertionError("session DB loaded")
        )
        adapter._ensure_session_db_async = AsyncMock(
            side_effect=AssertionError("session DB loaded")
        )
        adapter._bind_api_server_session = MagicMock(
            side_effect=AssertionError("session context bound")
        )
        completion = AsyncMock(return_value=_result())

        with (
            patch(
                "agent.plugin_llm.PluginLlm.acomplete_structured",
                completion,
            ),
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value=MODEL,
            ),
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )
                body = await response.json()

        assert response.status == 200
        assert body["boundary"] == STRUCTURED_INFERENCE_BOUNDARY
        assert body["capabilities"] == STRUCTURED_INFERENCE_CAPABILITIES
        assert body["output"] == {"eligible": True, "confidence": 0.91}
        assert body["provider"] == "openai-codex"
        assert body["model"] == MODEL
        assert body["system_fingerprint"] == "fp-provider-2026"
        assert body["backend_revision"].startswith("provider-fingerprint-sha256:")
        assert body["revision_quality"] == "provider_fingerprint"
        assert body["enforcement"] == {
            "json_schema": "posthoc_strict",
            "max_output_tokens": "posthoc_usage_limit",
            "temperature": "provider_default_uncontrolled",
        }
        assert body["usage"]["total_tokens"] == 138

        adapter._run_agent.assert_not_awaited()
        adapter._create_agent.assert_not_called()
        adapter._ensure_session_db.assert_not_called()
        adapter._ensure_session_db_async.assert_not_awaited()
        adapter._bind_api_server_session.assert_not_called()

        completion.assert_awaited_once()
        call = completion.await_args
        assert call.kwargs["purpose"] == "crypto_replay.classify"
        assert call.kwargs["input"][0].text == "Classify this public disclosure."
        # Purpose is audit metadata only; it is not injected into model input.
        assert "crypto_replay.classify" not in call.kwargs["instructions"]
        assert call.kwargs.get("system_prompt") is None
        assert call.kwargs.get("agent_id") is None
        assert call.kwargs.get("profile") is None
        assert call.kwargs["provider"] == "openai-codex"
        assert call.kwargs["model"] == MODEL
        assert call.kwargs["temperature"] is None
        assert call.kwargs["max_tokens"] == 512
        assert call.kwargs["strict_json"] is True
        assert call.kwargs["strict_route"] is True
        assert adapter._inflight_structured_inference == 0

    @pytest.mark.asyncio
    async def test_shares_gateway_concurrency_limit_and_work_accounting(self):
        adapter = _adapter()
        adapter._max_concurrent_runs = 1
        adapter._inflight_agent_runs = 1
        completion = AsyncMock(return_value=_result())
        with patch(
            "agent.plugin_llm.PluginLlm.acomplete_structured",
            completion,
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                limited = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )
                limited_body = await limited.json()

        assert limited.status == 429
        assert limited_body["error"]["code"] == "rate_limit_exceeded"
        completion.assert_not_awaited()
        assert adapter._inflight_structured_inference == 0

        adapter._inflight_agent_runs = 0

        async def assert_accounted(**_kwargs):
            assert adapter.active_agent_work_count() == 1
            return _result()

        with patch(
            "agent.plugin_llm.PluginLlm.acomplete_structured",
            side_effect=assert_accounted,
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                accepted = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )

        assert accepted.status == 200
        assert adapter.active_agent_work_count() == 0

    @pytest.mark.asyncio
    async def test_codex_temperature_is_rejected_instead_of_silently_ignored(self):
        adapter = _adapter()
        completion = AsyncMock(return_value=_result())
        with patch(
            "agent.plugin_llm.PluginLlm.acomplete_structured",
            completion,
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(temperature=0),
                    headers=AUTH_HEADERS,
                )
                body = await response.json()

        assert response.status == 422
        assert body["error"]["code"] == "temperature_not_supported"
        completion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unavailable_active_route_fails_before_provider_call(self):
        adapter = _adapter()
        completion = AsyncMock(return_value=_result())
        with (
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=RuntimeError("credential resolution failed"),
            ),
            patch(
                "agent.plugin_llm.PluginLlm.acomplete_structured",
                completion,
            ),
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )
                body = await response.json()

        assert response.status == 503
        assert body["error"]["code"] == "provider_route_unavailable"
        completion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_purpose_cannot_select_auxiliary_task_configuration(self):
        adapter = _adapter()
        provider_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"eligible": True, "confidence": 0.91})
                    )
                )
            ],
            model=MODEL,
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=18,
                total_tokens=138,
            ),
            system_fingerprint="fp-provider-2026",
        )
        create = AsyncMock(return_value=provider_response)
        client = SimpleNamespace(
            base_url="https://chatgpt.com/backend-api/codex",
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )
        observed_tasks = []
        client_lookup = MagicMock(return_value=(client, MODEL))

        def resolve_task(task, provider, model, base_url, api_key):
            observed_tasks.append(task)
            if task == "crypto_replay.classify":
                return (
                    provider,
                    model,
                    "https://alternate.invalid/v1",
                    "alternate-key",
                    None,
                )
            return provider, model, base_url, api_key, None

        def task_extra_body(task):
            if task == "crypto_replay.classify":
                return {"malicious_aux_setting": True}
            return {}

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                side_effect=resolve_task,
            ),
            patch(
                "agent.auxiliary_client._get_task_extra_body",
                side_effect=task_extra_body,
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                client_lookup,
            ),
            patch("agent.aux_accounting.record_aux_usage"),
        ):
            async with TestClient(TestServer(_app(adapter))) as http:
                response = await http.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )

        assert response.status == 200
        assert observed_tasks == [None]
        provider_kwargs = create.await_args.kwargs
        assert "crypto_replay.classify" not in repr(provider_kwargs["messages"])
        assert "malicious_aux_setting" not in provider_kwargs.get(
            "extra_body",
            {},
        )
        assert provider_kwargs["extra_headers"] == {
            "X-Provider-Route": "primary",
        }
        lookup_kwargs = client_lookup.call_args.kwargs
        assert lookup_kwargs["base_url"] == RUNTIME["base_url"]
        assert lookup_kwargs["api_key"] == RUNTIME["api_key"]
        assert lookup_kwargs["api_mode"] == "codex_responses"
        assert lookup_kwargs["main_runtime"]["provider"] == "openai-codex"
        assert lookup_kwargs["main_runtime"]["model"] == MODEL

    @pytest.mark.asyncio
    async def test_unknown_field_and_oversized_body_never_reach_provider(self):
        adapter = _adapter()
        completion = AsyncMock(return_value=_result())
        with (
            patch(
                "agent.plugin_llm.PluginLlm.acomplete_structured",
                completion,
            ),
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value=MODEL,
            ),
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                unknown = await client.post(
                    "/v1/inference/structured",
                    json=_payload(session_id="forbidden"),
                    headers=AUTH_HEADERS,
                )
                unknown_body = await unknown.json()
                oversized = await client.post(
                    "/v1/inference/structured",
                    data=b"x" * (MAX_STRUCTURED_INFERENCE_REQUEST_BYTES + 1),
                    headers=AUTH_HEADERS,
                )
                malformed = await client.post(
                    "/v1/inference/structured",
                    data=b"{",
                    headers=AUTH_HEADERS,
                )
                malformed_body = await malformed.json()

        assert unknown.status == 400
        assert unknown_body["error"]["code"] == "unsupported_fields"
        assert oversized.status == 413
        assert malformed.status == 400
        assert malformed_body["error"]["code"] == "invalid_json"
        completion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_schema_invalid_provider_output_fails_closed(self):
        adapter = _adapter()
        completion = AsyncMock(
            return_value=_result(parsed={"eligible": "yes", "confidence": 0.9})
        )
        with (
            patch(
                "agent.plugin_llm.PluginLlm.acomplete_structured",
                completion,
            ),
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value=MODEL,
            ),
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )
                body = await response.json()

        assert response.status == 502
        assert body["error"]["code"] == "invalid_structured_output"

    @pytest.mark.asyncio
    async def test_missing_usage_fails_closed(self):
        adapter = _adapter()
        completion = AsyncMock(
            return_value=_result(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )
        )
        with patch(
            "agent.plugin_llm.PluginLlm.acomplete_structured",
            completion,
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )
                body = await response.json()

        assert response.status == 502
        assert body["error"]["code"] == "usage_unavailable"

    @pytest.mark.asyncio
    async def test_posthoc_output_token_limit_fails_closed(self):
        adapter = _adapter()
        completion = AsyncMock(
            return_value=_result(
                output_tokens=513,
                total_tokens=633,
            )
        )
        with patch(
            "agent.plugin_llm.PluginLlm.acomplete_structured",
            completion,
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )
                body = await response.json()

        assert response.status == 502
        assert body["error"]["code"] == "output_token_limit_exceeded"

    @pytest.mark.asyncio
    async def test_backend_identity_drift_fails_closed(self):
        adapter = _adapter()
        completion = AsyncMock(return_value=_result(model="different-model"))
        with patch(
            "agent.plugin_llm.PluginLlm.acomplete_structured",
            completion,
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )
                body = await response.json()

        assert response.status == 502
        assert body["error"]["code"] == "backend_identity_mismatch"

    @pytest.mark.asyncio
    async def test_missing_provider_fingerprint_is_labeled_configuration_only(self):
        adapter = _adapter()
        completion = AsyncMock(return_value=_result(system_fingerprint=""))
        with patch(
            "agent.plugin_llm.PluginLlm.acomplete_structured",
            completion,
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )
                body = await response.json()

        assert response.status == 200
        assert "system_fingerprint" not in body
        assert body["revision_quality"] == "configuration_only"
        assert body["backend_revision"].startswith("configuration-sha256:")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider_text",
        [
            '{"eligible": true, "eligible": false, "confidence": 0.9}',
            '{"eligible": true, "confidence": NaN}',
            '{"eligible": true, "confidence": 1e999}',
            '```json\n{"eligible": true, "confidence": 0.9}\n```',
        ],
    )
    async def test_non_strict_provider_json_fails_closed(self, provider_text):
        adapter = _adapter()
        completion = AsyncMock(return_value=_result(text=provider_text))
        with (
            patch(
                "agent.plugin_llm.PluginLlm.acomplete_structured",
                completion,
            ),
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value=MODEL,
            ),
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/v1/inference/structured",
                    json=_payload(),
                    headers=AUTH_HEADERS,
                )
                body = await response.json()

        assert response.status == 502
        assert body["error"]["code"] == "invalid_structured_output"
