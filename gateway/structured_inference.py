"""Validation contract for the tool-free structured inference HTTP endpoint.

The endpoint deliberately accepts a much smaller payload than either OpenAI
API surface exposed by the gateway.  Keeping validation in a standalone
module makes the boundary independently testable and, more importantly,
prevents future chat/session fields from being accepted accidentally.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit


MAX_STRUCTURED_INFERENCE_REQUEST_BYTES = 128 * 1024
MAX_STRUCTURED_INFERENCE_PROMPT_BYTES = 64 * 1024
MAX_STRUCTURED_INFERENCE_SCHEMA_BYTES = 48 * 1024
MAX_STRUCTURED_INFERENCE_SCHEMA_DEPTH = 32
MAX_STRUCTURED_INFERENCE_SCHEMA_NODES = 4096
MAX_STRUCTURED_INFERENCE_OUTPUT_TOKENS = 8192
_RESPONSES_PROVIDERS_WITHOUT_SAMPLING_CONTROLS = frozenset({
    "openai-codex",
    "xai-oauth",
})

_ALLOWED_FIELDS = frozenset({
    "model",
    "prompt",
    "json_schema",
    "schema_name",
    "purpose",
    "max_output_tokens",
    "temperature",
})
_REQUIRED_FIELDS = frozenset({
    "model",
    "prompt",
    "json_schema",
    "schema_name",
    "purpose",
    "max_output_tokens",
})
_SCHEMA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_SUPPORTED_SCHEMA_DIALECTS = frozenset({
    "https://json-schema.org/draft/2020-12/schema",
    "https://json-schema.org/draft/2020-12/schema#",
})
_SCHEMA_MAP_KEYWORDS = frozenset({
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
})
_SCHEMA_LIST_KEYWORDS = frozenset({
    "allOf",
    "anyOf",
    "oneOf",
    "prefixItems",
})
_SCHEMA_SINGLE_KEYWORDS = frozenset({
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
})


@dataclass(frozen=True)
class StructuredInferenceRequest:
    """A fully validated request ready for ``PluginLlm``."""

    model: str
    prompt: str
    json_schema: Dict[str, Any]
    schema_name: str
    purpose: str
    max_output_tokens: int
    temperature: Optional[float]


class StructuredInferenceValidationError(ValueError):
    """Safe, client-facing request validation failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status: int = 400,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.param = param


def _encoded_size(value: str) -> int:
    return len(value.encode("utf-8"))


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"JSON number is outside the finite float range: {value}")
    return parsed


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _validate_schema_shape(schema: Dict[str, Any]) -> None:
    """Bound schema complexity and forbid unsafe schema evaluation."""

    # Count every JSON value, including values nested under schema maps. This
    # remains deliberately syntax-agnostic so unknown extension keywords
    # cannot hide an oversized payload from the complexity bound.
    stack: list[tuple[Any, int]] = [(schema, 1)]
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_STRUCTURED_INFERENCE_SCHEMA_NODES:
            raise StructuredInferenceValidationError(
                "json_schema is too complex",
                code="schema_too_complex",
                param="json_schema",
            )
        if depth > MAX_STRUCTURED_INFERENCE_SCHEMA_DEPTH:
            raise StructuredInferenceValidationError(
                "json_schema is nested too deeply",
                code="schema_too_deep",
                param="json_schema",
            )
        if isinstance(value, dict):
            stack.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value)

    # Walk actual subschemas separately. Keys in `properties`, `$defs`, and
    # similar maps are instance/schema names, not JSON Schema keywords (for
    # example, a caller may legitimately define a property named "$ref").
    schema_stack: list[Dict[str, Any]] = [schema]
    while schema_stack:
        current = schema_stack.pop()
        for ref_keyword in {"$ref", "$dynamicRef", "$recursiveRef"}:
            if ref_keyword not in current:
                continue
            reference = current[ref_keyword]
            if not isinstance(reference, str) or not reference.startswith("#"):
                raise StructuredInferenceValidationError(
                    "json_schema may only use document-local references",
                    code="remote_schema_reference",
                    param="json_schema",
                )

        # Python's regex engine has no validation timeout. Caller-controlled
        # patterns can therefore block the gateway event loop through
        # catastrophic backtracking even with bounded input sizes.
        if "pattern" in current or "patternProperties" in current:
            raise StructuredInferenceValidationError(
                "json_schema regex keywords are not supported",
                code="unsupported_schema_keyword",
                param="json_schema",
            )

        for keyword, child in current.items():
            if keyword in _SCHEMA_MAP_KEYWORDS and isinstance(child, dict):
                schema_stack.extend(
                    subschema
                    for subschema in child.values()
                    if isinstance(subschema, dict)
                )
            elif keyword in _SCHEMA_LIST_KEYWORDS and isinstance(child, list):
                schema_stack.extend(
                    subschema for subschema in child if isinstance(subschema, dict)
                )
            elif keyword in _SCHEMA_SINGLE_KEYWORDS:
                if isinstance(child, dict):
                    schema_stack.append(child)
                elif keyword == "items" and isinstance(child, list):
                    schema_stack.extend(
                        subschema for subschema in child if isinstance(subschema, dict)
                    )


def _check_json_schema(schema: Dict[str, Any]) -> None:
    """Validate the schema itself before it reaches a provider."""

    dialect = schema.get("$schema")
    if dialect is not None and (
        not isinstance(dialect, str) or dialect not in _SUPPORTED_SCHEMA_DIALECTS
    ):
        raise StructuredInferenceValidationError(
            "json_schema must use JSON Schema Draft 2020-12",
            code="unsupported_schema_dialect",
            param="json_schema",
        )

    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError as exc:  # pragma: no cover - a broken web installation
        raise RuntimeError(
            "jsonschema is required for strict structured inference"
        ) from exc

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise StructuredInferenceValidationError(
            f"Invalid json_schema: {exc.message}",
            code="invalid_json_schema",
            param="json_schema",
        ) from exc


def parse_structured_inference_request(
    raw_body: bytes,
    *,
    active_model: str,
) -> StructuredInferenceRequest:
    """Parse a strict request body and pin it to the active Hermes model."""

    if len(raw_body) > MAX_STRUCTURED_INFERENCE_REQUEST_BYTES:
        raise StructuredInferenceValidationError(
            "Request body too large",
            code="body_too_large",
            status=413,
        )
    try:
        body = json.loads(
            raw_body,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StructuredInferenceValidationError(
            "Invalid JSON in request body",
            code="invalid_json",
        ) from exc

    if not isinstance(body, dict):
        raise StructuredInferenceValidationError(
            "Request body must be a JSON object",
            code="invalid_request_body",
        )

    unsupported = sorted(set(body) - _ALLOWED_FIELDS)
    if unsupported:
        raise StructuredInferenceValidationError(
            f"Unsupported field(s): {', '.join(unsupported)}",
            code="unsupported_fields",
        )
    missing = sorted(_REQUIRED_FIELDS - set(body))
    if missing:
        raise StructuredInferenceValidationError(
            f"Missing required field(s): {', '.join(missing)}",
            code="missing_fields",
        )

    model = body["model"]
    if not isinstance(model, str) or not model.strip():
        raise StructuredInferenceValidationError(
            "model must be a non-empty string",
            code="invalid_model",
            param="model",
        )
    model = model.strip()
    if _encoded_size(model) > 256 or any(ord(char) < 32 for char in model):
        raise StructuredInferenceValidationError(
            "model is invalid or too long",
            code="invalid_model",
            param="model",
        )
    if not active_model:
        raise StructuredInferenceValidationError(
            "Hermes has no active model configured",
            code="model_not_configured",
            status=503,
            param="model",
        )
    if model != active_model:
        raise StructuredInferenceValidationError(
            f"Requested model does not match the active Hermes model ({active_model})",
            code="model_mismatch",
            status=409,
            param="model",
        )

    prompt = body["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise StructuredInferenceValidationError(
            "prompt must be a non-empty string",
            code="invalid_prompt",
            param="prompt",
        )
    if _encoded_size(prompt) > MAX_STRUCTURED_INFERENCE_PROMPT_BYTES:
        raise StructuredInferenceValidationError(
            "prompt is too large",
            code="prompt_too_large",
            status=413,
            param="prompt",
        )

    schema = body["json_schema"]
    if not isinstance(schema, dict):
        raise StructuredInferenceValidationError(
            "json_schema must be a JSON object",
            code="invalid_json_schema",
            param="json_schema",
        )
    try:
        schema_bytes = json.dumps(
            schema,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StructuredInferenceValidationError(
            "json_schema is not JSON serializable",
            code="invalid_json_schema",
            param="json_schema",
        ) from exc
    if len(schema_bytes) > MAX_STRUCTURED_INFERENCE_SCHEMA_BYTES:
        raise StructuredInferenceValidationError(
            "json_schema is too large",
            code="schema_too_large",
            status=413,
            param="json_schema",
        )
    if schema.get("type") != "object":
        raise StructuredInferenceValidationError(
            "json_schema root type must be 'object'",
            code="invalid_json_schema",
            param="json_schema",
        )
    _validate_schema_shape(schema)
    _check_json_schema(schema)

    schema_name = body["schema_name"]
    if not isinstance(schema_name, str) or not _SCHEMA_NAME_RE.fullmatch(schema_name):
        raise StructuredInferenceValidationError(
            "schema_name must match [A-Za-z_][A-Za-z0-9_.-]{0,127}",
            code="invalid_schema_name",
            param="schema_name",
        )

    purpose = body["purpose"]
    if (
        not isinstance(purpose, str)
        or not purpose.strip()
        or _encoded_size(purpose) > 128
        or any(ord(char) < 32 for char in purpose)
    ):
        raise StructuredInferenceValidationError(
            "purpose must be a non-empty string of at most 128 bytes",
            code="invalid_purpose",
            param="purpose",
        )

    max_output_tokens = body["max_output_tokens"]
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or not 1 <= max_output_tokens <= MAX_STRUCTURED_INFERENCE_OUTPUT_TOKENS
    ):
        raise StructuredInferenceValidationError(
            "max_output_tokens must be an integer from 1 to "
            f"{MAX_STRUCTURED_INFERENCE_OUTPUT_TOKENS}",
            code="invalid_max_output_tokens",
            param="max_output_tokens",
        )

    temperature = body.get("temperature")
    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or not 0.0 <= float(temperature) <= 2.0
        ):
            raise StructuredInferenceValidationError(
                "temperature must be a finite number from 0 to 2",
                code="invalid_temperature",
                param="temperature",
            )
        temperature = float(temperature)

    return StructuredInferenceRequest(
        model=model,
        prompt=prompt,
        json_schema=schema,
        schema_name=schema_name,
        purpose=purpose.strip(),
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


def validate_structured_inference_output(
    output_text: str,
    *,
    json_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Strictly parse and validate provider output at the HTTP boundary."""

    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("Structured inference output was empty")
    output = json.loads(
        output_text,
        parse_constant=_reject_nonfinite_json_constant,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_reject_duplicate_object_keys,
    )
    if not isinstance(output, dict):
        raise ValueError("Structured inference output was not a JSON object")
    from jsonschema import Draft202012Validator

    Draft202012Validator(json_schema).validate(output)
    return output


STRUCTURED_INFERENCE_BOUNDARY = "hermes-structured-no-tools-no-memory-v1"

STRUCTURED_INFERENCE_CAPABILITIES = {
    "agent_loop": False,
    "memory_access": False,
    "session_history": False,
    "tool_execution": False,
}


def structured_inference_backend_revision(
    *,
    provider: str,
    model: str,
    gateway_version: str,
    system_fingerprint: str = "",
    api_mode: str = "",
    base_url: str = "",
) -> str:
    """Return a stable, non-secret identifier for the resolved backend."""

    route_base_url = ""
    raw_base_url = (base_url or "").strip()
    if raw_base_url:
        try:
            parsed_url = urlsplit(raw_base_url)
            hostname = (parsed_url.hostname or "").lower()
            if parsed_url.scheme and hostname:
                if ":" in hostname and not hostname.startswith("["):
                    hostname = f"[{hostname}]"
                netloc = hostname
                if parsed_url.port is not None:
                    netloc = f"{netloc}:{parsed_url.port}"
                route_base_url = urlunsplit((
                    parsed_url.scheme.lower(),
                    netloc,
                    parsed_url.path.rstrip("/"),
                    "",
                    "",
                ))
        except ValueError:
            route_base_url = ""

    material = {
        "api_mode": (api_mode or "").strip().lower(),
        # Userinfo and query/fragment values can carry credentials. The
        # scheme/host/port/path still distinguish the concrete backend route
        # without binding token rotation into the revision identifier.
        "base_url": route_base_url,
        "boundary": STRUCTURED_INFERENCE_BOUNDARY,
        "gateway_version": gateway_version,
        "model": model,
        "provider": provider,
        "system_fingerprint": system_fingerprint,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    quality = structured_inference_revision_quality(
        system_fingerprint=system_fingerprint
    )
    prefix = (
        "provider-fingerprint-sha256"
        if quality == "provider_fingerprint"
        else "configuration-sha256"
    )
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def structured_inference_revision_quality(*, system_fingerprint: str) -> str:
    """Describe whether the revision includes a provider-originated signal."""

    return "provider_fingerprint" if system_fingerprint else "configuration_only"


def structured_inference_enforcement(
    *,
    provider: str,
    temperature: Optional[float],
    api_mode: str = "",
    base_url: str = "",
) -> Dict[str, str]:
    """Return honest control-enforcement metadata or reject unsupported input."""

    provider_norm = (provider or "").strip().lower()
    api_mode_norm = (api_mode or "").strip().lower()
    try:
        route_hostname = (urlsplit(base_url or "").hostname or "").lower()
    except ValueError:
        route_hostname = ""
    direct_openai_responses = (
        api_mode_norm == "codex_responses" and route_hostname == "api.openai.com"
    )
    lacks_sampling_controls = (
        provider_norm in _RESPONSES_PROVIDERS_WITHOUT_SAMPLING_CONTROLS
        or (api_mode_norm == "codex_responses" and not direct_openai_responses)
    )
    if lacks_sampling_controls and temperature is not None:
        raise StructuredInferenceValidationError(
            f"Provider {provider_norm} does not support temperature controls; "
            "omit temperature to use its provider default",
            code="temperature_not_supported",
            status=422,
            param="temperature",
        )
    return {
        "json_schema": "posthoc_strict",
        "max_output_tokens": (
            "posthoc_usage_limit"
            if lacks_sampling_controls
            else "provider_and_posthoc_usage_limit"
        ),
        "temperature": (
            "provider_default_uncontrolled"
            if temperature is None
            else "provider_requested"
        ),
    }
