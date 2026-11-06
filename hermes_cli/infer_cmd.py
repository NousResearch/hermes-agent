"""One script-facing inference invocation using Hermes's configured LLM route."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import SimpleNamespace


def _prompt_text(value: str | None) -> str:
    if value is None and sys.stdin.isatty():
        raise ValueError("provide a prompt or pipe one to standard input")
    prompt = sys.stdin.read() if value in (None, "-") else value
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    return prompt


def _schema_file(path: str | None) -> dict | None:
    if path is None:
        return None
    try:
        schema = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON Schema file: {exc.msg}") from exc
    if not isinstance(schema, dict):
        raise ValueError("JSON Schema file must contain an object")
    try:
        from jsonschema.exceptions import SchemaError
        from jsonschema.validators import validator_for
    except ImportError as exc:
        raise RuntimeError("--schema requires the optional jsonschema package") from exc
    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"invalid JSON Schema: {exc.message}") from exc
    return schema


@dataclass
class _InferenceRoute:
    model: str
    provider: str | None
    base_url: str | None = None
    api_key: str | None = None


def _model_route(model: str | None, provider: str | None) -> _InferenceRoute:
    """Select a configured route without depending on the agent one-shot command."""
    if provider and not model:
        raise ValueError("--provider requires --model")
    from hermes_cli.config import load_config_readonly, split_model_config_default
    from hermes_cli.model_switch import resolve_startup_model_route
    from hermes_cli.models import detect_provider_for_model

    config = load_config_readonly()
    model_cfg = config.get("model") or {}
    if isinstance(model_cfg, dict):
        configured_default = model_cfg.get("default") or model_cfg.get("model")
        configured_provider = str(model_cfg.get("provider") or "").strip()
    else:
        configured_default = model_cfg
        configured_provider = ""
    default, nested_provider = split_model_config_default(configured_default)
    configured_provider = (nested_provider or configured_provider
                           or os.getenv("HERMES_INFERENCE_PROVIDER", "").strip())
    requested_model = (model or "").strip() or os.getenv("HERMES_INFERENCE_MODEL", "").strip()
    selected_model = requested_model or default
    selected_provider = (provider or "").strip() or None
    if requested_model:
        route = resolve_startup_model_route(
            requested_model, explicit_provider=selected_provider or "",
            current_provider=selected_provider or configured_provider or "auto",
            user_providers=config.get("providers"), custom_providers=config.get("custom_providers"),
            alias_config=config,
        )
        if route is not None:
            if selected_provider and route.base_url:
                raise ValueError("--provider cannot override a URL-bound model alias")
            return _InferenceRoute(route.model, route.provider, route.base_url or None,
                                   route.api_key or None)
        if selected_provider is None:
            detected = detect_provider_for_model(requested_model, configured_provider or "auto")
            if detected:
                selected_provider, selected_model = detected
    return _InferenceRoute(selected_model, selected_provider or configured_provider or None)


def _response_json(response):
    """JSON-safe snapshot of the completion object Hermes received."""
    def default(value):
        if callable(getattr(value, "model_dump", None)):
            return value.model_dump(mode="json")
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, SimpleNamespace):
            return vars(value)
        if isinstance(value, (bytes, bytearray)):
            return {"base64": base64.b64encode(value).decode("ascii")}
        if hasattr(value, "__dict__"):
            return {key: item for key, item in vars(value).items() if not key.startswith("_")}
        return str(value)

    return json.loads(json.dumps(response, default=default, ensure_ascii=False))


def _infer(args) -> dict:
    if args.include_response and not args.json:
        raise ValueError("--include-response requires --json")
    prompt = _prompt_text(args.prompt)
    schema = _schema_file(args.schema)
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    choice = _model_route(args.model, args.provider)

    from agent.plugin_llm import (
        _build_structured_messages, _extract_text, _extract_usage,
        _json_response_format, _parse_structured_text, _resolve_attribution,
    )
    from agent.auxiliary_client import call_llm

    messages = ([{"role": "user", "content": prompt}] if schema is None else
                _build_structured_messages(
                    instructions="Respond to the input according to the JSON Schema.",
                    inputs=[{"type": "text", "text": prompt}], json_mode=True,
                    json_schema=schema, schema_name=None, system_prompt=None))
    route_info: dict[str, str] = {}
    response = call_llm(
        messages=messages, provider=choice.provider, model=choice.model or None,
        base_url=choice.base_url, api_key=choice.api_key, api_mode=None,
        max_tokens=args.max_tokens, timeout=args.timeout,
        extra_body=_json_response_format(json_mode=True, json_schema=schema) if schema else None,
        route_info=route_info,
    )
    text = _extract_text(response)
    actual_provider, actual_model = _resolve_attribution(
        provider_override=choice.provider, model_override=choice.model,
        response=response, route_info=route_info,
    )
    result = {"ok": True, "text": text, "parsed": None,
              "provider": actual_provider, "model": actual_model,
              "usage": asdict(_extract_usage(response)), "error": None}
    if args.include_response:
        result["response"] = _response_json(response)
    if schema is not None:
        try:
            parsed, content_type = _parse_structured_text(
                text=text, json_mode=True, json_schema=schema)
        except ValueError as exc:
            result["ok"] = False
            result["error"] = {"code": "schema_mismatch", "message": str(exc)}
        else:
            if content_type == "json":
                result["parsed"] = parsed
            else:
                result["ok"] = False
                result["error"] = {"code": "invalid_json", "message": "model did not return valid JSON for --schema"}
    return result


def cmd_infer(args) -> int:
    """Print one response or a JSON envelope with the raw model text on validation failure."""
    try:
        result = _infer(args)
    except Exception as exc:
        # Provider SDKs use their own exception families; preserve the traceback
        # for debugging while returning a stable error to a JSON caller.
        logging.getLogger(__name__).debug("infer request failed", exc_info=True)
        if args.json:
            failure = {"ok": False, "text": None, "parsed": None,
                       "provider": None, "model": None, "usage": None,
                       "error": {"code": "request_failed", "message": str(exc)}}
            if args.include_response:
                failure["response"] = None
            print(json.dumps(failure, ensure_ascii=False))
        else:
            print(f"hermes infer: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    elif result["ok"]:
        print(result["text"])
    else:
        print(f"hermes infer: {result['error']['message']}", file=sys.stderr)
    return 0 if result["ok"] else 1
