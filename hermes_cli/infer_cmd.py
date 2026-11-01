"""One configured model call for scripts, using the host's existing LLM route."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path


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


def _model_route(model: str | None, provider: str | None):
    """Use the same configured-model and alias resolution as CLI one-shot mode."""
    if provider and not model:
        raise ValueError("--provider requires --model")
    from hermes_cli.config import load_config_readonly, split_model_config_default
    from hermes_cli.oneshot import _resolve_model_and_provider

    config = load_config_readonly()
    choice = _resolve_model_and_provider(config, model, provider)
    model_cfg = config.get("model") or {}
    if not choice.provider and isinstance(model_cfg, dict):
        _, nested_provider = split_model_config_default(model_cfg.get("default"))
        choice.provider = nested_provider or str(model_cfg.get("provider") or "").strip() or None
    if model and provider:
        # The one-shot helper short-circuits on an explicit provider. Resolve a
        # direct alias here too, retaining its endpoint/key origin guard.
        from hermes_cli.model_switch import resolve_startup_model_route

        route = resolve_startup_model_route(
            model, explicit_provider=provider, current_provider=provider,
            user_providers=config.get("providers"), custom_providers=config.get("custom_providers"),
            alias_config=config,
        )
        if route is not None:
            choice.model, choice.provider = route.model, route.provider
            choice.base_url, choice.api_key = route.base_url or None, route.api_key or None
    return choice


def _infer(args) -> dict:
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
        base_url=choice.base_url, api_key=choice.api_key, api_mode=choice.api_mode,
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
            print(json.dumps({"ok": False, "text": None, "parsed": None,
                              "provider": None, "model": None, "usage": None,
                              "error": {"code": "request_failed", "message": str(exc)}},
                             ensure_ascii=False))
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
