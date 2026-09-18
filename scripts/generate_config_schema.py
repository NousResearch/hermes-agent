#!/usr/bin/env python3
"""Generate JSON Schema for ~/.hermes/config.yaml from DEFAULT_CONFIG.

Best-effort schema:
- infers structure and primitive types from hermes_cli.config.DEFAULT_CONFIG
- adds targeted refinements for fields whose valid shape is broader than defaults
- emits a JSON Schema draft 2020-12 document suitable for yaml-language-server
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli.config import DEFAULT_CONFIG

SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"


def infer_schema(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": ["null", "string", "number", "integer", "boolean", "object", "array"]}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        if value:
            first = infer_schema(value[0])
            return {"type": "array", "items": first}
        return {"type": "array", "items": {}}
    if isinstance(value, dict):
        props = {k: infer_schema(v) for k, v in value.items()}
        return {
            "type": "object",
            "properties": props,
            "additionalProperties": True,
        }
    return {}


def set_path(root: dict[str, Any], path: list[str], schema: dict[str, Any]) -> None:
    cur = root
    for part in path[:-1]:
        cur = cur.setdefault("properties", {}).setdefault(part, {"type": "object", "properties": {}, "additionalProperties": True})
    cur.setdefault("properties", {})[path[-1]] = schema


def main() -> None:
    schema = {
        "$schema": SCHEMA_URI,
        "$id": "https://hermes-agent.nousresearch.com/schemas/hermes-config.schema.json",
        "title": "Hermes Agent config.yaml",
        "description": "Schema for Hermes Agent configuration (~/.hermes/config.yaml)",
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }

    for key, value in DEFAULT_CONFIG.items():
        schema["properties"][key] = infer_schema(deepcopy(value))

    # Root-level refinements
    schema["properties"]["model"] = {
        "oneOf": [
            {"type": "string", "description": "Legacy shorthand model string, e.g. anthropic/claude-sonnet-4"},
            {
                "type": "object",
                "properties": {
                    "default": {"type": "string"},
                    "provider": {"type": "string"},
                    "base_url": {"type": "string"},
                    "api_key": {"type": "string"},
                    "context_length": {"type": "integer", "minimum": 1},
                    "api_mode": {"type": "string"},
                    "max_tokens": {"type": "integer", "minimum": 1},
                    "max_completion_tokens": {"type": "integer", "minimum": 1},
                    "temperature": {"type": "number"},
                    "top_p": {"type": "number"},
                },
                "additionalProperties": True,
            },
        ]
    }

    custom_provider_entry = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "base_url": {"type": "string"},
            "api_key": {"type": "string"},
            "key_env": {"type": "string"},
            "api_mode": {"type": "string"},
            "model": {"type": "string"},
            "models": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "context_length": {"type": "integer", "minimum": 1},
                        "timeout_seconds": {"type": "integer", "minimum": 1},
                        "stale_timeout_seconds": {"type": "integer", "minimum": 1},
                    },
                    "additionalProperties": True,
                },
            },
            "context_length": {"type": "integer", "minimum": 1},
            "rate_limit_delay": {"type": "number", "minimum": 0},
            "request_timeout_seconds": {"type": "integer", "minimum": 1},
            "stale_timeout_seconds": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": True,
    }
    schema["properties"]["custom_providers"] = {
        "type": "array",
        "items": custom_provider_entry,
    }

    fallback_entry = {
        "type": "object",
        "properties": {
            "provider": {"type": "string"},
            "model": {"type": "string"},
            "base_url": {"type": "string"},
            "api_key": {"type": "string"},
            "key_env": {"type": "string"},
            "api_mode": {"type": "string"},
            "reasoning_effort": {"type": "string"},
            "max_retries": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": True,
    }
    schema["properties"]["fallback_model"] = {
        "oneOf": [
            fallback_entry,
            {"type": "array", "items": fallback_entry},
        ]
    }

    schema["properties"]["platform_toolsets"] = {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    schema["properties"]["providers"] = {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "request_timeout_seconds": {"type": "integer", "minimum": 1},
                "stale_timeout_seconds": {"type": "integer", "minimum": 1},
                "models": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "timeout_seconds": {"type": "integer", "minimum": 1},
                            "stale_timeout_seconds": {"type": "integer", "minimum": 1},
                        },
                        "additionalProperties": True,
                    },
                },
            },
            "additionalProperties": True,
        },
    }

    schema["properties"]["hooks"] = {
        "type": "object",
        "description": "Shell hook registrations keyed by event name.",
        "additionalProperties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "matcher": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "object", "additionalProperties": True},
                        ]
                    },
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": True,
            },
        },
    }

    schema["properties"]["quick_commands"] = {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "command": {"type": "string"},
                "description": {"type": "string"},
                "cwd": {"type": "string"},
                "shell": {"type": "string"},
                "env": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "additionalProperties": True,
        },
    }

    set_path(schema, ["approvals", "mode"], {"type": "string", "enum": ["manual", "smart", "off"]})
    set_path(schema, ["approvals", "cron_mode"], {"type": "string", "enum": ["deny", "approve"]})
    set_path(schema, ["display", "final_response_markdown"], {"type": "string", "enum": ["render", "strip", "raw"]})
    set_path(schema, ["display", "busy_input_mode"], {"type": "string", "enum": ["interrupt", "queue", "steer"]})
    set_path(schema, ["display", "tui_status_indicator"], {"type": "string", "enum": ["kaomoji", "emoji", "unicode", "ascii"]})
    set_path(schema, ["browser", "dialog_policy"], {"type": "string", "enum": ["must_respond", "auto_dismiss", "auto_accept"]})
    set_path(schema, ["code_execution", "mode"], {"type": "string", "enum": ["project", "strict"]})
    set_path(schema, ["human_delay", "mode"], {"type": "string", "enum": ["off", "fixed", "random"]})
    set_path(schema, ["context", "engine"], {"type": "string"})
    set_path(schema, ["stt", "provider"], {"type": "string", "enum": ["local", "groq", "openai", "mistral"]})
    set_path(schema, ["tts", "provider"], {"type": "string", "enum": ["edge", "elevenlabs", "openai", "xai", "minimax", "mistral", "neutts"]})
    set_path(schema, ["agent", "image_input_mode"], {"type": "string", "enum": ["auto", "native", "text"]})
    set_path(schema, ["terminal", "backend"], {"type": "string", "enum": ["local", "docker", "ssh", "modal", "daytona", "singularity"]})
    set_path(schema, ["terminal", "modal_mode"], {"type": "string", "enum": ["auto", "sandbox", "app"]})

    # Auxiliary task configs share same shape and support provider=main
    aux_task_schema = {
        "type": "object",
        "properties": {
            "provider": {"type": "string"},
            "model": {"type": "string"},
            "base_url": {"type": "string"},
            "api_key": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1},
            "extra_body": {"type": "object", "additionalProperties": True},
            "download_timeout": {"type": "integer", "minimum": 1},
            "max_concurrency": {"type": "integer", "minimum": 1},
            "context_length": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": True,
    }
    for task in ["vision", "web_extract", "compression", "session_search", "skills_hub", "approval", "mcp", "title_generation", "flush_memories"]:
        set_path(schema, ["auxiliary", task], aux_task_schema)

    out = Path(__file__).resolve().parents[1] / "website" / "static" / "schemas" / "hermes-config.schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
