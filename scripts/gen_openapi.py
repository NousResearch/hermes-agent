#!/usr/bin/env python3
"""
OpenAPI Specification Generator for Hermes REST API

Scans api_server.py route handlers and generates an OpenAPI 3.0 spec.
"""

import ast
import json
import os
import sys
from pathlib import Path


API_SERVER_PATH = Path(__file__).resolve().parent.parent / "gateway" / "platforms" / "api_server.py"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "api" / "openapi.yaml"

OPENAPI_TEMPLATE = {
    "openapi": "3.0.3",
    "info": {
        "title": "Hermes Agent REST API",
        "description": "OpenAI-compatible API for the Hermes agent. Supports chat completions, responses, runs, models, and health checks.",
        "version": "0.13.0",
        "contact": {"name": "Hermes Agent", "url": "https://github.com/NousResearch/hermes-agent"},
    },
    "servers": [{"url": "http://localhost:8642", "description": "Local development"}],
    "paths": {},
    "components": {
        "schemas": {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "type": {"type": "string"},
                            "param": {"type": "string", "nullable": True},
                            "code": {"type": "string", "nullable": True},
                        },
                    }
                },
            },
            "Model": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "object": {"type": "string"},
                    "created": {"type": "integer"},
                    "owned_by": {"type": "string"},
                },
            },
        }
    },
}

ROUTE_DOCS = {
    "_handle_health": {
        "summary": "Health check",
        "description": "Basic health check endpoint. Returns OK if the server is running.",
        "responses": {"200": {"description": "Server is healthy"}},
    },
    "_handle_health_detailed": {
        "summary": "Detailed health check",
        "description": "Returns detailed health information including gateway state and active agents.",
        "responses": {"200": {"description": "Detailed health information"}},
    },
    "_handle_models": {
        "summary": "List models",
        "description": "Returns a list of available models compatible with the OpenAI API.",
        "responses": {
            "200": {
                "description": "List of models",
                "content": {"application/json": {"schema": {"type": "object", "properties": {"data": {"type": "array", "items": {"$ref": "#/components/schemas/Model"}}}}}},
            }
        },
    },
    "_handle_chat_completions": {
        "summary": "Create chat completion",
        "description": "Creates a chat completion compatible with OpenAI's /v1/chat/completions endpoint.",
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string", "description": "Model to use"},
                            "messages": {"type": "array", "description": "Conversation messages"},
                            "stream": {"type": "boolean", "default": False},
                            "temperature": {"type": "number"},
                            "max_tokens": {"type": "integer"},
                        },
                        "required": ["model", "messages"],
                    }
                }
            },
        },
        "responses": {"200": {"description": "Chat completion response"}},
    },
    "_handle_responses": {
        "summary": "Create response",
        "description": "Creates a response compatible with OpenAI's /v1/responses endpoint.",
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string"},
                            "input": {"description": "Input to the response"},
                            "instructions": {"type": "string"},
                            "stream": {"type": "boolean", "default": False},
                        },
                    }
                }
            },
        },
        "responses": {"200": {"description": "Response object"}},
    },
    "_handle_runs": {
        "summary": "Create run",
        "description": "Creates a batch run for processing multiple requests.",
        "responses": {"200": {"description": "Run created"}},
    },
    "_handle_list_jobs": {
        "summary": "List jobs",
        "description": "Lists all batch jobs.",
        "responses": {"200": {"description": "List of jobs"}},
    },
}


def _extract_routes() -> list:
    """Extract route registrations from api_server.py using AST."""
    if not API_SERVER_PATH.exists():
        print(f"Warning: {API_SERVER_PATH} not found. Using default routes.", file=sys.stderr)
        return []

    try:
        with open(API_SERVER_PATH, encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        print(f"Error reading {API_SERVER_PATH}: {e}", file=sys.stderr)
        return []

    routes = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, "attr"):
                if node.func.attr in ("get", "post", "put", "delete"):
                    path = ""
                    method = node.func.attr.upper()
                    handler = ""
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            path = arg.value
                            break
                    for kw in node.keywords:
                        if kw.arg == "handler" and hasattr(kw.value, "attr"):
                            handler = kw.value.attr
                        elif kw.arg == "path" and isinstance(kw.value, ast.Constant):
                            path = kw.value.value
                    if handler and path:
                        routes.append({"method": method, "path": path, "handler": handler})
    except SyntaxError:
        pass

    return routes


def _build_path_item(route: dict) -> dict:
    """Build an OpenAPI path item from a route."""
    handler_name = route["handler"]
    doc = ROUTE_DOCS.get(handler_name, {})

    path_item = {
        "summary": doc.get("summary", handler_name.replace("_handle_", "").replace("_", " ").title()),
        "description": doc.get("description", ""),
        "tags": [handler_name.split("_")[2] if handler_name.count("_") >= 2 else "general"],
        "responses": doc.get("responses", {"200": {"description": "Success"}}),
    }

    if doc.get("requestBody"):
        path_item["requestBody"] = doc["requestBody"]

    return {route["method"].lower(): path_item}


def generate() -> dict:
    """Generate the complete OpenAPI spec."""
    spec = OPENAPI_TEMPLATE.copy()
    routes = _extract_routes()

    if not routes:
        routes = [
            {"method": "GET", "path": "/health", "handler": "_handle_health"},
            {"method": "GET", "path": "/v1/health", "handler": "_handle_health_detailed"},
            {"method": "GET", "path": "/v1/models", "handler": "_handle_models"},
            {"method": "POST", "path": "/v1/chat/completions", "handler": "_handle_chat_completions"},
            {"method": "POST", "path": "/v1/responses", "handler": "_handle_responses"},
            {"method": "POST", "path": "/v1/runs", "handler": "_handle_runs"},
            {"method": "GET", "path": "/v1/jobs", "handler": "_handle_list_jobs"},
        ]

    for route in routes:
        path = route["path"]
        if path not in spec["paths"]:
            spec["paths"][path] = {}
        spec["paths"][path].update(_build_path_item(route))

    return spec


def main():
    spec = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.dump(spec, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    route_count = len(spec["paths"])
    print(f"Generated OpenAPI spec at {OUTPUT_PATH} ({route_count} routes documented)")


if __name__ == "__main__":
    main()
