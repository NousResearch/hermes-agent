"""Hermes tool handlers for DaVinci Resolve."""

from __future__ import annotations

import json
from typing import Any, Callable

try:
    from .resolve_bridge import operations
except ImportError:
    from resolve_bridge import operations


def _json_call(fn: Callable[..., dict[str, Any]], params: dict[str, Any]) -> str:
    try:
        return json.dumps(fn(**params), sort_keys=True)
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
            sort_keys=True,
        )


def handle_probe(params, **kwargs):
    del kwargs
    return _json_call(lambda: operations.probe(), params or {})


def handle_capabilities(params, **kwargs):
    del kwargs
    return _json_call(lambda: operations.capabilities(), params or {})


def handle_launch(params, **kwargs):
    del kwargs
    return _json_call(operations.launch_resolve, params or {})


def handle_project_summary(params, **kwargs):
    del kwargs
    return _json_call(lambda: operations.project_summary(), params or {})


def handle_import_media(params, **kwargs):
    del kwargs
    return _json_call(operations.import_media, params or {})


def handle_create_timeline(params, **kwargs):
    del kwargs
    return _json_call(operations.create_timeline, params or {})


def handle_append_to_current_timeline(params, **kwargs):
    del kwargs
    return _json_call(operations.append_to_current_timeline, params or {})


def handle_timeline_marker(params, **kwargs):
    del kwargs
    return _json_call(operations.add_timeline_marker, params or {})


def handle_generate_fcpxml_timeline(params, **kwargs):
    del kwargs
    return _json_call(operations.generate_fcpxml_timeline, params or {})


def handle_generate_marker_csv(params, **kwargs):
    del kwargs
    return _json_call(operations.generate_marker_csv, params or {})


def handle_scan_media_folder(params, **kwargs):
    del kwargs
    return _json_call(operations.scan_media_folder, params or {})


def handle_create_scripted_timeline(params, **kwargs):
    del kwargs
    return _json_call(operations.create_scripted_timeline, params or {})


def handle_render_timeline(params, **kwargs):
    del kwargs
    return _json_call(operations.render_timeline, params or {})


def handle_render_status(params, **kwargs):
    del kwargs
    return _json_call(operations.render_status, params or {})
