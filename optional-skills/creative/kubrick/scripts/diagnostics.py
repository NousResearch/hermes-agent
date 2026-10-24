#!/usr/bin/env python3
"""Stable structured diagnostics for Kubrick's unified Hermes operator surface."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, NoReturn

SCHEMA_VERSION = "1.0.0"


def diagnostic(
    *,
    status: str,
    code: str,
    exit_code: int,
    message: str,
    reason_vector: list[str] | None = None,
    context: dict[str, Any] | None = None,
    recoverable: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "code": code,
        "exit_code": exit_code,
        "message": message,
        "reason_vector": reason_vector or [code],
        "context": context or {},
        "recoverable": recoverable,
    }


def wants_json() -> bool:
    return os.environ.get("KUBRICK_DIAGNOSTICS", "").lower() in {"json", "1", "true"}


def emit(payload: dict[str, Any], *, stream: Any = None) -> None:
    target = stream or sys.stderr
    if wants_json():
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")), file=target)
    else:
        print(payload["message"], file=target)


def abort(payload: dict[str, Any], *, stream: Any = None) -> NoReturn:
    emit(payload, stream=stream)
    raise SystemExit(payload["exit_code"])
