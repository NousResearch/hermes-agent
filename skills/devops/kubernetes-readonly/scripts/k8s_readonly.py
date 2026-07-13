#!/usr/bin/env python3
"""Read-only kubectl helper: validates JSON requests, runs allowlisted kubectl argv.

Prints a single JSON object to stdout (machine-readable for agents).

Stdout/stderr are streamed with a hard byte cap so oversized kubectl output
cannot inflate process memory before truncation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from typing import Any, IO

from pydantic import TypeAdapter

from k8s_models import (
    K8sRequest,
    OpApiResources,
    OpClusterInfo,
    OpDescribe,
    OpExplain,
    OpGet,
    OpTopNodes,
    OpTopPods,
    OpVersion,
)

_ADAPTER = TypeAdapter(K8sRequest)
_MAX_CAPTURE = 2_000_000
_DEFAULT_TIMEOUT_SEC = 120
_READ_CHUNK = 65_536


def _kubectl_bin() -> str | None:
    return shutil.which("kubectl")


def _argv_for(req: K8sRequest) -> list[str]:
    k = "kubectl"
    if isinstance(req, OpVersion):
        return [k, "version", "-o", "json"]
    if isinstance(req, OpClusterInfo):
        return [k, "cluster-info"]
    if isinstance(req, OpApiResources):
        cmd = [k, "api-resources", "-o", "wide"]
        if req.api_group:
            cmd += ["--api-group", req.api_group]
        return cmd
    if isinstance(req, OpExplain):
        cmd = [k, "explain", req.resource]
        if req.recursive:
            cmd.append("--recursive")
        return cmd
    if isinstance(req, OpGet):
        cmd = [k, "get", req.resource]
        if req.name:
            cmd.append(req.name)
        if req.all_namespaces:
            cmd.append("-A")
        elif req.namespace:
            cmd.extend(["-n", req.namespace])
        if req.output != "default":
            cmd.extend(["-o", req.output])
        return cmd
    if isinstance(req, OpDescribe):
        cmd = [k, "describe", req.resource, req.name]
        if req.namespace:
            cmd.extend(["-n", req.namespace])
        return cmd
    if isinstance(req, OpTopPods):
        cmd = [k, "top", "pods"]
        if req.all_namespaces:
            cmd.append("-A")
        elif req.namespace:
            cmd.extend(["-n", req.namespace])
        return cmd
    if isinstance(req, OpTopNodes):
        return [k, "top", "nodes"]
    raise TypeError("unhandled request type")


def _read_stream_bounded(stream: IO[bytes], limit: int, sink: list[Any]) -> None:
    """Read ``stream`` into ``sink`` as (text, truncated), never exceeding ``limit`` bytes retained."""
    buf = bytearray()
    truncated = False
    try:
        while True:
            chunk = stream.read(_READ_CHUNK)
            if not chunk:
                break
            if truncated:
                continue
            remaining = limit - len(buf)
            if remaining <= 0:
                truncated = True
                continue
            if len(chunk) > remaining:
                buf.extend(chunk[:remaining])
                truncated = True
                continue
            buf.extend(chunk)
    finally:
        try:
            stream.close()
        except Exception:
            pass
    sink.append((buf.decode("utf-8", errors="replace"), truncated))


def _capture_bounded(
    proc: subprocess.Popen[bytes],
    *,
    limit: int,
    timeout: float,
) -> tuple[str, str, bool]:
    """Join a process while retaining at most ``limit`` bytes per stream."""
    out_holder: list[Any] = []
    err_holder: list[Any] = []
    assert proc.stdout is not None and proc.stderr is not None
    t_out = threading.Thread(
        target=_read_stream_bounded, args=(proc.stdout, limit, out_holder), daemon=True
    )
    t_err = threading.Thread(
        target=_read_stream_bounded, args=(proc.stderr, limit, err_holder), daemon=True
    )
    t_out.start()
    t_err.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        raise
    t_out.join(timeout=5)
    t_err.join(timeout=5)
    out_text, out_trunc = out_holder[0] if out_holder else ("", False)
    err_text, err_trunc = err_holder[0] if err_holder else ("", False)
    return out_text, err_text, out_trunc or err_trunc


def run_request(
    req: K8sRequest,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
    max_capture: int = _MAX_CAPTURE,
) -> dict[str, Any]:
    """Execute validated request; returns JSON-serializable result."""
    kb = _kubectl_bin()
    if not kb:
        return {
            "ok": False,
            "error": "kubectl_not_found",
            "hint": "Install kubectl and ensure it is on PATH.",
        }
    argv = _argv_for(req)
    argv[0] = kb
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return {
            "ok": False,
            "error": "kubectl_spawn_failed",
            "detail": str(exc),
            "argv": argv,
        }
    try:
        out, err, truncated = _capture_bounded(proc, limit=max_capture, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": "kubectl_timeout",
            "detail": f"exceeded {timeout}s",
            "argv": argv,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "truncated": False,
        }
    return {
        "ok": proc.returncode == 0,
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": out,
        "stderr": err,
        "truncated": truncated,
    }


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"ok": False, "error": "empty_stdin", "hint": "POST JSON request body."}))
        sys.exit(2)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"ok": False, "error": "invalid_json", "detail": str(e)}))
        sys.exit(2)
    try:
        req = _ADAPTER.validate_python(payload)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "validation_failed", "detail": str(e)}))
        sys.exit(2)
    print(json.dumps(run_request(req)))


if __name__ == "__main__":
    main()
