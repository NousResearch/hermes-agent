"""Detection of running llama-server instances.

Probes well-known local roots and fingerprints genuine llama-server via
/props (build_info + model fields — Ollama and LM Studio answer /v1/models
but not /props). The credential is reachability; detection never needs a
key, but honors one if the probed server requires it (401 -> detected,
auth_required=True).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

# Always 127.0.0.1 — resolving localhost costs ~2s/request on Windows.
DEFAULT_PROBE_PORTS = (8080,)  # llama-server default; managed port comes from config


@dataclass
class DetectedServer:
    base_url: str            # OpenAI-compatible /v1 root
    build_info: str          # e.g. "b10290-c8e03ce81"
    model_path: str          # currently loaded model (may be empty in router mode)
    n_ctx: int | None
    router_mode: bool        # GET /models answered -> router management available
    auth_required: bool


def _get(url: str, timeout_s: int = 3) -> tuple[int, dict | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        return 0, None


def probe_port(port: int) -> DetectedServer | None:
    """One port: /props fingerprint, then /models for router capability."""
    root = f"http://127.0.0.1:{port}"
    status, props = _get(f"{root}/props")
    if status == 401:
        return DetectedServer(base_url=f"{root}/v1", build_info="", model_path="",
                              n_ctx=None, router_mode=False, auth_required=True)
    if status != 200 or not isinstance(props, dict):
        return None
    build = str(props.get("build_info", ""))
    if not build:
        return None  # answers /props but isn't llama-server
    n_ctx = None
    dgs = props.get("default_generation_settings")
    if isinstance(dgs, dict):
        n_ctx = dgs.get("n_ctx")
    models_status, models = _get(f"{root}/models")
    return DetectedServer(
        base_url=f"{root}/v1",
        build_info=build,
        model_path=str(props.get("model_path", "")),
        n_ctx=n_ctx,
        router_mode=(models_status == 200 and isinstance(models, dict)
                     and "data" in models),
        auth_required=False,
    )


def detect_server(extra_ports: tuple[int, ...] = ()) -> DetectedServer | None:
    """First hit across default + extra ports (managed port, config port)."""
    seen = set()
    for port in (*DEFAULT_PROBE_PORTS, *extra_ports):
        if port in seen:
            continue
        seen.add(port)
        hit = probe_port(port)
        if hit:
            return hit
    return None
