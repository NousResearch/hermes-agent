"""DGX config helpers — read/write the ``dgx:`` block in config.yaml
and the ``model:`` block that points at DGX endpoints.

Multi-node design: the ``dgx.nodes`` dict holds named DGX instances;
``dgx.active_node`` selects which one is currently active.  Single-node
configs (flat ``dgx.host`` etc.) are transparently migrated on first read.

Defaults are intentionally unset (None) for host-specific values so that
nothing accidentally talks to a stranger's network. The user supplies the
host by running ``hermes dgx setup`` (interactive), which writes the ``dgx:``
block to config.yaml. Host, ports and the LiteLLM host are non-secret
behavioral settings, so per AGENTS.md they live in config.yaml — never in new
``HERMES_*`` env vars (``.env`` is for secrets only).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# SSH user falls back to the local $USER so the same machine can be both the
# controller and the GPU host without configuring anything. $USER is the
# standard login env var, not a HERMES_* config knob; everything else comes
# from config.yaml (written by `hermes dgx setup`).
_DEFAULT_SSH_USER = os.environ.get("USER") or "dgx"

NODE_DEFAULTS: Dict[str, Any] = {
    "host": None,
    "ssh_user": _DEFAULT_SSH_USER,
    "ollama_port": 11434,
    "vllm_port": 30800,
    "name": "DGX Spark",
}

DEFAULTS: Dict[str, Any] = {
    # flat keys kept for backwards compat and single-node convenience
    "host": None,
    "ssh_user": _DEFAULT_SSH_USER,
    "ollama_port": 11434,
    "vllm_port": 30800,
    "vllm_32b_port": 30881,
    "litellm_host": None,
    "litellm_port": 4000,
    "active_endpoint": "ollama",
    "default_model": "qwen2.5-coder:latest",
}

ENDPOINT_LABELS = {
    "ollama":    "Ollama (direct, no auth)",
    "vllm":      "vLLM 3B (port 30800, always ready)",
    "vllm-32b":  "vLLM 32B (port 30881, qwen2.5-coder-32b)",
    "litellm":   "LiteLLM proxy (HA pool, requires API key)",
}

# ---------------------------------------------------------------------------
# Config accessors
# ---------------------------------------------------------------------------

def load_dgx_config() -> Dict[str, Any]:
    from hermes_cli.config import load_config
    cfg = load_config()
    dgx = dict(DEFAULTS)
    dgx.update(cfg.get("dgx") or {})
    # Inject _active_node for tools that want the currently selected node
    dgx["_active_node"] = _resolve_active_node(dgx)
    return dgx


def _resolve_active_node(dgx: Dict[str, Any]) -> Dict[str, Any]:
    """Return the active node dict (host, ssh_user, ports).

    Falls back to the flat dgx keys for single-node configs.
    """
    # single-node / legacy: synthesise a node from flat keys
    return {
        "host":        dgx["host"],
        "ssh_user":    dgx["ssh_user"],
        "ollama_port": dgx["ollama_port"],
        "vllm_port":   dgx["vllm_port"],
        "name":        dgx.get("name", "DGX Spark"),
    }


def save_dgx_config(dgx: Dict[str, Any]) -> None:
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    # Don't persist the injected _active_node helper key
    to_save = {k: v for k, v in dgx.items() if not k.startswith("_")}
    cfg["dgx"] = to_save
    save_config(cfg)


def apply_endpoint(dgx: Dict[str, Any], endpoint: Optional[str] = None) -> None:
    """Write model.provider + model.base_url to point at the given endpoint."""
    from hermes_cli.config import load_config, save_config

    ep = endpoint or dgx.get("active_endpoint", "ollama")
    node = dgx.get("_active_node") or _resolve_active_node(dgx)
    host = node["host"]

    if ep == "ollama":
        base_url = f"http://{host}:{node['ollama_port']}/v1"
        provider = "ollama"
    elif ep == "vllm":
        port = node['vllm_port']
        base_url = f"http://{host}:{port}/v1"
        provider = "custom"
    elif ep == "vllm-32b":
        port = dgx.get("vllm_32b_port", 30881)
        base_url = f"http://{host}:{port}/v1"
        provider = "custom"
    elif ep == "litellm":
        lh = dgx.get("litellm_host")
        lp = dgx.get("litellm_port", 4000)
        if not lh:
            raise ValueError(
                "litellm endpoint requires dgx.litellm_host "
                "(run `hermes dgx setup` to configure it)"
            )
        base_url = f"http://{lh}:{lp}/v1"
        provider = "custom"
    else:
        raise ValueError(f"Unknown endpoint: {ep!r}")

    dgx["active_endpoint"] = ep

    cfg = load_config()
    if not isinstance(cfg.get("model"), dict):
        cfg["model"] = {}
    cfg["model"]["provider"] = provider
    cfg["model"]["base_url"] = base_url
    to_save = {k: v for k, v in dgx.items() if not k.startswith("_")}
    cfg["dgx"] = to_save
    save_config(cfg)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

class DGXNotConfigured(RuntimeError):
    """Raised when a DGX host has not been configured yet."""


def _require_host(node: Dict[str, Any]) -> str:
    host = node.get("host")
    if not host:
        raise DGXNotConfigured(
            "DGX host is not configured. Run `hermes dgx setup` to configure it."
        )
    return host


def ollama_base(dgx: Dict[str, Any]) -> str:
    node = dgx.get("_active_node") or _resolve_active_node(dgx)
    return f"http://{_require_host(node)}:{node['ollama_port']}"


def vllm_base(dgx: Dict[str, Any]) -> str:
    node = dgx.get("_active_node") or _resolve_active_node(dgx)
    return f"http://{_require_host(node)}:{node['vllm_port']}"


def litellm_base(dgx: Dict[str, Any]) -> Optional[str]:
    """Return the LiteLLM base URL, or None if not configured."""
    lh = dgx.get("litellm_host")
    if not lh:
        return None
    lp = dgx.get("litellm_port", 4000)
    return f"http://{lh}:{lp}"
