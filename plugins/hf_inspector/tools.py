"""Agent-facing tools for the hf_inspector plugin.

Provides:
  hf_inspect_model — fetch architecture, parameter count, context length, license, and gated status.
  hf_list_quants   — discover GGUF, AWQ, and GPTQ files with sizes and download links.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HF_API_BASE = "https://huggingface.co/api/models"
_USER_AGENT = "Hermes-Agent-HF-Inspector/1.0"
_TIMEOUT_SECONDS = 10


HF_INSPECT_MODEL_SCHEMA = {
    "name": "hf_inspect_model",
    "description": (
        "Inspect a Hugging Face model repository to retrieve metadata, including "
        "architecture, parameter count, context length (max position embeddings), "
        "pipeline task, license, downloads, tags, and gated status."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_id": {
                "type": "string",
                "description": (
                    "Hugging Face repository ID (e.g. 'NousResearch/Hermes-3-Llama-3.1-8B', "
                    "'Qwen/Qwen2.5-Coder-32B-Instruct'). Required."
                ),
            },
        },
        "required": ["model_id"],
    },
}


HF_LIST_QUANTS_SCHEMA = {
    "name": "hf_list_quants",
    "description": (
        "Discover available quantized files (GGUF, AWQ, GPTQ) in a Hugging Face "
        "repository or its dedicated GGUF companion repo, reporting quantization types, "
        "file sizes, and direct download links."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_id": {
                "type": "string",
                "description": (
                    "Hugging Face repository ID (e.g. 'NousResearch/Hermes-3-Llama-3.1-8B-GGUF' "
                    "or 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF'). Required."
                ),
            },
        },
        "required": ["model_id"],
    },
}


def _fetch_hf_api(endpoint: str) -> Dict[str, Any]:
    """Fetch JSON from Hugging Face REST API."""
    req = urllib.request.Request(
        endpoint,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data)


def _format_size(bytes_num: Optional[int]) -> str:
    """Format bytes into readable size string (MB/GB)."""
    if bytes_num is None or bytes_num <= 0:
        return "Unknown size"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_num < 1024.0:
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.2f} PB"


def _format_params(params_num: Optional[int]) -> str:
    """Format parameter count into human readable B/M count."""
    if not params_num or params_num <= 0:
        return "Unknown"
    if params_num >= 1_000_000_000:
        return f"{params_num / 1_000_000_000:.1f}B"
    if params_num >= 1_000_000:
        return f"{params_num / 1_000_000:.1f}M"
    return str(params_num)


def handle_hf_inspect_model(model_id: str, **kwargs: Any) -> str:
    """Fetch and format model metadata from Hugging Face."""
    clean_id = (model_id or "").strip()
    if not clean_id:
        return "Error: model_id is required."

    url = f"{_HF_API_BASE}/{clean_id}"
    try:
        data = _fetch_hf_api(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Error: Hugging Face model '{clean_id}' not found (404)."
        if e.code == 401 or e.code == 403:
            return f"Error: Model '{clean_id}' is gated or private (HTTP {e.code})."
        return f"Error fetching Hugging Face model '{clean_id}': HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"Network error connecting to Hugging Face: {e.reason}"
    except Exception as e:
        return f"Unexpected error inspecting model '{clean_id}': {e}"

    # Extract model metadata fields
    pipeline_tag = data.get("pipeline_tag") or "unknown"
    downloads = data.get("downloads", 0)
    likes = data.get("likes", 0)
    gated = data.get("gated", False)
    tags = data.get("tags") or []
    
    # Safetensors parameter counts
    safetensors_info = data.get("safetensors") or {}
    total_params = safetensors_info.get("total")
    params_str = _format_params(total_params)

    # Config / architecture attributes
    config = data.get("config") or {}
    arch = None
    if isinstance(data.get("transformersInfo"), dict):
        arch = data["transformersInfo"].get("architectures")
    if not arch and "architectures" in config:
        arch = config.get("architectures")
    arch_str = ", ".join(arch) if isinstance(arch, list) else str(arch or "Unknown")

    # Context length / max position embeddings
    context_length = (
        config.get("max_position_embeddings")
        or config.get("context_length")
        or config.get("seq_length")
        or config.get("max_sequence_length")
    )
    context_str = f"{context_length:,} tokens" if isinstance(context_length, int) else "Not specified"

    # License
    license_tag = "Unknown"
    for tag in tags:
        if tag.startswith("license:"):
            license_tag = tag.split(":", 1)[1]
            break

    out = [
        f"# Hugging Face Model: {clean_id}",
        f"- Pipeline / Task: {pipeline_tag}",
        f"- Architecture: {arch_str}",
        f"- Parameters: {params_str}",
        f"- Context Length: {context_str}",
        f"- License: {license_tag}",
        f"- Community Stats: {downloads:,} downloads | {likes:,} likes",
        f"- Gated Access: {'Yes (Requires approval)' if gated else 'No (Public)'}",
        f"- URL: https://huggingface.co/{clean_id}",
    ]

    return "\n".join(out)


def handle_hf_list_quants(model_id: str, **kwargs: Any) -> str:
    """List quantized files (GGUF, AWQ, GPTQ) in a repository."""
    clean_id = (model_id or "").strip()
    if not clean_id:
        return "Error: model_id is required."

    url = f"{_HF_API_BASE}/{clean_id}"
    try:
        data = _fetch_hf_api(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Error: Hugging Face repository '{clean_id}' not found (404)."
        return f"Error accessing repository '{clean_id}': HTTP {e.code} {e.reason}"
    except Exception as e:
        return f"Unexpected error listing files for '{clean_id}': {e}"

    siblings = data.get("siblings") or []
    if not siblings:
        return f"No sibling files found in repository '{clean_id}'."

    quants: List[Dict[str, Any]] = []
    for f in siblings:
        rfilename = f.get("rfilename", "")
        low = rfilename.lower()
        if low.endswith(".gguf") or "awq" in low or "gptq" in low:
            size_bytes = f.get("size")
            size_str = _format_size(size_bytes)
            file_url = f"https://huggingface.co/{clean_id}/resolve/main/{rfilename}"
            quants.append({
                "filename": rfilename,
                "size": size_str,
                "url": file_url,
            })

    if not quants:
        return (
            f"No GGUF, AWQ, or GPTQ quantized files found in '{clean_id}'.\n"
            f"Tip: If '{clean_id}' is the base model, try checking companion GGUF repos (e.g. '{clean_id}-GGUF' or bartowski quants)."
        )

    out = [f"# Quantized Files in {clean_id} ({len(quants)} found):"]
    for q in quants:
        out.append(f"- `{q['filename']}` ({q['size']}) -> {q['url']}")

    return "\n".join(out)
