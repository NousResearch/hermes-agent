#!/usr/bin/env python3
"""
check_deps.py — Check if a ComfyUI workflow's dependencies (custom nodes and models) are installed.

Queries the running ComfyUI server for installed nodes (via /object_info) and models
(via /models/{folder}), then diffs against what the workflow requires.

Usage:
    python3 check_deps.py workflow_api.json
    python3 check_deps.py workflow_api.json --host 127.0.0.1 --port 8188
    python3 check_deps.py workflow_api.json --host https://cloud.comfy.org --api-key KEY

Output format:
    {
      "is_ready": true/false,
      "missing_nodes": ["NodeClassName", ...],
      "missing_models": [{"class_type": "...", "field": "...", "value": "...", "folder": "..."}],
      "installed_nodes_count": 123,
      "required_nodes": ["KSampler", "CLIPTextEncode", ...]
    }

Requires: Python 3.10+, requests (or urllib as fallback)
"""

import json
import sys
import argparse
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error

# Known model loader node types and which folder they reference
MODEL_LOADERS = {
    "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
    "CheckpointLoader": ("ckpt_name", "checkpoints"),
    "unCLIPCheckpointLoader": ("ckpt_name", "checkpoints"),
    "LoraLoader": ("lora_name", "loras"),
    "LoraLoaderModelOnly": ("lora_name", "loras"),
    "VAELoader": ("vae_name", "vae"),
    "ControlNetLoader": ("control_net_name", "controlnet"),
    "DiffControlNetLoader": ("control_net_name", "controlnet"),
    "CLIPLoader": ("clip_name", "clip"),
    "DualCLIPLoader": ("clip_name1", "clip"),
    "UNETLoader": ("unet_name", "unet"),
    "DiffusionModelLoader": ("model_name", "diffusion_models"),
    "UpscaleModelLoader": ("model_name", "upscale_models"),
    "CLIPVisionLoader": ("clip_name", "clip_vision"),
    "StyleModelLoader": ("style_model_name", "style_models"),
    "GLIGENLoader": ("gligen_name", "gligen"),
    "HypernetworkLoader": ("hypernetwork_name", "hypernetworks"),
}


def http_get(url: str, headers: dict = None) -> tuple:
    """GET request, returns (status_code, body_text)."""
    if HAS_REQUESTS:
        r = requests.get(url, headers=headers or {}, timeout=30)
        return r.status_code, r.text
    else:
        req = urllib.request.Request(url, headers=headers or {})
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()


def check_deps(workflow_path: str, host: str = "http://127.0.0.1:8188", api_key: str = None):
    """Check workflow dependencies against a running server."""
    # Load workflow
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Validate format
    if "nodes" in workflow and "links" in workflow:
        return {"error": "Workflow is in editor format, not API format."}

    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    parsed_host = urlparse(host)
    hostname = (parsed_host.hostname or "").lower()
    is_cloud_host = hostname == "cloud.comfy.org" or hostname.endswith(".cloud.comfy.org")
    is_cloud = is_cloud_host or api_key is not None
    base = host.rstrip("/")

    # Get installed node types
    object_info_url = f"{base}/api/object_info" if is_cloud else f"{base}/object_info"
    status, body = http_get(object_info_url, headers)
    if status != 200:
        return {"error": f"Cannot reach server at {host}. Is ComfyUI running? HTTP {status}"}

    installed_nodes = set(json.loads(body).keys())

    # Find required node types from workflow
    required_nodes = set()
    for node_id, node in workflow.items():
        if isinstance(node, dict) and "class_type" in node:
            required_nodes.add(node["class_type"])

    missing_nodes = sorted(required_nodes - installed_nodes)

    # Check model dependencies
    missing_models = []
    model_cache = {}  # folder → set of installed model filenames

    for node_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        class_type = node["class_type"]
        if class_type not in MODEL_LOADERS:
            continue

        field, folder = MODEL_LOADERS[class_type]
        inputs = node.get("inputs", {})
        model_name = inputs.get(field)

        if not model_name or not isinstance(model_name, str):
            continue

        # Fetch installed models for this folder (cached)
        if folder not in model_cache:
            models_url = f"{base}/api/models/{folder}" if is_cloud else f"{base}/models/{folder}"
            s, b = http_get(models_url, headers)
            if s == 200:
                model_cache[folder] = set(json.loads(b))
            else:
                model_cache[folder] = set()

        if model_name not in model_cache[folder]:
            missing_models.append({
                "node_id": node_id,
                "class_type": class_type,
                "field": field,
                "value": model_name,
                "folder": folder,
            })

    is_ready = len(missing_nodes) == 0 and len(missing_models) == 0

    return {
        "is_ready": is_ready,
        "missing_nodes": missing_nodes,
        "missing_models": missing_models,
        "installed_nodes_count": len(installed_nodes),
        "required_nodes": sorted(required_nodes),
    }


def main():
    parser = argparse.ArgumentParser(description="Check ComfyUI workflow dependencies")
    parser.add_argument("workflow", help="Path to workflow API JSON file")
    parser.add_argument("--host", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    parser.add_argument("--port", type=int, help="Server port (overrides --host port)")
    parser.add_argument("--api-key", help="API key for cloud")
    args = parser.parse_args()

    # Handle --port override
    host = args.host
    if args.port and ":" not in host.split("//")[-1]:
        host = f"{host}:{args.port}"

    result = check_deps(args.workflow, host=host, api_key=args.api_key)
    print(json.dumps(result, indent=2))

    if result.get("error"):
        sys.exit(1)
    if not result.get("is_ready", False):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
