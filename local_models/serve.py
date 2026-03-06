#!/usr/bin/env python3
"""Local model server for hermes-agent.

Launches Qwen3.5-9B via mlx_vlm on Apple Silicon GPU.

Usage:
    python3 -m local_models.serve qwen

    # Then in another terminal:
    hermes --provider local --model local/qwen3.5-9b
"""

import argparse
import sys

# Model alias → (model_path, default_port)
MODELS = {
    "qwen": ("mlx-community/Qwen3.5-9B-4bit", 8800),
}

MODEL_IDS = {
    "qwen": "local/qwen3.5-9b",
}


def serve_vlm(model_path: str, port: int):
    """Launch mlx_vlm server for Qwen3.5-9B (VLM architecture).

    Patches the default model path so mlx_vlm preloads our model,
    then starts the uvicorn server in-process.
    """
    print(f"Starting MLX-VLM server: {model_path} on port {port}")
    print(f"  Endpoint: http://127.0.0.1:{port}/v1")
    print(f"  Model ID: local/qwen3.5-9b")
    print()
    print(f"  Connect hermes:")
    print(f"    uv run hermes --provider local --model local/qwen3.5-9b")
    print()

    # Patch the default model path before importing the server app
    import mlx_vlm.server as vlm_server
    vlm_server.DEFAULT_MODEL_PATH = model_path

    # Hermes sends model="local/qwen3.5-9b" but mlx_vlm tries to load it from HF.
    # Monkey-patch the model loader to rewrite aliases to real HF repo paths.
    _MODEL_ALIASES = {
        "local/qwen3.5-9b": model_path,
        "qwen3.5-9b": model_path,
    }
    _orig_load = vlm_server.load_model_resources
    def _patched_load(mp, adapter_path=None):
        mp = _MODEL_ALIASES.get(mp, mp)
        return _orig_load(mp, adapter_path)
    vlm_server.load_model_resources = _patched_load

    # Also patch the VLMRequest default so the model field default is correct
    vlm_server.VLMRequest.model_fields["model"].default = model_path

    # mlx_vlm exposes routes at / (e.g. /chat/completions) but OpenAI clients
    # expect /v1/chat/completions. Mount the app under /v1 and also at / for compat.
    from starlette.applications import Starlette
    from starlette.routing import Mount
    wrapper = Starlette(routes=[
        Mount("/v1", app=vlm_server.app),
        Mount("/", app=vlm_server.app),
    ])

    import uvicorn
    try:
        uvicorn.run(wrapper, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        pass


def main():
    p = argparse.ArgumentParser(
        description="Launch Qwen3.5-9B local model server for hermes-agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  qwen  Qwen3.5-9B (4-bit MLX-VLM, GPU) — port 8800

Examples:
  %(prog)s qwen                    # Qwen3.5-9B VLM on GPU
  %(prog)s qwen --port 9000        # Override port
""",
    )
    p.add_argument("model", choices=list(MODELS.keys()), help="Which model to serve")
    p.add_argument("--port", type=int, default=None, help="Override default port")
    args = p.parse_args()

    model_path, default_port = MODELS[args.model]
    port = args.port or default_port

    serve_vlm(model_path, port)


if __name__ == "__main__":
    main()
