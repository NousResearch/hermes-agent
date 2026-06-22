"""Agent tools for the dgx plugin.

Registers two tools so the agent can manage the DGX mid-conversation
without the user needing to run CLI commands manually:

  dgx_gpu_status   — current GPU memory + loaded models
  dgx_pull_model   — pull an Ollama model onto the DGX

There is intentionally NO ``dgx_run`` (arbitrary remote shell) tool: free-form
command execution must go through the host terminal tool, which applies the
dangerous-command approval gate. A bespoke SSH-exec tool would bypass it.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schemas (OpenAI function-call format)
# ---------------------------------------------------------------------------

DGX_GPU_STATUS_SCHEMA = {
    "name": "dgx_gpu_status",
    "description": (
        "Get current GPU memory usage and models loaded in GPU memory on the "
        "DGX Spark. Use before pulling a large model to verify there is enough "
        "free unified memory (128 GB total on a single Spark)."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

DGX_PULL_MODEL_SCHEMA = {
    "name": "dgx_pull_model",
    "description": (
        "Pull an Ollama model into the DGX Spark. Call dgx_gpu_status first to "
        "confirm enough free unified memory. Large models (70B+) can take several "
        "minutes — the tool blocks until the pull completes or times out."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": (
                    "Ollama model name, e.g. nemotron3:70b, qwen2.5-coder:32b, "
                    "deepseek-r1:70b. Use the full tag to ensure a specific variant."
                ),
            },
        },
        "required": ["model"],
    },
}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_dgx_gpu_status(**_kwargs) -> str:
    from plugins.dgx._dgx_config import load_dgx_config
    from plugins.dgx.cli import _get_json, _ssh_run, ollama_base

    dgx = load_dgx_config()
    node = dgx.get("_active_node", dgx)  # multi-node aware
    host, user = node["host"], node["ssh_user"]
    if not host:
        return "DGX is not configured. Run `hermes dgx setup` first."
    parts: list[str] = []

    # nvidia-smi
    ok, out = _ssh_run(
        user, host,
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits",
        timeout=10,
    )
    if ok and out:
        parts.append(f"GPU ({host}):\n{out}")
    else:
        parts.append(f"GPU ({host}): unavailable ({out})")

    # ollama ps — what's loaded right now
    ok2, out2 = _ssh_run(user, host, "ollama ps", timeout=10)
    if ok2 and out2.strip():
        parts.append(f"Loaded models:\n{out2}")
    else:
        parts.append("Loaded models: none")

    return "\n\n".join(parts)


def handle_dgx_pull_model(model: str, **_kwargs) -> str:
    from plugins.dgx._dgx_config import load_dgx_config
    from plugins.dgx.cli import _ssh_run

    dgx = load_dgx_config()
    node = dgx.get("_active_node", dgx)
    host, user = node["host"], node["ssh_user"]
    if not host:
        return "DGX is not configured. Run `hermes dgx setup` first."
    ok, out = _ssh_run(user, host, f"ollama pull {model}", timeout=600)
    if ok:
        return f"Successfully pulled {model} on {host}."
    return f"Failed to pull {model} on {host}: {out}"
