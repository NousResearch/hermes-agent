"""``local_runtime`` defaults: the managed llama.cpp runtime section of ``DEFAULT_CONFIG``.

Pure-data leaf module, like ``hermes_cli.config_defaults`` (which embeds this dict). Comments are the
user-facing documentation of each key.
"""

# Managed llama.cpp runtime (docs: user-guide/local-models): official binaries, one supervised
# llama-server in router mode. The policy sizes context/VRAM; model_overrides is the explicit
# per-model escape hatch.
LOCAL_RUNTIME_DEFAULTS = {
    # Off = detection-only (Hermes still finds an external llama-server you run).
    "enabled": False,
    # Engine versions and every dependent library are pinned by pm/lock.json.
    # auto = CUDA on NVIDIA, Metal on macOS, Vulkan on other GPUs, else CPU. Explicit:
    # cuda|metal|vulkan|hip|cpu.
    "backend": "auto",
    "models_max": 4,  # Router process: how many models may be resident at once.
    "port": 0,  # Port for the managed server. 0 = pick a free port at spawn.
    # Extra ports detection probes for an external llama-server (besides 8080).
    "detect_ports": [],
    # A llama-server binary to supervise instead of the PM engine ("" = the PM engine).
    "executable_path": "",
    # Extra read-only directories of GGUFs served beside the managed models dir.
    "model_dirs": [],
    # model id -> llama-server preset keys laid over the launch policy's (e.g.
    # {"my-model": {"ctx-size": 32768, "flash-attn": "on"}}). Empty = policy only.
    "model_overrides": {},
}
