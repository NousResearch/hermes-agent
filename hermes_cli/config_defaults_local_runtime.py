"""Default block for the managed llama.cpp runtime (see ``DEFAULT_CONFIG["local_runtime"]``).

Pure-data leaf module, like the other ``config_defaults_*`` siblings: comments here are the
user-facing documentation of that section of ``config.yaml``.
"""

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
    # Heavy-engine residency: this llama-server and the media sidecar (image/video/audio
    # generation) each want the GPU to themselves, and admission arbitrates the one shared slot.
    # This build only has an observer. "shadow" logs, per boot attempt, what admission WOULD have
    # decided (grant or refusal, and who holds the slot) to local_runtime/residency-shadow.jsonl
    # and boots anyway - nothing is refused yet. "off" does not even ask.
    # An unrecognised value is rejected the way a bad "backend" is, at boot: a mode this build
    # cannot read must not quietly mean "arbitration is on".
    "residency_mode": "off",
}
