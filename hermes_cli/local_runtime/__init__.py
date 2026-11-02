"""Managed llama.cpp runtime.

``binaries`` answers which backend this machine uses and which PM-pinned
engine is installed (the engine bytes live in pm's machine-wide store);
``bootstrap`` boots the managed server from what is installed — never a
download; ``supervisor`` spawns and supervises one llama-server in router
mode (readiness is a touch generation, never health-200 alone); ``detect``
finds an already-running llama-server (external or ours).

The package-level names resolve on first access. Shipped updaters import
``hermes_cli.local_runtime.processes`` from the NEW checkout after the pull,
while their process still holds the OLD ``hermes_platform`` and other
first-party modules. An eager re-export here would run every submodule
against those stale modules and crash the update on any name added since.
"""

from importlib import import_module

_EXPORTS = {
    "binaries": ("BACKEND_PACKAGES", "BinaryResolutionError", "Engine", "ensure_engine",
                 "installed_engine", "pinned_tag", "resolve_backend", "select_backend"),
    "bootstrap": ("ensure_local_runtime", "shutdown_local_runtime"),
    "context_policy": ("FLOOR", "growth_decision", "initial_window", "ladder", "launch_args"),
    "growth": ("clear_window_override", "load_window_overrides", "maybe_grow_window",
               "save_window_override"),
    "detect": ("detect_server",),
    "endpoint": ("resolve_llamacpp_endpoint",),
    "estimator": ("HardwareBudget", "ctx_bytes", "physics_check", "profile_from_gguf"),
    "gguf": ("read_gguf_header",),
    "hardware": ("probe_budget",),
    "presets": ("generate_presets",),
    "supervisor": ("LlamaServerSupervisor",),
}
_SOURCE = {name: module for module, names in _EXPORTS.items() for name in names}


def __getattr__(name: str):
    module = _SOURCE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
