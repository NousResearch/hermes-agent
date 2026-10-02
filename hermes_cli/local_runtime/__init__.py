"""Managed llama.cpp runtime.

``binaries`` answers which backend this machine uses and which PM-pinned
engine is installed (the engine bytes live in pm's machine-wide store);
``bootstrap`` boots the managed server from what is installed — never a
download; ``supervisor`` spawns and supervises one llama-server in router
mode (readiness is a touch generation, never health-200 alone); ``detect``
finds an already-running llama-server (external or ours).
"""

from hermes_cli.local_runtime.binaries import (  # noqa: F401
    BACKEND_PACKAGES, BinaryResolutionError, Engine, ensure_engine,
    installed_engine, pinned_tag, resolve_backend, select_backend)
from hermes_cli.local_runtime.bootstrap import ensure_local_runtime, shutdown_local_runtime  # noqa: F401
from hermes_cli.local_runtime.context_policy import (  # noqa: F401
    FLOOR, growth_decision, initial_window, ladder, launch_args)
from hermes_cli.local_runtime.growth import (  # noqa: F401
    clear_window_override, load_window_overrides, maybe_grow_window, save_window_override)
from hermes_cli.local_runtime.detect import detect_server  # noqa: F401
from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint  # noqa: F401
from hermes_cli.local_runtime.estimator import (  # noqa: F401
    HardwareBudget, ctx_bytes, physics_check, profile_from_gguf)
from hermes_cli.local_runtime.gguf import read_gguf_header  # noqa: F401
from hermes_cli.local_runtime.hardware import probe_budget  # noqa: F401
from hermes_cli.local_runtime.presets import generate_presets  # noqa: F401
from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor  # noqa: F401
