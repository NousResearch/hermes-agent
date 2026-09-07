"""Managed llama.cpp runtime.

Hermes downloads, verifies, supervises, and updates one llama-server, and
decides per machine which model build and context window to run. Key
modules:

- ``binaries``  — resolve/download/verify official llama.cpp release zips
                  into ``$HERMES_HOME/runtimes/llamacpp/<tag>/``.
- ``supervisor``— spawn and supervise one llama-server in router mode;
                  readiness is a touch generation, never health-200 alone.
- ``detect``    — find an already-running llama-server (external or ours).
- ``estimator`` / ``context_policy`` / ``growth`` — price context memory
  per architecture and run the window ladder (zero-spill start, grow
  toward native max, compress only at the top).
- ``catalog`` / ``presets`` — the curated model list and the per-model
  launch flags that carry policy decisions to the router.

Everything is driven by the ``local_runtime`` section of config.yaml.
"""

from hermes_cli.local_runtime.binaries import (  # noqa: F401
    BinaryResolutionError,
    ensure_runtime_installed,
    resolve_assets,
    select_backend,
)
from hermes_cli.local_runtime.bootstrap import (  # noqa: F401
    ensure_local_runtime,
    shutdown_local_runtime,
)
from hermes_cli.local_runtime.context_policy import (  # noqa: F401
    FLOOR,
    growth_decision,
    initial_window,
    ladder,
    launch_args,
)
from hermes_cli.local_runtime.growth import (  # noqa: F401
    clear_window_override,
    load_window_overrides,
    maybe_grow_window,
    save_window_override,
)
from hermes_cli.local_runtime.detect import detect_server  # noqa: F401
from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint  # noqa: F401
from hermes_cli.local_runtime.estimator import (  # noqa: F401
    HardwareBudget,
    ctx_bytes,
    physics_check,
    profile_from_gguf,
)
from hermes_cli.local_runtime.gguf import read_gguf_header  # noqa: F401
from hermes_cli.local_runtime.hardware import probe_budget  # noqa: F401
from hermes_cli.local_runtime.presets import generate_presets  # noqa: F401
from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor  # noqa: F401
