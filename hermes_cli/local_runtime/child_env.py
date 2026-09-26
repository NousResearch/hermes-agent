"""Environment policy for native inference children."""

from __future__ import annotations

from collections.abc import Mapping


_CREDENTIAL_ENV_MARKERS = ("_API_KEY", "_TOKEN", "_SECRET", "PASSWORD", "_CREDENTIALS")


def server_child_env(base_env: Mapping[str, str]) -> dict[str, str]:
    """Return the environment a native inference child (llama-server) gets.

    Provider and tool credentials never belong in a native child that talks to nobody but
    us — and on Windows they are not merely leaked: the bundled OpenMP runtime died with
    STATUS_HEAP_CORRUPTION during initialisation with one `*_API_KEY` present and loaded fine
    with only that variable removed (#116109, confirmed on a model-free libomp.dll probe).
    Everything else (PATH, CUDA_*, HSA_*, OMP_*, TEMP, …) passes through untouched. Applied by
    the llama-server supervisor only: the generic contained-process spawner also backs bounded
    probes (git / PowerShell / update probes), whose children legitimately need GH_TOKEN,
    HF_TOKEN, …
    """
    return {
        key: value for key, value in base_env.items()
        if not any(marker in key.upper() for marker in _CREDENTIAL_ENV_MARKERS)
    }
