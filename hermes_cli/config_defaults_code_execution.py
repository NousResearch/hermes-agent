"""Pure-data factory for ``execute_code`` defaults (``code_execution:``).

``mode``: ``project`` runs in the session cwd with the active venv/conda python so project deps
and relative paths resolve; ``strict`` uses an isolated temp dir with hermes-agent's own python
(``sys.executable``), so project deps/relative paths do not work. Env scrubbing (*_API_KEY,
*_TOKEN, *_SECRET, ...) and the tool whitelist apply in both modes.

Session kernels: always on locally (``kernel_mode`` is ignored) and remotely
(``tools/code_kernel_remote.py``; a backend that cannot spawn a kernel fails open to per-call).
One kernel per (session owner, mode, interpreter, cwd, tool-set) keeps state across calls and
turns; subagents get their own. Kernels die with the session, after ``kernel_idle_timeout`` idle
seconds, or by LRU eviction past ``max_session_kernels``. A timed-out/interrupted cell kills the
kernel; env is frozen at spawn (reset=true after changing passthrough). Tool RPC authority
(approval, session, allow-list, call budget) is rebound per cell — that runtime boundary is the
cross-cell enforcement.
"""


def _code_execution(mode: str = "project", kernel_idle_timeout: int = 1800,
                    max_session_kernels: int = 4) -> dict:
    """One ``code_execution`` config block (see ``DEFAULT_CONFIG['code_execution']``)."""
    return {"mode": mode, "kernel_idle_timeout": kernel_idle_timeout,
            "max_session_kernels": max_session_kernels}
