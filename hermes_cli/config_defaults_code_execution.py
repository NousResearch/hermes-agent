"""``execute_code`` configuration defaults (pure-data leaf, no imports).

Kept out of ``config_defaults.py`` so that facade only shrinks: a section that
grows a key moves here instead of pushing the file past its line cap.
"""

# project = run in the session cwd with the active venv/conda python so project deps and
# relative paths resolve. strict = isolated temp dir with hermes-agent's own python
# (sys.executable): max isolation, project deps/relative paths won't work. Env scrubbing
# (*_API_KEY, *_TOKEN, *_SECRET, ...) and the tool whitelist apply in both modes.
#
# Session kernels are always on locally (`kernel_mode` is ignored) and remotely
# (tools/code_kernel_remote.py; a backend that cannot spawn a kernel fails open to
# per-call). One kernel per (session owner, mode, interpreter, cwd, tool-set) keeps state
# across calls and turns; subagents get their own. Kernels die with the session, after
# kernel_idle_timeout idle seconds, or by LRU eviction past max_session_kernels. A
# timed-out/interrupted cell kills the kernel; env is frozen at spawn (reset=true after
# changing passthrough). Tool RPC authority (approval, session, allow-list, call budget) is
# rebound per cell - that runtime boundary is the cross-cell enforcement.
CODE_EXECUTION_DEFAULTS = {
    "mode": "project",
    "kernel_idle_timeout": 1800,
    "max_session_kernels": 4,
    # Address-space budget for ONE kernel child, in MB. A cell is otherwise bounded only
    # by the 300 s wall clock, which is ample time to allocate hundreds of GB: one runaway
    # cell took down the backend, the desktop renderer and unrelated host processes. Past
    # this budget the cell dies ALONE (Windows: a Job Object; POSIX: RLIMIT_AS in the
    # runner). 0 = unlimited, which also drops the Windows kill-with-host containment.
    "memory_limit_mb": 4096,
}
