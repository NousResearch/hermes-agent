"""Per-thread sandbox substrate (S7 — integration with DeerFlow §1.C).

Purpose
-------
Give each bus task a per-thread workspace directory:
    ~/.hermes/threads/{task_id}/user-data/{workspace,uploads,outputs}

Agent code sees **virtual paths** (`/mnt/user-data/...`) which get translated
to actual paths on the host. This lets us later swap the backing store
(local filesystem → Docker volume → K8s PVC) without changing agent code.

Not in this slice
-----------------
- Real tool integration (bash/read_file/write_file): would require replacing
  the existing tool implementations; parked for a later pass
- Docker/K8s providers: LocalSandbox only for now

What this enables
-----------------
- `workspace_to_wiki_sync.py` already handles OpenClaw → shared wiki copy
  (D-direction sync). With per-thread dirs, we can also copy
  `~/.hermes/threads/{id}/outputs/*` to shared wiki as a separate direction
- Dashboard can display per-thread outputs
- Cleanup is just `rm -rf threads/{id}/`
"""

from agent_bus.sandbox.sandbox import (
    Sandbox,
    SandboxProvider,
    SandboxError,
)
from agent_bus.sandbox.local import LocalSandbox, LocalSandboxProvider
from agent_bus.sandbox.virtual_path import (
    VIRTUAL_ROOT,
    translate_virtual_path,
    replace_virtual_paths_in_text,
)

__all__ = [
    "Sandbox",
    "SandboxProvider",
    "SandboxError",
    "LocalSandbox",
    "LocalSandboxProvider",
    "VIRTUAL_ROOT",
    "translate_virtual_path",
    "replace_virtual_paths_in_text",
]
