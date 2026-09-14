"""NOVA's own record of the automations it created.

The runtime's cron record has nowhere to put governance metadata — no tenant, no
declaring principal, no spec digest — and Hermes core is not being changed to add one.
So NOVA keeps its own file beside its audit log, keyed by the runtime's job id.

This is a **record, not a source of truth**. The runtime owns whether an automation
exists, whether it is paused and when it next runs; this file only answers "who declared
this, when, and against which reviewed declaration". A job present in the runtime with no
entry here is an automation created outside NOVA — worth showing as exactly that, rather
than hiding or inventing provenance for it.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

#: Beside the audit log, inside NOVA's own state directory under the runtime home.
REGISTRY_DIRNAME = "nova"
REGISTRY_FILENAME = "automations.json"


def registry_path(home: Path) -> Path:
    return Path(home) / REGISTRY_DIRNAME / REGISTRY_FILENAME


def _read(home: Path) -> dict[str, Any]:
    path = registry_path(home)
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A damaged registry must not take the screen down: the runtime still knows what
        # exists, and provenance degrades to "unknown" rather than to an error.
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write(home: Path, data: dict[str, Any]) -> None:
    """Atomic replace, so a crash mid-write cannot leave a half-parsed registry."""
    path = registry_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".automations-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def record(
    home: Path,
    *,
    job_id: str,
    tenant_id: str,
    compiled,
    actor: str,
    correlation_id: str,
    created_at: str,
) -> None:
    """Note that NOVA created this automation, and from what."""
    data = _read(home)
    data[job_id] = {
        "tenant_id": tenant_id,
        "automation_id": compiled.spec.id,
        "agent_id": compiled.agent_id,
        "digest": compiled.digest,
        "declared_by": actor,
        "declared_at": created_at,
        "correlation_id": correlation_id,
        "reason": compiled.spec.reason,
        "permissions": list(compiled.spec.permissions),
        "knowledge": list(compiled.spec.knowledge),
        "channels": list(compiled.spec.channels),
    }
    _write(home, data)


def forget(home: Path, job_id: str) -> None:
    """Drop the record for a deleted automation. Silent when absent."""
    data = _read(home)
    if data.pop(job_id, None) is not None:
        _write(home, data)


def governance(home: Path, job_id: str, *, tenant_id: str = "") -> Optional[dict[str, Any]]:
    """Provenance for one job, or None when NOVA did not create it.

    Tenant-checked even though the registry lives inside the tenant's own home: defence
    in depth costs one comparison, and a home shared by mistake should not become a
    cross-tenant read.
    """
    entry = _read(home).get(job_id)
    if entry is None:
        return None
    if tenant_id and entry.get("tenant_id") not in ("", tenant_id):
        return None
    return entry
