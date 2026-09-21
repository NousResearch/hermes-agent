"""Atomic serialization for validated model-performance route tables."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.model_performance_router import CompiledRouteTable, compile_profile_routes


def _serialized(
    compiled: CompiledRouteTable, *, artifact_digest: str, policy_digest: str
) -> dict[str, Any]:
    return {
        "version": 1,
        "artifact_digest": artifact_digest,
        "policy_digest": policy_digest,
        "profiles": {
            profile: {surface: row.as_dict() for surface, row in sorted(rows.items())}
            for profile, rows in sorted(compiled.items())
        },
    }


def _write_atomic(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def compile_profile_configs(
    profiles: Sequence[str],
    artifact: Mapping[str, Any],
    *,
    destination: Path | None = None,
) -> CompiledRouteTable:
    """Validate the complete matrix before optionally replacing its config file."""

    compiled = compile_profile_routes(profiles, artifact)
    if destination is not None:
        payload = _serialized(
            compiled,
            artifact_digest=str(artifact["artifact_digest"]),
            policy_digest=str(artifact["policy_digest"]),
        )
        _write_atomic(destination, payload)
    return compiled
