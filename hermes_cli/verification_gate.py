"""Deterministic completion verification used at the Kanban done boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CARD_TYPE_ARTIFACTS = {
    "pr-merged": "metadata.verification.artifact with type=pr-merged, url, merged=true, and merge_commit",
    "deployed": "metadata.verification.artifact with type=deployed, url, and a successful smoke_test",
    "document-committed": "metadata.verification.artifact with type=document-committed, path, and commit",
    "decision-recorded": "metadata.verification.artifact with type=decision-recorded, path, and decision",
}

_REQUIRED_FIELDS = {
    "pr-merged": ("url", "merged", "merge_commit"),
    "deployed": ("url", "smoke_test"),
    "document-committed": ("path", "commit"),
    "decision-recorded": ("path", "decision"),
}


def _failure(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def verify_completion(payload: Any, *, card_type: str) -> dict[str, Any]:
    """Return a stable, side-effect-free report on the artifact a typed contract requires."""
    failures: list[dict[str, str]] = []
    artifact = _mapping(_mapping(payload).get("artifact"))
    artifact_type = artifact.get("type") == card_type
    if not artifact_type:
        failures.append(_failure("missing_artifact", f"required artifact: {CARD_TYPE_ARTIFACTS[card_type]}"))
    else:
        missing = [key for key in _REQUIRED_FIELDS[card_type] if not artifact.get(key)]
        smoke = _mapping(artifact.get("smoke_test"))
        if card_type == "deployed" and (smoke.get("exit_code") != 0 or not smoke.get("command")):
            missing.append("smoke_test.exit_code=0")
        if missing:
            failures.append(_failure("incomplete_artifact", "artifact is missing: " + ", ".join(missing)))
    return {"ok": not failures, "failures": failures, "checks": {"required_artifact": artifact_type}}
