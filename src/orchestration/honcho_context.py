"""Optional Honcho / group-model hints (profile-safe, no hard Honcho import)."""

from __future__ import annotations

import os
from typing import Optional

from hermes_constants import display_hermes_home, get_hermes_home


def group_session_label(orchestration_id: str, extra: Optional[str] = None) -> str:
    """Stable label for correlating swarm turns with long-term user memory."""

    base = orchestration_id.strip() or "orch"
    if extra and extra.strip():
        return f"{base}:{extra.strip()}"
    return base


def honcho_memory_preamble(*, orch_id: str, role: str = "coordinator") -> str:
    """Short prose block suitable for appending to delegate context."""

    home = display_hermes_home()
    return (
        f"[multi-agent orchestration] id={orch_id} role={role} "
        f"hermes_home={home} "
        "Use Honcho tools only when enabled; respect profile isolation."
    )


def peer_namespace_hint() -> str:
    """Echo active profile path for Honcho peer resolution."""

    return f"HONCHO_NAMESPACE_HINT={get_hermes_home()}"


def optional_honcho_env_patch() -> dict[str, str]:
    """Extra env keys downstream plugins may read (best-effort)."""

    patch = {"HERMES_ORCH_MARK": "1"}
    if os.getenv("HONCHO_APP_ID"):
        patch["HERMES_ORCH_HONCHO"] = "1"
    return patch
