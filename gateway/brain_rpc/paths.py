"""Vault path normalization and resolution (contract §4.3–4.5)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from gateway.brain_rpc.errors import FORBIDDEN, INVALID_ARGUMENT, BrainRpcError


def normalize_vault_path(raw: Any) -> str:
    """Normalize a vault-root-relative POSIX path.

    Rules:
      - Must be a non-empty string starting with ``/``
      - Reject ``..`` segments and null bytes
      - Collapse ``.`` and duplicate slashes
      - Result is absolute-looking (starts with /) but relative to vault root
    """
    if raw is None or not isinstance(raw, str) or not raw.strip():
        raise BrainRpcError(INVALID_ARGUMENT, "path is required", details={"path": raw})
    s = raw.strip().replace("\\", "/")
    if "\x00" in s:
        raise BrainRpcError(INVALID_ARGUMENT, "invalid path", details={"path": raw})
    if not s.startswith("/"):
        raise BrainRpcError(
            INVALID_ARGUMENT,
            "path must start with /",
            details={"path": raw},
        )
    parts: list[str] = []
    for seg in s.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise BrainRpcError(
                FORBIDDEN,
                "path escape rejected",
                details={"path": raw},
            )
        parts.append(seg)
    return "/" + "/".join(parts) if parts else "/"


def resolve_under_vault(vault_root: Path, vault_path: str) -> Tuple[Path, str]:
    """Map a normalized vault path to a real filesystem path under vault_root.

    Returns ``(fs_path, normalized_vault_path)``. Raises on breakout.
    """
    norm = normalize_vault_path(vault_path)
    root = vault_root.resolve()
    # Build relative path under root (strip leading /)
    rel = norm.lstrip("/")
    candidate = (root / rel).resolve() if rel else root
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BrainRpcError(
            FORBIDDEN,
            "path escape rejected",
            details={"path": norm},
        ) from exc
    # Extra defense: ensure string prefix match with sep boundary
    root_s = str(root)
    cand_s = str(candidate)
    if cand_s != root_s and not cand_s.startswith(root_s + os.sep):
        raise BrainRpcError(
            FORBIDDEN,
            "path escape rejected",
            details={"path": norm},
        )
    return candidate, norm
