from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


class MissingReadOnlyCredential(RuntimeError):
    pass


def scheduled_credential(environment: Mapping[str, str], name: str, cache_path: Path) -> str:
    """Read a pre-provisioned secret without refreshing, caching, or writing it."""
    del cache_path
    value = environment.get(name, "").strip()
    if not value:
        raise MissingReadOnlyCredential(f"missing pre-provisioned read-only credential: {name}")
    return value
