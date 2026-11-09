"""Refuse to run a production gateway from a disposable (scratch) runtime.

A scratch dir under ``profiles/<p>/cache/scratch`` is pruned when idle, so a daemon started from
its venv or checkout loses its code under itself and keeps running it (t_962fc9a1: four orphans).
``gateway run`` checks the interpreter, its prefix and the package that serves ``hermes_cli``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SCRATCH_MARKER = "/cache/scratch/"


def is_scratch_path(path: str | os.PathLike[str]) -> bool:
    resolved = os.path.realpath(os.fspath(path)).replace(os.sep, "/")
    return SCRATCH_MARKER in resolved + "/"


def runtime_candidates() -> list[str]:
    import hermes_cli

    return [sys.executable, sys.prefix, str(Path(hermes_cli.__file__).resolve().parent)]


def find_scratch_runtime(candidates: list[str]) -> str | None:
    return next((c for c in candidates if c and is_scratch_path(c)), None)


def enforce_production_runtime(
    allow_scratch: bool = False, candidates: list[str] | None = None
) -> None:
    """Exit non-zero when the runtime is scratch; ``allow_scratch`` (tests only) logs a WARNING."""
    offender = find_scratch_runtime(
        runtime_candidates() if candidates is None else candidates
    )
    if offender is None:
        return
    if allow_scratch:
        logger.warning(
            "running gateway from scratch dir %s because --allow-scratch was given",
            offender,
        )
        return
    print(
        f"refusing to run production gateway from scratch dir {offender}",
        file=sys.stderr,
    )
    raise SystemExit(1)
