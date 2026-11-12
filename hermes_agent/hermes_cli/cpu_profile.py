"""``hermes --cpu-profile PATH``: process-wide cProfile, written when the process exits.

Enabled once after argument parsing and written from ``atexit``, so short commands and
long-running ones (``gateway run``) are covered without wrapping every handler. The one-shot
(``-z``) exit skips ``atexit`` on purpose (``main._exit_after_oneshot``) and calls
:func:`write_cpu_profile` itself. A relative PATH lands in the active profile's
``logs/profiles/``. Profiles cover THIS process only: under ``--tui`` that is the launcher,
while the agent runs in the TUI's own backend process.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_profiler = None
_profile_path: Path | None = None


def resolve_profile_path(path: str | os.PathLike[str]) -> Path:
    """Absolute (or ``~``) paths as given; relative ones under ``<hermes home>/logs/profiles``."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        from hermes_agent.hermes_constants import get_hermes_home

        target = get_hermes_home() / "logs" / "profiles" / target
    return target.resolve()


def enable_cpu_profile(path: str | os.PathLike[str] | None) -> Path | None:
    """Start profiling to ``path``; idempotent so the fast-path → full-parser handoff can't
    register a second writer. Returns the resolved path, or None when ``path`` is empty."""
    global _profiler, _profile_path
    if not path:
        return None
    if _profiler is not None:
        return _profile_path
    import cProfile

    target = resolve_profile_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _profiler, _profile_path = cProfile.Profile(), target
    _profiler.enable()
    atexit.register(write_cpu_profile)
    return target


def enable_cpu_profile_from_args(args) -> None:
    """Honour ``--cpu-profile`` on a parsed CLI namespace and say where the profile goes."""
    enabled = enable_cpu_profile(getattr(args, "cpu_profile", None))
    if enabled is not None:
        print(f"CPU profiling enabled: {enabled}", file=sys.stderr)


def write_cpu_profile() -> None:
    """Stop profiling and write ``PATH`` (pstats) plus ``PATH.txt`` (top 80 by cumulative time)."""
    global _profiler
    profiler, target = _profiler, _profile_path
    if profiler is None or target is None:
        return
    _profiler = None
    profiler.disable()
    import pstats

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(target))
        with target.with_suffix(target.suffix + ".txt").open("w", encoding="utf-8") as fh:
            pstats.Stats(profiler, stream=fh).sort_stats("cumulative").print_stats(80)
    except OSError:
        logger.exception("Failed to write CPU profile: %s", target)
