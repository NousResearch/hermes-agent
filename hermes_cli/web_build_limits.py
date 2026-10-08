"""Web UI build resource caps (issue #63338).

``npm run build`` in ``web/`` runs the TypeScript solution builder plus Vite 8's
Rust-native bundler (Rolldown). On small hosts (2–4 vCPU VPS, ~4–10 GB RAM) that
combination saturates every CPU (200%+ ``top`` readings) and can OOM the box or
freeze SSH sessions for the ~20 s build window. Rolldown does not expose a JS-level
worker/thread option, but its native binding parallelizes with rayon, which honours
``RAYON_NUM_THREADS`` — and V8's heap is capped via ``--max-old-space-size``.

The defaults are deliberately conservative: bounded to half the available cores so
one build can never monopolize a small host, and a heap ceiling sized from the
container/cgroup limit (75% of it, same shape as the TUI launcher's sizing) so a
constrained container gets a graceful V8 allocation failure instead of a silent
cgroup OOM kill.

User overrides always win: an explicit ``NODE_OPTIONS`` ``--max-old-space-size`` or
``RAYON_NUM_THREADS`` in the environment is left untouched, and
``HERMES_WEB_BUILD_MAX_OLD_SPACE_SIZE`` / ``HERMES_WEB_BUILD_THREADS`` allow
tailoring without code changes.
"""

from __future__ import annotations

import os

# Never let the heap ceiling drop below this on constrained-but-usable hosts:
# below ~1 GB V8 spends the build GC-thrashing instead of making progress.
_MIN_WEB_BUILD_HEAP_MB = 1024
# Ceiling even on huge machines; the dashboard bundle is small and a larger cap
# only inflates the resident set (and OOM risk on shared VPS hosts).
_MAX_WEB_BUILD_HEAP_MB = 4096
# Likewise for parallelism: rolldown caps its own JS-plugin workers at 8, and
# beyond ~half-a-dozen threads the small dashboard graph doesn't speed up.
_MAX_WEB_BUILD_THREADS = 8

_TRUTHY = {"1", "true", "yes", "on"}


def _cgroup_memory_limit_mb() -> int | None:
    """Return the cgroup memory limit in MB, or None when unconstrained."""
    limit = None
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= (1 << 50):  # the v1 "unlimited" sentinel
            return None
        limit = value
        break
    if limit is None:
        return None
    return limit // (1024 * 1024)


def _available_cores() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _bounded_threads() -> int:
    """Half the cores, clamped to [1, _MAX_WEB_BUILD_THREADS]."""
    return max(1, min(_MAX_WEB_BUILD_THREADS, _available_cores() // 2 or 1))


def _bounded_heap_mb() -> int:
    """Heap ceiling from the cgroup limit, clamped to [_MIN, _MAX]."""
    limit_mb = _cgroup_memory_limit_mb()
    if not limit_mb:
        return _MAX_WEB_BUILD_HEAP_MB
    return max(_MIN_WEB_BUILD_HEAP_MB, min(_MAX_WEB_BUILD_HEAP_MB, int(limit_mb * 0.75)))


def _env_flag(env: dict[str, str], name: str) -> bool:
    return str(env.get(name, "")).strip().lower() in _TRUTHY


def web_build_limits(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of *env* bounding the web dashboard build's CPU and heap (#63338).

    Never overrides the user: an existing ``NODE_OPTIONS`` containing
    ``--max-old-space-size`` or an existing ``RAYON_NUM_THREADS`` is preserved.
    ``HERMES_WEB_BUILD_LIGHT=1`` tightens the caps (1 thread, 1 GB heap) for hosts
    that cannot spare full CPU during the build.
    """
    env = dict(env if env is not None else os.environ)
    light = light_build_requested(env)

    # --- V8 heap cap ---------------------------------------------------------
    tokens = env.get("NODE_OPTIONS", "").split()
    if not any(t.startswith("--max-old-space-size=") for t in tokens):
        explicit = (env.get("HERMES_WEB_BUILD_MAX_OLD_SPACE_SIZE") or "").strip()
        try:
            heap_mb = int(explicit) if explicit else (
                _MIN_WEB_BUILD_HEAP_MB if light else _bounded_heap_mb()
            )
        except ValueError:
            heap_mb = _MIN_WEB_BUILD_HEAP_MB if light else _bounded_heap_mb()
        heap_mb = max(_MIN_WEB_BUILD_HEAP_MB, heap_mb)
        tokens.append(f"--max-old-space-size={heap_mb}")
        env["NODE_OPTIONS"] = " ".join(t for t in tokens if t).strip()

    # --- Native bundler parallelism cap -------------------------------------
    if not env.get("RAYON_NUM_THREADS"):
        explicit = (env.get("HERMES_WEB_BUILD_THREADS") or "").strip()
        try:
            threads = int(explicit) if explicit else (1 if light else _bounded_threads())
        except ValueError:
            threads = 1 if light else _bounded_threads()
        env["RAYON_NUM_THREADS"] = str(max(1, threads))

    return env


def apply_web_build_limits(env: dict[str, str]) -> dict[str, str]:
    """Apply :func:`web_build_limits` into *env* (in place) and return it."""
    env.update(web_build_limits(env))
    return env


def light_build_requested(env: dict[str, str] | None = None) -> bool:
    """True when the caller asked for the minimal-footprint web build.

    ``HERMES_WEB_BUILD_LIGHT=1``: caps tighten to a single bundler thread and a
    1 GB heap — for VPS hosts that cannot spare full CPU during the build
    (the closed PR #78131's ``build:light`` idea).
    """
    return _env_flag(env if env is not None else dict(os.environ), "HERMES_WEB_BUILD_LIGHT")
