"""The gateway/serve glibc arena cap (``runtime.malloc_arena_max``) bounds per-thread malloc arenas."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

# Each thread's first malloc takes a NEW arena while glibc is under its limit, so N live threads
# allocating at once (ctypes releases the GIL) cost ~N arenas uncapped. Arena heaps are the
# 64 MiB-aligned anonymous mappings in /proc/self/maps.
_CHILD = textwrap.dedent("""
    import ctypes, json, sys, threading
    from hermes_cli.resource_limits import apply_malloc_arena_max
    applied = apply_malloc_arena_max({"runtime": {"malloc_arena_max": int(sys.argv[1])}})
    libc = ctypes.CDLL(None)
    libc.malloc.restype = ctypes.c_void_p
    libc.free.argtypes = [ctypes.c_void_p]
    gate, done = threading.Barrier(25), threading.Event()
    def work():
        ptr = libc.malloc(4096)
        gate.wait()
        done.wait()
        libc.free(ptr)
    threads = [threading.Thread(target=work) for _ in range(24)]
    [t.start() for t in threads]
    gate.wait()
    heaps = 0
    for line in open("/proc/self/maps"):
        fields = line.split()
        lo, hi = (int(x, 16) for x in fields[0].split("-"))
        if len(fields) == 5 and lo % (64 << 20) == 0 and hi - lo <= 64 << 20:
            heaps += 1
    done.set()
    [t.join() for t in threads]
    print(json.dumps({"applied": applied, "heaps": heaps}))
""")


def _run_child(cap: int, **env_overrides: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in {"MALLOC_ARENA_MAX", "GLIBC_TUNABLES"}}
    env.update(env_overrides, PYTHONPATH=str(_REPO))
    out = subprocess.run([sys.executable, "-c", _CHILD, str(cap)], env=env, capture_output=True,
                         text=True, encoding="utf-8", timeout=60, stdin=subprocess.DEVNULL, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.platforms("linux")
def test_arena_cap_bounds_per_thread_arenas_and_zero_keeps_glibc_default():
    capped = _run_child(2)
    uncapped = _run_child(0)

    assert capped["applied"] is True
    assert capped["heaps"] <= 2, capped
    assert uncapped["applied"] is False
    assert uncapped["heaps"] > 4, uncapped  # control: the probe really sees per-thread arenas


@pytest.mark.platforms("linux")
def test_operator_malloc_arena_max_env_wins_over_config():
    result = _run_child(2, MALLOC_ARENA_MAX="16")

    assert result["applied"] is False
    assert result["heaps"] > 2, result  # the config's cap of 2 was not applied over the operator's 16
