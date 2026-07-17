"""Tests for the shared health-check cache/dedup (Gate-5 contention fix).

Run with bare python3 (no project venv / no gateway import required):

    python3 tests/gateway/test_health_check_cache.py

Exercises the real dedup contract: concurrent/excessive calls within the TTL
must NOT re-run the expensive compute; a fresh cache (or expired TTL) must.
"""

import importlib.util
import json
import tempfile
from pathlib import Path

REPO = Path("/Users/jon/.hermes/hermes-agent")
MODULE = REPO / "scripts" / "health_check_cache.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "health_check_cache_test", str(MODULE)
    )
    assert spec is not None and spec.loader is not None, "spec resolution failed"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run():
    mod = _load()
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn(mod)
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL", name, "->", repr(exc))
    return 1 if failures else 0


def test_dedup_within_ttl(mod):
    calls = {"n": 0}

    def work():
        calls["n"] += 1
        return {"pass": True, "count": calls["n"]}

    with tempfile.TemporaryDirectory() as td:
        r1, c1 = mod.compute_with_cache("d1", work, ttl=60, cache_dir=td)
        r2, c2 = mod.compute_with_cache("d1", work, ttl=60, cache_dir=td)
        assert c1 is False, "first call should compute"
        assert c2 is True, "second call should serve cache"
        assert r1 == r2, "cached result must equal computed result"
        assert calls["n"] == 1, f"compute ran {calls['n']}x, expected 1"


def test_ttl_expiry_recomputes(mod):
    calls = {"n": 0}

    def work():
        calls["n"] += 1
        return {"pass": True, "count": calls["n"]}

    with tempfile.TemporaryDirectory() as td:
        mod.compute_with_cache("d2", work, ttl=1, cache_dir=td)
        import time as _t

        _t.sleep(1.2)
        r, cached = mod.compute_with_cache("d2", work, ttl=1, cache_dir=td)
        assert cached is False, "expired TTL should recompute"
        assert calls["n"] == 2, f"expected 2 computes after expiry, got {calls['n']}"


def test_concurrent_processes_share_cache(mod):
    # Two separate python processes hitting the same cache file must dedupe.
    import subprocess

    with tempfile.TemporaryDirectory() as td:
        count_file = Path(td) / "counts.txt"
        child_src = (
            "import sys\n"
            "sys.path.insert(0, '%s')\n"
            "from health_check_cache import compute_with_cache\n"
            "calls = {'n': 0}\n"
            "def w():\n"
            "    calls['n'] += 1\n"
            "    return {'pass': True, 'c': calls['n']}\n"
            "compute_with_cache('d3', w, ttl=60, cache_dir='%s')\n"
            "open('%s', 'a').write(str(calls['n']) + '\\n')\n"
        ) % (str(REPO / "scripts"), td, count_file)
        child_py = Path(td) / "child.py"
        child_py.write_text(child_src)
        p1 = subprocess.Popen(["/usr/bin/python3", str(child_py)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(["/usr/bin/python3", str(child_py)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        o1, e1 = p1.communicate(); o2, e2 = p2.communicate()
        if e1.strip():
            print("child1 stderr:", e1.decode()[:300])
        if e2.strip():
            print("child2 stderr:", e2.decode()[:300])
        nums = [int(x) for x in count_file.read_text().split() if x.strip()]
        total = sum(nums)
        # Exactly one compute across both processes (the other hit the cache).
        assert total == 1, f"total computes across 2 procs = {total}, expected 1; raw={nums}"


def test_stale_cache_ignored(mod):
    with tempfile.TemporaryDirectory() as td:
        cp = mod._cache_path("d4", td)
        cp.write_text("{ not valid json", encoding="utf-8")
        calls = {"n": 0}

        def work():
            calls["n"] += 1
            return {"pass": True}

        r, cached = mod.compute_with_cache("d4", work, ttl=60, cache_dir=td)
        assert cached is False, "corrupt cache must be ignored (recompute)"
        assert calls["n"] == 1


def test_read_cached_off_path(mod):
    # read_cached returns fresh value WITHOUT computing; cold cache -> None.
    calls = {"n": 0}

    def work():
        calls["n"] += 1
        return {"pass": True, "c": calls["n"]}

    with tempfile.TemporaryDirectory() as td:
        assert mod.read_cached("d5", ttl=60, cache_dir=td) is None, "cold cache must be None"
        # populate via compute_with_cache
        mod.compute_with_cache("d5", work, ttl=60, cache_dir=td)
        val = mod.read_cached("d5", ttl=60, cache_dir=td)
        assert val is not None and val.get("pass") is True, "fresh read must return value"
        assert calls["n"] == 1, "read_cached must NOT compute"


if __name__ == "__main__":
    raise SystemExit(_run())
