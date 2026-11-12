"""The gateway/serve glibc arena cap (``runtime.malloc_arena_max``) bounds per-thread malloc arenas."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_agent.hermes_cli.resource_limits import glibc_malloc

_REPO = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.skipif(glibc_malloc() is None, reason="arena caps are glibc malloc behaviour (not musl)"),
]

# Each thread's first malloc takes a NEW arena while glibc is under its limit, so N live threads
# allocating at once (ctypes releases the GIL) cost ~N arenas uncapped. malloc_info() lists one
# <heap> per arena, main arena included.
_WORKERS = textwrap.dedent("""
    import ctypes, threading
    def arenas_under_load():
        libc = ctypes.CDLL(None)
        libc.malloc.restype = ctypes.c_void_p
        libc.free.argtypes = [ctypes.c_void_p]
        libc.open_memstream.restype = ctypes.c_void_p
        libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
        libc.fclose.argtypes = [ctypes.c_void_p]
        gate, done = threading.Barrier(25), threading.Event()
        def work():
            ptr = libc.malloc(4096)
            gate.wait()
            done.wait()
            libc.free(ptr)
        threads = [threading.Thread(target=work) for _ in range(24)]
        [t.start() for t in threads]
        gate.wait()
        buf, size = ctypes.c_void_p(), ctypes.c_size_t()
        stream = libc.open_memstream(ctypes.byref(buf), ctypes.byref(size))
        libc.malloc_info(0, stream)
        libc.fclose(stream)
        xml = ctypes.string_at(buf, size.value).decode()
        libc.free(buf)
        done.set()
        [t.join() for t in threads]
        return xml.count("<heap nr=")
""")

# The real CLI entry point: importing hermes_agent.hermes_cli.main runs its startup (profile, .env, config) for
# the given argv; only then do worker threads start.
_ENTRY_CHILD = _WORKERS + textwrap.dedent("""
    import json, sys
    sys.argv = ["hermes", *json.loads(sys.argv[1])]
    import hermes_cli.main  # noqa: F401
    print(json.dumps({"heaps": arenas_under_load()}))
""")

_HELPER_CHILD = _WORKERS + textwrap.dedent("""
    import json, sys
    from hermes_cli.resource_limits import apply_malloc_arena_max
    applied = apply_malloc_arena_max({"runtime": {"malloc_arena_max": int(sys.argv[1])}})
    print(json.dumps({"applied": applied, "heaps": arenas_under_load()}))
""")


def _run(code: str, arg: str, home: Path, **env_overrides: str) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in {"MALLOC_ARENA_MAX", "GLIBC_TUNABLES"} and not k.startswith("HERMES_")}
    env.update(HOME=str(home), HERMES_HOME=str(home / ".hermes"), PYTHONPATH=str(_REPO), **env_overrides)
    out = subprocess.run([sys.executable, "-c", code, arg], check=False, env=env, cwd=str(home),
                         capture_output=True, text=True, encoding="utf-8", timeout=120,
                         stdin=subprocess.DEVNULL)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _home(tmp_path: Path, config: str = "", dotenv: str = "") -> Path:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    if config:
        (hermes_home / "config.yaml").write_text(config, encoding="utf-8")
    (hermes_home / ".env").write_text(dotenv, encoding="utf-8")
    return tmp_path


def test_serve_startup_caps_arenas_before_glibc_latches_its_limit(tmp_path):
    # arena_test=1 makes glibc latch its limit at the second arena, so only a cap set before any
    # thread allocates can hold.
    home = _home(tmp_path, config="runtime:\n  malloc_arena_max: 2\n")
    result = _run(_ENTRY_CHILD, json.dumps(["serve"]), home, GLIBC_TUNABLES="glibc.malloc.arena_test=1")

    assert result["heaps"] <= 2, result


def test_gateway_startup_applies_malloc_arena_max_loaded_from_profile_env(tmp_path):
    # glibc reads MALLOC_ARENA_MAX only at exec; the profile .env is loaded after that.
    home = _home(tmp_path, config="runtime:\n  malloc_arena_max: 0\n", dotenv="MALLOC_ARENA_MAX=2\n")
    result = _run(_ENTRY_CHILD, json.dumps(["gateway", "run"]), home)

    assert result["heaps"] <= 2, result


def test_non_server_commands_keep_glibc_default_arenas(tmp_path):
    home = _home(tmp_path, config="runtime:\n  malloc_arena_max: 2\n")
    result = _run(_ENTRY_CHILD, json.dumps(["sessions", "list"]), home)

    assert result["heaps"] > 4, result  # control: the probe really sees per-thread arenas


def test_exported_malloc_arena_max_wins_over_config(tmp_path):
    result = _run(_HELPER_CHILD, "2", _home(tmp_path), MALLOC_ARENA_MAX="16")

    assert result["applied"] is False
    assert result["heaps"] > 2, result  # glibc honours the operator's 16, not the config's 2
