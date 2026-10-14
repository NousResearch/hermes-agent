"""Interpreter hygiene after an update (PM lifecycle, failure class 4).

A PM install runs every Hermes process on PM's bundled interpreter with the selected
generation's site-packages. Two kinds of stray environment sit next to that on real machines:

* the pre-PM in-tree ``venv/`` a main-era install leaves behind after migrating (#123965), and
* a repo-local ``.venv`` a user or a dev tool created with another Python (3.11 here; #123972,
  whose Windows crash is ``pydantic_core`` built for the wrong ABI; this is the Linux analogue).

Both are seeded as real ``uv venv`` environments whose site-packages carry a ``sitecustomize``
and a ``.pth`` hook that record every interpreter that ever puts them on ``sys.path``. After a
dependency-changing ``hermes update``:

* the gateway (``hermes gateway run``) boots, and no process in its tree ever loaded, mapped or
  put either stray venv on its path;
* the workers a Hermes process spawns after the update use the PM interpreter and import their
  deps: ``execute_code`` (#124049) and a Kanban worker spawned by ``hermes kanban dispatch``
  (#124542, #122500), each proven by what reaches the loopback provider.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.pm import _pm as P
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

STRAY = ("venv", ".venv")
HOOK = """import os, sys
try:
    with open({log!r}, "a", encoding="utf-8") as fh:
        fh.write(repr((sys.executable, sys.argv[:3])) + "\\n")
except OSError:
    pass
"""


def _stray_python() -> str:
    """Another Python than PM's: 3.11 when uv can find one (the #123972 shape), else the test's."""
    uv = I.real_uv()
    assert uv is not None
    cp = subprocess.run([uv, "python", "find", "3.11"], capture_output=True, text=True, timeout=120, check=False)
    return cp.stdout.strip() if cp.returncode == 0 and cp.stdout.strip() else sys.executable


def _seed_stray_venvs(sb: I.Sandbox) -> dict[str, Path]:
    uv = I.real_uv()
    assert uv is not None
    python, logs = _stray_python(), {}
    for name in STRAY:
        venv = sb.checkout / name
        shutil.rmtree(venv, ignore_errors=True)
        cp = subprocess.run([uv, "venv", "-q", "--python", python, str(venv)], capture_output=True, text=True,
                            timeout=300, env={k: v for k, v in __import__("os").environ.items() if k != "VIRTUAL_ENV"}, check=False)
        assert cp.returncode == 0, f"harness: uv venv {venv} failed:\n{cp.stderr}"
        site = next(venv.glob("lib/python3*/site-packages"))
        log = sb.root / f"stray-{name.strip('.')}-loaded.log"
        (site / "sitecustomize.py").write_text(HOOK.format(log=str(log)), encoding="utf-8")
        (site / "e2e_stray_hook.py").write_text(HOOK.format(log=str(log)), encoding="utf-8")
        (site / "e2e_stray.pth").write_text("import e2e_stray_hook\n", encoding="utf-8")
        logs[name] = log
    return logs


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the pm hygiene suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def updated(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("pm-hygiene")
    sb, origin = P.install_head(root)
    P.configure(sb, provider.base_url)
    logs = _seed_stray_venvs(sb)
    target = P.publish_dependency_release(origin, root, 1)
    up = P.update(sb, env=P.lazy_env(sb))
    P.ok(up, "hermes update failed")
    assert I.git("rev-parse", "HEAD", cwd=sb.checkout) == target
    return {"sb": sb, "logs": logs}


def _marks(logs: dict[str, Path]) -> dict[str, int]:
    return {name: log.stat().st_size if log.exists() else 0 for name, log in logs.items()}


def _stray_hits(logs: dict[str, Path], since: dict[str, int] | None = None) -> dict[str, str]:
    """Hook records written after ``since`` (per cell, so one cell's red never bleeds into the next)."""
    since = since or {}
    hits = {name: log.read_bytes()[since.get(name, 0):].decode("utf-8", "replace")
            for name, log in logs.items() if log.exists()}
    return {name: text for name, text in hits.items() if text}


def test_gateway_after_update_never_touches_a_stray_venv(updated):
    sb, logs = updated["sb"], updated["logs"]
    mark = _marks(logs)
    gw = P.Gateway(sb, P.lazy_env(sb), sb.root / "hygiene-gateway.log")
    seen: dict[int, tuple[str, list[str], dict[str, str]]] = {}
    try:
        gw.wait_running()
        deadline = time.monotonic() + 8  # supervised workers start right after "running"
        while time.monotonic() < deadline:
            for pid in gw.pids():
                info = P.process_files(pid)
                if info[0]:
                    seen[pid] = info
            time.sleep(0.5)
    finally:
        gw.stop()
    strays = [str(sb.checkout / n) + "/" for n in STRAY]
    pythons = {pid: exe for pid, (exe, _, _) in seen.items() if "python" in Path(exe).name}
    assert pythons, f"harness: saw no Python process in the gateway tree: {seen.keys()}\n{gw.output()[-3000:]}"
    mapped = {pid: [m for m in maps if any(m.startswith(s) for s in strays)] for pid, (_, maps, _) in seen.items()}
    mapped = {pid: m for pid, m in mapped.items() if m}
    env_refs = {pid: {k: v for k, v in env.items() if k in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")
                      and any(s.rstrip("/") in v for s in strays)} for pid, (_, _, env) in seen.items()}
    env_refs = {pid: e for pid, e in env_refs.items() if e}
    stray_exes = {pid: exe for pid, exe in pythons.items() if any(exe.startswith(s) for s in strays)}
    hits = _stray_hits(logs, mark)
    assert not (hits or mapped or env_refs or stray_exes), (
        "the gateway put a stray in-tree venv on an interpreter's path after `hermes update`:\n"
        f"hooks fired: {hits}\nmapped files: {mapped}\nenvironment: {env_refs}\nexecutables: {stray_exes}\n"
        f"--- gateway output ---\n{gw.output()[-4000:]}\n" + P.diagnostics(sb))


def _turn_with_tool(sb: I.Sandbox, provider: FakeLLMServer, prompt: str, call: ToolCall) -> tuple[str, str]:
    """One ``hermes -z`` turn where the model calls ``call``; returns (tool result, transcript)."""
    n = len(provider.main_requests())
    provider.push(call, Text("done"))
    cp = P.run_env(sb, [sb.hermes, "-z", prompt], P.lazy_env(sb), timeout=600)
    reqs = provider.main_requests()[n:]
    tool_msgs = [m for r in reqs for m in r["messages"] if m.get("role") == "tool"]
    content = tool_msgs[-1]["content"] if tool_msgs else ""
    if not isinstance(content, str):
        content = json.dumps(content)
    return content, I.describe(cp) + f"\nprovider requests: {len(reqs)}"


def test_execute_code_after_update_imports_third_party_deps(updated, provider):
    sb, mark = updated["sb"], _marks(updated["logs"])
    code = "import ruamel.yaml, httpx, openai, pydantic\nprint('EXEC-DEPS-OK', pydantic.VERSION)"
    result, transcript = _turn_with_tool(sb, provider, "run the dependency probe",
                                         ToolCall("execute_code", {"code": code}))
    assert result, "harness: the execute_code tool never returned a result to the model\n" + transcript
    assert "EXEC-DEPS-OK" in result, (
        f"execute_code on a PM install cannot import third-party deps: {result[-1500:]}\n" + transcript)
    hits = _stray_hits(updated["logs"], mark)
    assert not hits, f"execute_code loaded a stray venv: {hits}"


def test_kanban_worker_spawned_after_update_boots(updated, provider):
    sb, mark = updated["sb"], _marks(updated["logs"])
    n = len(provider.main_requests())
    log = sb.root / "kanban-dispatch.log"
    # One sandbox for dispatcher + worker (the worker outlives `kanban dispatch`, as on a real host);
    # the sandbox stays up until the worker's first model call is observed, then is torn down.
    script = (f'H="{sb.hermes}"\n"$H" kanban init >/dev/null\n'
              '"$H" kanban create "pm hygiene kanban probe" --assignee default --json\n'
              '"$H" kanban dispatch --json\nsleep 600\n')
    with log.open("w") as out:
        proc = subprocess.Popen(H.sandbox_argv(["/bin/sh", "-c", script], writable=[sb.root]), env=P.lazy_env(sb),
                                cwd=str(sb.root), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        try:
            deadline, task_id, reached = time.monotonic() + 300, None, []
            while time.monotonic() < deadline and not reached:
                text = log.read_text(errors="replace")
                ids = re.findall(r'"id": "(t_[0-9a-f]+)"', text)
                task_id = ids[0] if ids else task_id
                if task_id:
                    reached = [r for r in provider.main_requests()[n:] if task_id in json.dumps(r["messages"])]
                time.sleep(0.5)
        finally:
            H.kill_tree(proc)
    worker_log = sb.hermes_home / "kanban" / "logs" / f"{task_id}.log"
    detail = (f"--- dispatcher ---\n{log.read_text(errors='replace')[-3000:]}\n--- worker log ---\n"
              + (worker_log.read_text(errors="replace")[-3000:] if task_id and worker_log.exists() else "(none)"))
    assert task_id, "harness: `hermes kanban create` printed no task id\n" + detail
    assert reached, (
        f"the Kanban worker for {task_id} spawned after `hermes update` never reached the provider "
        "(it died at spawn)\n" + detail)
    hits = _stray_hits(updated["logs"], mark)
    assert not hits, f"the worker loaded a stray venv: {hits}"
