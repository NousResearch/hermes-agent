"""Tests for agent.plan_admission and its transport wiring.

Behavioural process-level tests prove host-wide serialisation on the Z.ai
GLM Coding Plan route, fail-closed busy behaviour, same-process re-entry
reuse, kernel-released stale holders and untouched non-plan routes. Wiring
tests prove both transport choke points carry the gate. Invented data only;
no network, no real provider calls.
"""
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from agent import plan_admission as pa

PLAN_A = "https://api.z.ai/api/coding/paas/v4"
PLAN_B = "https://open.bigmodel.cn/api/coding/paas/v4"


@pytest.fixture(autouse=True)
def admission_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "_admission_dir", lambda: str(tmp_path / "admission"))
    yield str(tmp_path / "admission")


HOLDER = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {root!r})
    from agent import plan_admission as pa
    pa._admission_dir = lambda: {adir!r}
    with pa.plan_admission({base_url!r}, model="glm-5.3", purpose="test", wait_seconds={wait}):
        print("HELD", flush=True)
        time.sleep({hold})
    """
)
CONTENDER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})
    from agent import plan_admission as pa
    pa._admission_dir = lambda: {adir!r}
    try:
        with pa.plan_admission({base_url!r}, model="glm-5.3", purpose="test", wait_seconds={wait}):
            print("ACQUIRED")
    except pa.PlanAdmissionBusy:
        print("BUSY")
    """
)


def _run(script: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", script.format(**kw)], capture_output=True, text=True, timeout=60, check=False)


def _log_rows(dirname) -> list:
    import os
    lp = os.path.join(str(dirname), "log.jsonl")
    if not os.path.exists(lp):
        return []
    with open(lp, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_route_predicate_official_plan_hosts_only():
    assert pa.is_plan_route(PLAN_A)
    assert pa.is_plan_route(PLAN_B)
    assert pa.is_plan_route("api.z.ai/api/coding/paas/v4")  # schemeless
    assert not pa.is_plan_route("https://api.z.ai/api/paas/v4")  # wrong path
    assert not pa.is_plan_route("https://api.openai.com/v1")
    assert not pa.is_plan_route("https://open.bigmodel.cn/api/coding/other")
    assert not pa.is_plan_route(None)
    assert not pa.is_plan_route("")


def test_second_process_fails_closed_while_first_holds(admission_dir):
    root = str(Path(pa.__file__).resolve().parents[1])
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER.format(root=root, adir=admission_dir, base_url=PLAN_A, wait=5, hold=3)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "HELD"  # holder is inside the gate
        contender = _run(CONTENDER, root=root, adir=admission_dir, base_url=PLAN_A, wait=1)
        assert "BUSY" in contender.stdout and "ACQUIRED" not in contender.stdout
        rows = _log_rows(admission_dir)
        busy = [r for r in rows if r["event"] == "busy"]
        assert busy and busy[-1]["pid"] != holder.pid  # contender logged busy, never acquired
        assert not [r for r in rows if r["event"] == "acquire" and r["pid"] not in (holder.pid,)]
    finally:
        holder.wait(timeout=30)
    # after release a fresh process acquires
    again = _run(CONTENDER, root=root, adir=admission_dir, base_url=PLAN_A, wait=10)
    assert "ACQUIRED" in again.stdout


def test_same_process_reentry_reuses_outer_hold(admission_dir):
    with pa.plan_admission(PLAN_A, model="glm-5.3", purpose="outer"):
        with pa.plan_admission(PLAN_A, model="glm-5.3", purpose="inner"):
            rows_mid = _log_rows(admission_dir)
    rows = _log_rows(admission_dir)
    acquires = [r for r in rows if r["event"] == "acquire"]
    releases = [r for r in rows if r["event"] == "release"]
    assert len(acquires) == 1 and acquires[0]["purpose"] == "outer"
    assert len(releases) == 1 and releases[0]["purpose"] == "outer"
    assert not [r for r in rows_mid if r["event"] == "release"]


def test_non_plan_route_never_touches_the_lock(admission_dir):
    with pa.plan_admission("https://api.openai.com/v1", model="gpt-x", purpose="other"):
        pass
    import os
    assert not os.path.exists(os.path.join(str(admission_dir), "log.jsonl"))
    assert not os.path.exists(os.path.join(str(admission_dir), "zai-plan.lock"))


def test_kernel_releases_lock_when_holder_dies_hard(admission_dir):
    root = str(Path(pa.__file__).resolve().parents[1])
    die = textwrap.dedent(
        """
        import sys, os
        sys.path.insert(0, {root!r})
        from agent import plan_admission as pa
        pa._admission_dir = lambda: {adir!r}
        with pa.plan_admission({base_url!r}, purpose="dying", wait_seconds=5):
            print("HELD", flush=True)
            os._exit(9)
        """
    )
    dead = subprocess.run([sys.executable, "-c", die.format(root=root, adir=admission_dir, base_url=PLAN_A)],
                          capture_output=True, text=True, timeout=30, check=False)
    assert "HELD" in dead.stdout and dead.returncode == 9
    survivor = _run(CONTENDER, root=root, adir=admission_dir, base_url=PLAN_A, wait=10)
    assert "ACQUIRED" in survivor.stdout  # flock died with the process


def test_wiring_covers_both_transport_choke_points():
    import inspect

    import agent.chat_completion_helpers as cch

    nonstream = inspect.getsource(cch._dispatch_nonstreaming_api_request)
    assert "plan_admission" in nonstream, "non-streaming dispatch must hold plan admission"
    stream_wrap = inspect.getsource(cch._StreamingCall._call_chat_completions)
    assert "plan_admission" in stream_wrap, "streaming attempt must hold plan admission"
    assert hasattr(cch._StreamingCall, "_call_chat_completions_ungated")
