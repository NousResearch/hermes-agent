"""One-shot exit waits (bounded) for the in-flight background review.

The review runs on a daemon ``bg-review`` thread. A one-shot run (``hermes chat -q``/``-Q``,
every Kanban worker) used to exit right after its turn, so interpreter exit killed the fork
mid-request: no memory/skill writes and neither a "complete" nor a "failed" log line.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from agent import background_review as br

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Agent:
    def __init__(self, thread=None):
        self._background_review_thread = thread


def _thread(target):
    t = threading.Thread(target=target, daemon=True, name="bg-review")
    t.start()
    return t


def test_no_review_thread_is_noop():
    assert br.wait_for_background_review(_Agent(None), timeout_s=5) == "none"
    assert br.wait_for_background_review(object(), timeout_s=5) == "none"


def test_finished_review_thread_is_noop():
    t = _thread(lambda: None)
    t.join()
    assert br.wait_for_background_review(_Agent(t), timeout_s=5) == "none"


def test_waits_for_in_flight_review_to_finish():
    finished = threading.Event()

    def _review():
        time.sleep(0.3)
        finished.set()

    assert br.wait_for_background_review(_Agent(_thread(_review)), timeout_s=10) == "done"
    assert finished.is_set()


def test_wait_is_bounded():
    release = threading.Event()
    t = _thread(lambda: release.wait(10))
    started = time.monotonic()
    try:
        assert br.wait_for_background_review(_Agent(t), timeout_s=0.2) == "timeout"
        assert time.monotonic() - started < 5
    finally:
        release.set()


def test_non_positive_timeout_skips_the_wait():
    release = threading.Event()
    t = _thread(lambda: release.wait(10))
    try:
        assert br.wait_for_background_review(_Agent(t), timeout_s=0) == "disabled"
        assert t.is_alive()
    finally:
        release.set()


@pytest.mark.parametrize("cfg, expected", [
    ({}, br._EXIT_WAIT_DEFAULT_S),
    (None, br._EXIT_WAIT_DEFAULT_S),
    ({"exit_wait_s": 7}, 7.0),
    ({"exit_wait_s": "3.5"}, 3.5),
    ({"exit_wait_s": 0}, 0.0),
    ({"exit_wait_s": "junk"}, br._EXIT_WAIT_DEFAULT_S),
])
def test_exit_wait_config_reader(cfg, expected):
    assert br.background_review_exit_wait_s(cfg) == expected


def test_default_config_carries_exit_wait_key():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    block = DEFAULT_CONFIG["auxiliary"]["background_review"]
    assert br.background_review_exit_wait_s(block) == float(block["exit_wait_s"])


def test_spawn_records_the_review_thread(monkeypatch):
    """The real spawn path stores the started thread where the exit wait reads it."""
    from run_agent import AIAgent

    release = threading.Event()
    monkeypatch.setattr(br, "spawn_background_review_thread",
                        lambda *a, **k: (lambda: release.wait(10), "prompt"))
    agent = object.__new__(AIAgent)
    agent._background_review_run = None
    agent._background_review_lock = threading.Lock()
    agent._background_review_thread = None
    agent._spawn_background_review_now(messages_snapshot=[], review_memory=True)
    thread = agent._background_review_thread
    try:
        assert isinstance(thread, threading.Thread) and thread.name == "bg-review" and thread.daemon
        assert thread.is_alive()
    finally:
        release.set()
    assert br.wait_for_background_review(agent, timeout_s=10) == "done"


def test_finalize_waits_after_flush_and_before_cleanup(monkeypatch):
    import cli as cli_mod

    order = []
    monkeypatch.setattr(cli_mod, "_wait_for_oneshot_background_completions", lambda c: order.append("linger"))
    monkeypatch.setattr(cli_mod, "_flush_one_shot_session_store", lambda c: order.append("flush"))
    monkeypatch.setattr(cli_mod, "_wait_for_oneshot_background_review", lambda c: order.append("review"))
    monkeypatch.setattr(cli_mod, "_notify_single_query_session_finalize", lambda c, **k: order.append("finalize"))
    monkeypatch.setattr(cli_mod, "_run_cleanup", lambda **k: order.append("cleanup"))

    class _FakeCli:
        agent = None
        session_id = "s1"

        def _release_active_session(self):
            order.append("release")

    cli_mod._finalize_single_query(_FakeCli())
    assert order == ["linger", "flush", "review", "finalize", "cleanup", "release"]


def test_finalize_survives_review_wait_failure(monkeypatch):
    import cli as cli_mod
    import agent.background_review as br_mod

    order = []

    def _boom(*a, **k):
        raise RuntimeError("wait exploded")

    monkeypatch.setattr(br_mod, "wait_for_background_review", _boom)
    monkeypatch.setattr(cli_mod, "_wait_for_oneshot_background_completions", lambda c: None)
    monkeypatch.setattr(cli_mod, "_flush_one_shot_session_store", lambda c: order.append("flush"))
    monkeypatch.setattr(cli_mod, "_notify_single_query_session_finalize", lambda c, **k: order.append("finalize"))
    monkeypatch.setattr(cli_mod, "_run_cleanup", lambda **k: order.append("cleanup"))

    class _FakeCli:
        agent = object()
        session_id = "s1"

        def _release_active_session(self):
            order.append("release")

    cli_mod._finalize_single_query(_FakeCli())
    assert order == ["flush", "finalize", "cleanup", "release"]


# ── real-process E2E: the daemon review survives a one-shot exit only with the wait ──

_E2E_CHILD = textwrap.dedent(
    """
    import sys, time, threading
    sys.path.insert(0, {repo!r})
    import agent.background_review as br
    import cli as cli_mod
    from run_agent import AIAgent

    marker = {marker!r}

    def _review():
        time.sleep(1.5)  # still "in the provider call" when the turn returns
        with open(marker, "w") as fh:
            fh.write("complete")

    br.spawn_background_review_thread = lambda *a, **k: (_review, "prompt")
    agent = object.__new__(AIAgent)
    agent._background_review_run = None
    agent._background_review_lock = threading.Lock()
    agent._background_review_thread = None
    agent._spawn_background_review_now(messages_snapshot=[], review_memory=True)

    class _Cli:
        pass
    c = _Cli()
    c.agent = agent
    c.session_id = "e2e"
    cli_mod._wait_for_oneshot_background_review(c)
    print("EXITING", flush=True)
    """
)


def _run_child(tmp_path: Path, exit_wait_s) -> Path:
    home = tmp_path / f"home_{exit_wait_s}"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"auxiliary:\n  background_review:\n    exit_wait_s: {exit_wait_s}\n", encoding="utf-8")
    marker = tmp_path / f"review_{exit_wait_s}.txt"
    script = tmp_path / f"child_{exit_wait_s}.py"
    script.write_text(_E2E_CHILD.format(repo=str(REPO_ROOT), marker=str(marker)), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env["HERMES_HOME"] = str(home)
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=120, env=env, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "EXITING" in proc.stdout
    return marker


@pytest.mark.platforms("posix")
def test_e2e_one_shot_exit_lets_review_finish(tmp_path):
    marker = _run_child(tmp_path, 30)
    assert marker.exists(), "one-shot exit killed the daemon review mid-flight"


@pytest.mark.platforms("posix")
def test_e2e_exit_wait_zero_drops_review(tmp_path):
    """Control: without the wait, interpreter exit kills the daemon (the original bug shape)."""
    marker = _run_child(tmp_path, 0)
    time.sleep(2.0)
    assert not marker.exists()
