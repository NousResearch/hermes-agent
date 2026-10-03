"""prepare_launch() must never ask the bootstrap to re-exec into the same state.

Regression for #126908: every CLI invocation (a chat message, ``gateway stop``,
``doctor``) relaunched into the source-update completion path instead of
reaching its command, even though ``hermes update`` reported success. Two
shapes of the same loop:

- the completion retry was deferred (backoff window / retry cap) so no sync
  ran, yet the stale ``not current`` verdict still requested a relaunch onto
  the interpreter this process already runs;
- a sync committed but the install still reports stale, so each relaunched
  process syncs again, forever.

Both now degrade loudly (run the command, or raise with the ``hermes update``
remedy) instead of re-execing. Behaviour contracts, not snapshots: the tests
drive the real ``prepare_launch`` with a real lock and real marker files,
stubbing only the PM boundary and the interpreter resolution.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from hermes_cli import venv_sync


@pytest.fixture(autouse=True)
def _no_tool_downloads(monkeypatch):
    """The launch sync publishes lockfile tools first; these tests cover the relaunch decision."""
    import pm.client

    monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: None)


@pytest.fixture
def completion_tail(monkeypatch):
    """Record the source-completion child instead of running product builds."""
    calls = []

    def call(command, **kwargs):
        calls.append(command)
        return 0

    monkeypatch.setattr(venv_sync.subprocess, "call", call)
    return calls


def _self_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='example'\n")
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "self"}))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.delenv("HERMES_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    return root


def _same_interpreter(monkeypatch):
    """Resolve the store interpreter to the running one: relaunch would be a no-op."""
    from hermes_cli import _launchers

    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: Path(sys.executable))


def test_capped_retry_with_stale_deps_runs_command_instead_of_relaunching(
    tmp_path, monkeypatch, completion_tail, capsys
):
    """Past the retry cap a stale install must NOT request a relaunch (#126908).

    No sync runs (the tail is left for an explicit ``hermes update``), and this
    process already runs the store interpreter, so returning it would make the
    bootstrap re-exec into the identical state on every CLI invocation.
    """
    import pm

    root = _self_checkout(tmp_path, monkeypatch)
    _same_interpreter(monkeypatch)
    venv_sync.arm_completion(root)
    attempts = venv_sync._completion_attempts_path(root)
    attempts.write_text(f"{venv_sync.COMPLETION_RETRY_MAX_ATTEMPTS}\n", encoding="utf-8")
    os.utime(attempts, (1.0, 1.0))  # ancient: age cannot rescue a capped record
    syncs = []
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: False)
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: syncs.append(a))

    assert venv_sync.prepare_launch(root, []) is None

    assert syncs == []
    assert completion_tail == []
    assert venv_sync.completion_pending_path(root).is_file()
    assert "hermes update" in capsys.readouterr().err


def test_backoff_retry_with_stale_deps_runs_command_instead_of_relaunching(
    tmp_path, monkeypatch, completion_tail
):
    """Inside the backoff window a stale install must NOT request a relaunch (#126908)."""
    import pm

    root = _self_checkout(tmp_path, monkeypatch)
    _same_interpreter(monkeypatch)
    venv_sync.arm_completion(root)
    venv_sync._completion_attempts_path(root).write_text("2\n", encoding="utf-8")
    may_retry, _, _ = venv_sync.completion_retry_state(root)
    assert not may_retry, "fixture must hold the backoff window open"
    syncs = []
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: False)
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: syncs.append(a))

    assert venv_sync.prepare_launch(root, []) is None

    assert syncs == []
    assert completion_tail == []
    assert venv_sync.completion_pending_path(root).is_file()


def test_nonconverging_sync_raises_instead_of_relaunching(
    tmp_path, monkeypatch, completion_tail
):
    """A sync that leaves the install stale must fail loudly, not relaunch (#126908)."""
    import pm

    root = _self_checkout(tmp_path, monkeypatch)
    _same_interpreter(monkeypatch)
    syncs = []
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: False)
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: syncs.append(a))

    with pytest.raises(RuntimeError, match="out of date"):
        venv_sync.prepare_launch(root, [])

    assert len(syncs) == 1


def test_successful_sync_relaunches_once_then_settles(tmp_path, monkeypatch, completion_tail):
    """The fixed path still converges: one relaunch onto the new generation, then run."""
    import pm

    root = _self_checkout(tmp_path, monkeypatch)
    _same_interpreter(monkeypatch)
    state = {"current": False}
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: state["current"])

    def sync(*args, **kwargs):
        state["current"] = True

    monkeypatch.setattr(pm, "sync_venv", sync)

    # First launch syncs and asks for the single convergence relaunch.
    assert venv_sync.prepare_launch(root, []) == Path(sys.executable)
    # The relaunched process (same interpreter, now current) runs its command.
    assert venv_sync.prepare_launch(root, []) is None
