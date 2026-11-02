"""`hermes pm install --without EXTRA` drops a recorded extra, and nothing that carries a
selection forward (recorded extras, the main-era venv scan, a lazy first use) re-adds it
until the user asks for it again with `--extra EXTRA`."""

from __future__ import annotations

import argparse
import importlib
from types import SimpleNamespace

import pytest

import pm.cli
from pm import extras, install
from pm.package import InstallError
from pm.paths import repo_root


@pytest.fixture()
def synced(monkeypatch):
    calls = []
    monkeypatch.setattr(pm.cli, "_install_names", lambda names, target=None, *, verify=True: 0)
    monkeypatch.setattr(importlib.import_module("pm.install"), "activate", lambda **kwargs: [])
    monkeypatch.setattr(importlib.import_module("pm.install"), "sync_venv",
                        lambda extras=None, **kwargs: calls.append(list(extras or [])))
    return calls


def _next_selection(recorded, requested=None):
    package = SimpleNamespace(project_root=repo_root, expected_stamp=lambda enabled, **_: "stamp")
    return install._target_selection(package, {"extras": recorded}, extras=requested, inputs={},
                                     repair=False, shipped=None, frozen=None)[0]


def test_without_drops_a_recorded_extra_from_every_later_selection(synced, tmp_path):
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False, without=["matrix"])) == 0

    # The update's sync carries recorded extras forward; the dropped one stays out, others stay.
    assert _next_selection(["all", "fal", "matrix"]) == ["all", "fal"]
    # A data root without facts re-scans the main-era venv, which is never deleted.
    site = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    for anchor in (*extras._anchors("matrix"), "fal_client"):
        (site / anchor).mkdir(parents=True)
    assert extras.legacy_selection(tmp_path) == ["all", "fal", "matrix"]
    extras.record_declined_extras(add=["matrix"], project_root=tmp_path)
    assert extras.legacy_selection(tmp_path) == ["all", "fal"]
    # A lazy first use refuses instead of silently reinstalling it.
    package = SimpleNamespace(project_root=repo_root)
    with pytest.raises(InstallError, match="pm install --extra matrix"):
        install._refuse_declined(package, ["matrix"], explicit=False)


def test_extra_opts_a_dropped_extra_back_in():
    extras.record_declined_extras(add=["matrix"])
    assert _next_selection(["all"], requested=["matrix"]) == ["all", "matrix"]
    install._refuse_declined(SimpleNamespace(project_root=repo_root), ["matrix"], explicit=True)


def test_without_refuses_an_extra_the_default_sync_selects_anyway(synced, capsys):
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False, without=["web"])) == 1
    assert "extras outside [all]" in capsys.readouterr().out
    assert synced == []


class _BlockingVenv:
    """The venv package with a build that records its selection and can be held mid-build."""

    def __init__(self, hold=None):
        self.built, self.hold = [], hold or {}

    project_root = staticmethod(repo_root)

    def expected_stamp(self, enabled, **_):
        return "+".join(enabled)

    def apply(self, enabled, **_):
        self.built.append(list(enabled))
        if (hook := self.hold.pop(tuple(enabled), None)) is not None:
            hook()
        return {}


def _recorded():
    from pm.environments import runtime_facts_path
    from pm.lock import Facts

    return (Facts(runtime_facts_path(repo_root())).get("venv") or {}).get("extras")


def _watch_lock(monkeypatch, asked=None):
    """Wrap the dependency lock: returns (threads holding it, event set once a second thread asks for it)."""
    import threading
    from contextlib import contextmanager

    holders, asked = set(), asked or threading.Event()
    real_lock = install._venv_install_lock

    @contextmanager
    def lock(**kwargs):
        if threading.current_thread() is not threading.main_thread():
            asked.set()
        with real_lock(**kwargs):
            holders.add(threading.get_ident())
            try:
                yield
            finally:
                holders.discard(threading.get_ident())

    monkeypatch.setattr(install, "_venv_install_lock", lock)
    return holders, asked


def test_a_sync_queued_behind_an_opt_in_keeps_the_extra(monkeypatch):
    """`--extra matrix` clears the decline while it still owns the dependency lock, so a bare
    update sync queued behind it never reads the stale decline and publishes without matrix."""
    import threading

    monkeypatch.setattr(install, "_feature_policy", lambda extras, *, repair: (None, None))
    package = _BlockingVenv()
    monkeypatch.setattr(install, "get_package", lambda name: package)
    install.sync_venv(["all"], explicit=True)
    extras.record_declined_extras(add=["matrix"])

    holders, _ = _watch_lock(monkeypatch)
    queued_sync_selected = threading.Event()
    real_target, real_record = install._target_selection, extras.record_declined_extras

    def target(*args, **kwargs):
        selection = real_target(*args, **kwargs)
        if threading.current_thread() is not threading.main_thread():
            queued_sync_selected.set()
        return selection

    def clear_decline(**kwargs):
        # Cleared after the lock is released, the queued sync goes first: make that interleave certain.
        if kwargs.get("remove") and threading.get_ident() not in holders:
            assert queued_sync_selected.wait(30)
        return real_record(**kwargs)

    queued_errors = []

    def queued_sync():
        try:
            install.sync_venv(explicit=True)
        except BaseException as exc:
            queued_errors.append(exc)

    queued = threading.Thread(target=queued_sync)
    package.hold[("all", "matrix")] = queued.start
    monkeypatch.setattr(install, "_target_selection", target)
    monkeypatch.setattr(extras, "record_declined_extras", clear_decline)

    install.sync_venv(["matrix"], explicit=True)
    queued.join(30)

    assert not queued.is_alive() and queued_errors == []
    assert package.built[-1] == ["all", "matrix"]
    assert _recorded() == ["all", "matrix"]
    assert extras.declined_extras() == frozenset()


def test_without_during_an_opt_in_sync_stays_recorded(monkeypatch):
    """`--without matrix` landing while `--extra matrix` builds waits for it, so the opt-in's
    decline clear cannot erase the newer opt-out."""
    import threading

    monkeypatch.setattr(install, "_feature_policy", lambda extras, *, repair: (None, None))
    settled, real_record = threading.Event(), extras.record_declined_extras
    _watch_lock(monkeypatch, asked=settled)

    def record(**kwargs):
        result = real_record(**kwargs)
        if kwargs.get("add"):
            settled.set()
        return result

    def mid_build():
        threading.Thread(target=install.decline_extras, args=(["matrix"],), name="opt-out").start()
        # Set once the opt-out queues on the lock or, unlocked, has written its decline.
        assert settled.wait(30)

    package = _BlockingVenv({("all", "matrix"): mid_build})
    monkeypatch.setattr(install, "get_package", lambda name: package)
    monkeypatch.setattr(extras, "record_declined_extras", record)

    install.sync_venv(["all", "matrix"], explicit=True)
    for thread in threading.enumerate():
        if thread.name == "opt-out":
            thread.join(30)

    assert extras.declined_extras() == frozenset({"matrix"})

