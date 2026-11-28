"""``local_runtime.preload``: warming a model must never cost a model that is already in memory.

Residency is demand-driven; the preload is an opt-in exception. Its contract is that it loads a staged
model that is not held, and does nothing when the router is at its admitted cap, since the router would
evict its least-recently-used model to make room. A load already in flight counts as held until the
router reports it, so two keystrokes (or two chats) cannot both pass the cap check."""

from __future__ import annotations

import threading

import pytest

from hermes_cli.local_runtime import bootstrap
from tests.waiting import wait_until


class FakeSupervisor:
    """The router as the supervisor reports it: statuses by model id, a cap, and a blockable load."""

    def __init__(self, statuses: dict[str, str], models_max: int):
        self.statuses = statuses
        self.models_max = models_max
        self.loads: list[str] = []
        self.load_started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def models(self, timeout_s: int = 30) -> dict[str, str]:
        return dict(self.statuses)

    def load_model(self, model_id: str, timeout_s: int = 600) -> None:
        self.loads.append(model_id)
        self.load_started.set()
        assert self.release.wait(timeout=10)


@pytest.fixture
def supervisor(monkeypatch):
    def install(statuses: dict[str, str], models_max: int) -> FakeSupervisor:
        sup = FakeSupervisor(statuses, models_max)
        monkeypatch.setattr(bootstrap, "_SUPERVISOR", sup)
        return sup

    yield install
    bootstrap._PRELOADING.clear()


def _settled() -> bool:
    return not bootstrap._PRELOADING


def test_preload_loads_a_staged_model_that_is_not_held(supervisor):
    sup = supervisor({"a": "unloaded"}, models_max=1)
    assert bootstrap.preload_model("a") == "started"
    wait_until(_settled, what="preload thread to finish")
    assert sup.loads == ["a"]


@pytest.mark.parametrize("status", ["loaded", "ready", "loading", "sleeping"])
def test_preload_skips_a_model_the_router_already_holds(supervisor, status):
    sup = supervisor({"a": status}, models_max=4)
    assert bootstrap.preload_model("a") == "held"
    assert sup.loads == []


def test_preload_never_evicts_a_resident_model_at_the_cap(supervisor):
    sup = supervisor({"a": "loaded", "b": "unloaded"}, models_max=1)
    assert bootstrap.preload_model("b") == "at_cap"
    assert sup.loads == []


def test_a_load_in_flight_counts_toward_the_cap_and_dedupes(supervisor):
    sup = supervisor({"a": "unloaded", "b": "unloaded"}, models_max=1)
    sup.release.clear()
    assert bootstrap.preload_model("a") == "started"
    assert sup.load_started.wait(timeout=5)
    # The router has not reported "a" as loading yet: the in-flight set is what holds it.
    assert bootstrap.preload_model("a") == "held"
    assert bootstrap.preload_model("b") == "at_cap"
    sup.release.set()
    wait_until(_settled, what="preload thread to finish")
    assert sup.loads == ["a"]


def test_preload_of_a_model_that_is_not_staged_does_nothing(supervisor):
    sup = supervisor({"a": "unloaded"}, models_max=1)
    assert bootstrap.preload_model("missing") == "not_staged"
    assert sup.loads == []


def test_preload_boots_the_server_when_this_process_has_not_and_then_loads(supervisor, monkeypatch):
    sup = FakeSupervisor({"a": "unloaded"}, models_max=1)
    supervisor({}, models_max=1)  # also arranges the in-flight set's cleanup
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "ensure_local_runtime", lambda config, force=False: sup)
    assert bootstrap.preload_model("a") == "started"
    wait_until(_settled, what="preload thread to finish")
    assert sup.loads == ["a"]


def test_two_requests_racing_the_boot_both_get_the_server_this_process_started(supervisor, monkeypatch):
    sup = FakeSupervisor({"a": "unloaded"}, models_max=1)
    supervisor({}, models_max=1)  # also arranges the in-flight set's cleanup
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    booting = threading.Event()

    def slow_boot(config, force=False):
        booting.set()
        # health: allow HX013 -- the delay is the behaviour under test: the second request must arrive mid-boot
        threading.Event().wait(0.3)
        monkeypatch.setattr(bootstrap, "_SUPERVISOR", sup)
        return sup

    monkeypatch.setattr(bootstrap, "ensure_local_runtime", slow_boot)
    verdicts: list[str] = []
    first = threading.Thread(target=lambda: verdicts.append(bootstrap.preload_model("a")))
    first.start()
    assert booting.wait(timeout=5)
    verdicts.append(bootstrap.preload_model("a"))
    first.join(timeout=10)

    assert sorted(verdicts) == ["held", "started"]
    wait_until(_settled, what="preload thread to finish")
    assert sup.loads == ["a"]


def test_preload_leaves_a_server_another_process_owns_alone(supervisor, monkeypatch):
    # Booting finds the server another process already supervises and returns no supervisor: its
    # cap is unknown here, so it is left alone.
    supervisor({}, models_max=1)
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "ensure_local_runtime", lambda config, force=False: None)
    assert bootstrap.preload_model("a") == "no_supervisor"
