"""Tests for the gateway-baseline-env snapshot mechanism (#31026).

The cron ticker runs as a background thread inside the gateway process and
mutates ``os.environ`` while a per-job profile context is active. The
gateway's asyncio thread keeps handling messages during that window and
must NOT see the cron-mutated values when making auth decisions, or it
will reject legitimate users and trigger an unwanted pairing flow.

The fix is a thread-safe snapshot published by cron's profile-context
manager that ``gateway_getenv`` consults in preference to ``os.environ``.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from hermes_constants import (
    clear_gateway_baseline_env,
    gateway_getenv,
    set_gateway_baseline_env,
)


@pytest.fixture(autouse=True)
def _isolate_snapshot():
    """Ensure each test starts and ends with no baseline snapshot active."""
    clear_gateway_baseline_env()
    yield
    clear_gateway_baseline_env()


def test_gateway_getenv_falls_back_to_live_env_when_no_snapshot(monkeypatch):
    monkeypatch.setenv("ISSUE_31026_KEY", "live-value")
    assert gateway_getenv("ISSUE_31026_KEY") == "live-value"
    assert gateway_getenv("ISSUE_31026_MISSING", "fallback") == "fallback"


def test_baseline_snapshot_shadows_live_env_during_cron_mutation(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "alice")
    set_gateway_baseline_env(dict(os.environ))

    # Simulate cron's load_dotenv(override=True) for a different profile.
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "bob")

    # Live env reflects the cron-thread mutation, but the gateway thread's
    # accessor returns the pre-mutation baseline.
    assert os.environ["FEISHU_ALLOWED_USERS"] == "bob"
    assert gateway_getenv("FEISHU_ALLOWED_USERS") == "alice"


def test_baseline_clear_restores_live_env_reads(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "alice")
    set_gateway_baseline_env(dict(os.environ))
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "bob")
    assert gateway_getenv("FEISHU_ALLOWED_USERS") == "alice"

    clear_gateway_baseline_env()
    # With no snapshot, gateway_getenv reads the live env (whatever it is now).
    assert gateway_getenv("FEISHU_ALLOWED_USERS") == "bob"


def test_baseline_default_returned_for_missing_key():
    set_gateway_baseline_env({"OTHER_KEY": "x"})
    assert gateway_getenv("MISSING_KEY", "default-val") == "default-val"
    # Key present in snapshot but empty string still returns the snapshot value.
    set_gateway_baseline_env({"MISSING_KEY": ""})
    assert gateway_getenv("MISSING_KEY", "default-val") == ""


def test_baseline_is_copied_not_aliased():
    """Mutating the dict we passed in must not affect later gateway reads."""
    snapshot = {"K": "v1"}
    set_gateway_baseline_env(snapshot)
    snapshot["K"] = "v2"  # mutate caller-owned dict
    assert gateway_getenv("K") == "v1"


def test_concurrent_reads_during_env_mutation_see_baseline(monkeypatch):
    """Simulate the actual bug scenario.

    Thread A (cron) sets a baseline, mutates os.environ for many iterations,
    then clears the baseline. Thread B (gateway) keeps reading
    ``gateway_getenv`` and must always see the baseline value while the
    snapshot is set.
    """
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "alice")
    stop = threading.Event()
    observed: list[str] = []
    reader_done = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            observed.append(gateway_getenv("FEISHU_ALLOWED_USERS", ""))
            time.sleep(0)
        reader_done.set()

    def mutator() -> None:
        set_gateway_baseline_env({"FEISHU_ALLOWED_USERS": "alice"})
        try:
            for i in range(200):
                os.environ["FEISHU_ALLOWED_USERS"] = f"bob-{i}"
                time.sleep(0)
        finally:
            clear_gateway_baseline_env()

    r = threading.Thread(target=reader)
    m = threading.Thread(target=mutator)
    r.start()
    m.start()
    m.join(timeout=5)
    stop.set()
    r.join(timeout=5)
    assert reader_done.is_set(), "reader thread did not exit"

    # While the baseline was set, every observation taken during the mutation
    # window must equal the baseline. After clear, observations can be any
    # of the mutated values — but we are only enforcing the during-window
    # invariant which the bug reported.
    #
    # Easier check: at least some observations happened, and none of them
    # equal a "bob-*" value while the baseline was set. Since we can't
    # distinguish per-observation easily, we assert the stronger property
    # that the FIRST few observations (taken before the snapshot was
    # cleared) are all "alice".
    #
    # The mutator clears the baseline only at the end, so all observations
    # before the clear must be "alice".
    leaked = [v for v in observed if v.startswith("bob-")]
    # The reader may continue briefly after the mutator joins; values seen
    # after clear can be "bob-*". To assert no leak DURING the window we
    # confirm that "alice" was observed AT LEAST once and that the count of
    # leaked "bob-*" observations is bounded by the gap between mutator
    # finishing and the reader being stopped — which should be very small.
    assert "alice" in observed, "reader never saw the baseline value"
    # The reader spins until stop is set after mutator joins, so the only
    # bob-* observations should be from the tail (post-clear). Their count
    # is small in practice.
    assert leaked == [] or all(v.startswith("bob-") for v in observed[-len(leaked):]), (
        "saw a leaked 'bob-*' value while the baseline snapshot was supposed "
        "to be active"
    )
