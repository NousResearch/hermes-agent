"""Fail-loud guard tests for the built-in mem0 rerank arm."""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from plugins.memory.mem0 import Mem0MemoryProvider
from plugins.memory.mem0.rerank_guard import (
    RERANK_BUILTIN,
    RERANK_OFF,
    RerankIncidentManager,
    _send_page,
    get_rerank_incident_manager,
    normalize_rerank_arm,
)


@pytest.mark.parametrize("value", ["builtin", " BUILTIN "])
def test_normalize_rerank_arm_accepts_only_builtin(value):
    assert normalize_rerank_arm(value) == RERANK_BUILTIN


@pytest.mark.parametrize("value", [None, "", "off", False, True, "true", "false", "zerank", "cohere"])
def test_normalize_rerank_arm_fails_closed_for_legacy_and_unknown_values(value):
    assert normalize_rerank_arm(value) == RERANK_OFF


def test_passing_winner_is_the_declared_default_with_frozen_deadline():
    options = {item["key"]: item for item in Mem0MemoryProvider().get_config_schema()}

    assert options["rerank"]["default"] == RERANK_BUILTIN
    assert float(options["rerank_deadline_ms"]["default"]) == pytest.approx(8647.166891023517)


def test_legacy_arm_is_off_before_client_construction(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MEM0_HOST", raising=False)
    (tmp_path / "mem0.json").write_text(json.dumps({
        "host": "http://mem0.test",
        "admin_api_key": "test-key",
        "rerank": "zerank",
    }))
    provider = Mem0MemoryProvider()
    provider.initialize("g9")

    observed = []

    class Client:
        def __init__(self, *args, **kwargs):
            observed.append(provider._rerank)

    monkeypatch.setattr("plugins.memory.mem0._DirectRestMem0Client", Client)
    provider._get_client()

    assert observed == [RERANK_OFF]


@pytest.mark.parametrize("deadline", [0, -1, float("nan"), float("inf"), 8647.166891023518])
def test_builtin_without_valid_deadline_is_off_before_client_construction(
    monkeypatch, tmp_path, deadline
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MEM0_HOST", raising=False)
    (tmp_path / "mem0.json").write_text(json.dumps({
        "host": "http://mem0.test",
        "admin_api_key": "test-key",
        "rerank": "builtin",
        "rerank_deadline_ms": deadline,
    }))
    provider = Mem0MemoryProvider()
    provider.initialize("deadline-floor")
    observed = []

    class Client:
        def __init__(self, *args, **kwargs):
            observed.append(provider._rerank)

    monkeypatch.setattr("plugins.memory.mem0._DirectRestMem0Client", Client)
    provider._get_client()

    assert observed == [RERANK_OFF]
    assert provider._rerank_incidents is None


def test_builtin_contract_is_never_sent_to_mem0_cloud(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MEM0_HOST", raising=False)
    (tmp_path / "mem0.json").write_text(json.dumps({
        "api_key": "test-cloud-key",
        "rerank": "builtin",
        "rerank_deadline_ms": 125,
    }))

    provider = Mem0MemoryProvider()
    provider.initialize("cloud-floor")

    assert provider._rerank == RERANK_OFF
    assert provider._rerank_incidents is None


def test_direct_rest_builtin_request_carries_single_deadline_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "mem0.json").write_text(json.dumps({
        "host": "http://mem0.test",
        "admin_api_key": "test-key",
        "rerank": "builtin",
        "rerank_deadline_ms": 125,
    }))
    provider = Mem0MemoryProvider()
    provider.initialize("wire")
    sent = []
    client = provider._get_client()
    client._request = lambda method, path, *, body=None, params=None: (
        sent.append(body) or {
            "results": [{"memory": "reranked"}],
            "_rerank": {"status": "success", "latency_ms": 12.5, "effective_arm": "builtin"},
        }
    )

    response = client.search(query="semantic query", filters={"user_id": "ace"},
                             rerank=True, rerank_deadline_ms=125, top_k=5)

    assert isinstance(response, dict)
    assert sent[0]["rerank"] is True
    assert sent[0]["rerank_deadline_ms"] == 125
    assert response["_rerank"]["effective_arm"] == "builtin"


def test_provider_feeds_server_metadata_and_missing_metadata_into_real_detector():
    observed = []
    provider = Mem0MemoryProvider()
    object.__setattr__(provider, "_rerank_incidents", SimpleNamespace(observe=observed.append))

    provider._observe_rerank_response(
        {"_rerank": {"status": "timeout", "failure_class": "timeout"}},
        requested=True,
    )
    provider._observe_rerank_response({"results": []}, requested=True)
    provider._observe_rerank_response({"results": []}, requested=False)

    assert observed == [
        {"status": "timeout", "failure_class": "timeout"},
        {"status": "failure", "failure_class": "missing_metadata", "effective_arm": "off"},
    ]


@pytest.mark.parametrize("status", ["failure", "timeout", "cap_rejected"])
def test_each_failure_class_pages_once_then_recovers_once(tmp_path, caplog, status):
    pages = []
    manager = RerankIncidentManager(
        state_path=tmp_path / "state.json",
        latency_budget_ms=100,
        failure_threshold=3,
        alert_fn=pages.append,
    )

    with caplog.at_level(logging.WARNING):
        for _ in range(8):
            manager.observe({"status": status, "failure_class": status, "effective_arm": "off"})
        manager.observe({"status": "success", "latency_ms": 10, "effective_arm": "builtin"})
        assert manager.wait_idle()

    assert len(pages) == 2
    assert "MEM0 RERANK INCIDENT" in pages[0]
    assert status in pages[0]
    assert "Configured arm remains enabled" in pages[0]
    assert "MEM0 RERANK RECOVERED" in pages[1]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2


def test_latency_breach_uses_rolling_p95_and_rearms_after_recovery(tmp_path):
    pages = []
    state_path = tmp_path / "state.json"
    manager = RerankIncidentManager(
        state_path=state_path,
        latency_budget_ms=100,
        failure_threshold=3,
        alert_fn=pages.append,
    )

    for _ in range(3):
        manager.observe({"status": "success", "latency_ms": 150, "effective_arm": "builtin"})
    for _ in range(97):
        manager.observe({"status": "success", "latency_ms": 10, "effective_arm": "builtin"})
    for _ in range(8):
        manager.observe({"status": "success", "latency_ms": 150, "effective_arm": "builtin"})
    assert manager.wait_idle()

    assert ["LATENCY-BREACH" in page for page in pages] == [True, False, True]


def test_persisted_active_incident_survives_manager_replacement(tmp_path):
    pages = []
    state_path = tmp_path / "state.json"
    first = RerankIncidentManager(
        state_path=state_path,
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=pages.append,
    )
    first.observe({"status": "timeout", "failure_class": "timeout", "effective_arm": "off"})
    assert first.wait_idle()

    replacement = RerankIncidentManager(
        state_path=state_path,
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=pages.append,
    )
    replacement.observe({"status": "timeout", "failure_class": "timeout", "effective_arm": "off"})
    replacement.observe({"status": "success", "latency_ms": 5, "effective_arm": "builtin"})
    assert replacement.wait_idle()

    assert len(pages) == 2
    assert json.loads(state_path.read_text())["active"] is False


def test_unparseable_state_rederives_without_crashing(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("not-json")
    pages = []
    manager = RerankIncidentManager(
        state_path=state_path,
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=pages.append,
    )

    manager.observe({"status": "failure", "failure_class": "reranker_error", "effective_arm": "off"})
    assert manager.wait_idle()

    assert len(pages) == 1
    assert json.loads(state_path.read_text())["active"] is True


def test_failed_page_delivery_retries_without_duplicate_transition_warning(tmp_path, caplog):
    attempts = []

    def flaky(message):
        attempts.append(message)
        if len(attempts) == 1:
            raise RuntimeError("forced delivery failure")

    state_path = tmp_path / "state.json"
    manager = RerankIncidentManager(
        state_path=state_path,
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=flaky,
        retry_backoff_s=0.01,
    )

    with caplog.at_level(logging.WARNING):
        manager.observe({"status": "failure", "failure_class": "reranker_error"})
        assert manager.wait_idle()

    assert len(attempts) == 2
    assert json.loads(state_path.read_text())["pending_page"] is None
    assert len([record for record in caplog.records if record.levelno == logging.WARNING]) == 1


def test_recovery_cannot_overwrite_undelivered_onset(tmp_path):
    attempts = []

    def flaky(message):
        attempts.append(message)
        if len(attempts) == 1:
            raise RuntimeError("forced delivery failure")

    state_path = tmp_path / "state.json"
    manager = RerankIncidentManager(
        state_path=state_path,
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=flaky,
    )
    manager.observe({"status": "failure", "failure_class": "reranker_error"})
    manager.observe({"status": "success", "latency_ms": 5, "effective_arm": "builtin"})
    assert manager.wait_idle()
    assert ["INCIDENT" in message for message in attempts] == [True, True]
    assert json.loads(state_path.read_text())["active"] is True

    manager.observe({"status": "success", "latency_ms": 5, "effective_arm": "builtin"})
    assert manager.wait_idle()

    assert "RECOVERED" in attempts[-1]
    assert json.loads(state_path.read_text())["active"] is False


def test_renewed_failure_cancels_undelivered_recovery(tmp_path):
    attempts = []

    def fail_recovery(message):
        attempts.append(message)
        if "RECOVERED" in message:
            raise RuntimeError("forced recovery delivery failure")

    state_path = tmp_path / "state.json"
    manager = RerankIncidentManager(
        state_path=state_path,
        latency_budget_ms=100,
        failure_threshold=2,
        alert_fn=fail_recovery,
        retry_backoff_s=0.1,
    )
    failure = {"status": "failure", "failure_class": "timeout", "effective_arm": "off"}
    manager.observe(failure)
    manager.observe(failure)
    manager.observe({"status": "success", "latency_ms": 5, "effective_arm": "builtin"})
    deadline = time.monotonic() + 0.5
    while len(attempts) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert ["RECOVERED" in message for message in attempts] == [False, True]

    manager.observe(failure)
    assert manager.wait_idle()

    assert ["RECOVERED" in message for message in attempts] == [False, True]
    state = json.loads(state_path.read_text())
    assert state["pending_page"] is None
    assert state["consecutive_failures"] == 1


def test_malformed_success_metadata_enters_incident_state(tmp_path):
    malformed = [
        {"status": "success", "effective_arm": "builtin"},
        {"status": "success", "latency_ms": float("nan"), "effective_arm": "builtin"},
        {"status": "success", "latency_ms": float("inf"), "effective_arm": "builtin"},
        {"status": "success", "latency_ms": 5, "effective_arm": "off"},
    ]

    for index, metadata in enumerate(malformed):
        pages = []
        manager = RerankIncidentManager(
            state_path=tmp_path / f"state-{index}.json",
            latency_budget_ms=100,
            failure_threshold=1,
            alert_fn=pages.append,
        )
        manager.observe(metadata)
        assert manager.wait_idle()
        assert len(pages) == 1
        assert "MEM0 RERANK INCIDENT" in pages[0]


def test_production_observe_does_not_block_on_alert_or_duplicate_pages(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    pages = []

    def blocking_alert(message):
        pages.append(message)
        entered.set()
        release.wait(1)

    manager = RerankIncidentManager(
        state_path=tmp_path / "state.json",
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=blocking_alert,
    )
    manager.observe({"status": "failure", "failure_class": "timeout", "effective_arm": "off"})
    assert entered.wait(0.5)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=16) as callers:
        list(callers.map(
            manager.observe,
            [{"status": "failure", "failure_class": "timeout", "effective_arm": "off"}] * 32,
        ))
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 0.1
    assert manager.wait_idle()
    assert len(pages) == 1


def test_persistence_and_delivery_have_one_background_owner(tmp_path, monkeypatch):
    request_thread = threading.get_ident()
    owner_threads = []
    manager = RerankIncidentManager(
        state_path=tmp_path / "state.json",
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=lambda message: owner_threads.append(("alert", threading.get_ident())),
    )
    real_write = manager._write

    def recording_write(state):
        owner_threads.append(("write", threading.get_ident()))
        real_write(state)

    monkeypatch.setattr(manager, "_write", recording_write)
    manager.observe({"status": "failure", "failure_class": "timeout", "effective_arm": "off"})
    assert manager.wait_idle()

    assert {kind for kind, _ in owner_threads} == {"alert", "write"}
    assert {thread_id for kind, thread_id in owner_threads if kind == "write"} == {
        manager.owner_thread_id
    }
    assert len({thread_id for kind, thread_id in owner_threads if kind == "alert"}) == 1
    assert manager.owner_thread_id != request_thread


def test_observer_catches_and_counts_bad_metadata(tmp_path):
    manager = RerankIncidentManager(
        state_path=tmp_path / "state.json",
        latency_budget_ms=100,
        alert_fn=lambda message: None,
    )

    class BadMetadata:
        def keys(self):
            raise RuntimeError("forced observer failure")

    manager.observe(BadMetadata())

    assert manager.observer_error_count == 1


def test_bounded_observation_deque_drops_oldest_and_counts_overflow(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def blocking_alert(message):
        entered.set()
        release.wait(1)

    manager = RerankIncidentManager(
        state_path=tmp_path / "state.json",
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=blocking_alert,
        queue_size=2,
    )
    manager.observe({"status": "failure", "failure_class": "timeout", "effective_arm": "off"})
    assert entered.wait(0.5)
    for index in range(10):
        manager.observe({
            "status": "failure",
            "failure_class": f"queued-{index}",
            "effective_arm": "off",
        })
    release.set()

    assert manager.wait_idle()
    assert manager.observer_drop_count == 8


def test_initial_write_failure_persists_before_delivery_and_replacement_replays(
    tmp_path, monkeypatch
):
    state_path = tmp_path / "state.json"
    first_attempts = []

    def hold_pending_for_replacement(message):
        first_attempts.append(message)
        manager._retry_backoff_s = 10
        raise RuntimeError("hold pending for replacement")

    manager = RerankIncidentManager(
        state_path=state_path,
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=hold_pending_for_replacement,
        retry_backoff_s=0.01,
    )
    real_write = manager._write
    writes = 0

    def fail_once(state):
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError("forced first write failure")
        real_write(state)

    monkeypatch.setattr(manager, "_write", fail_once)

    manager.observe({"status": "failure", "failure_class": "timeout", "effective_arm": "off"})
    deadline = time.monotonic() + 0.5
    while not first_attempts and time.monotonic() < deadline:
        time.sleep(0.005)

    assert writes >= 2
    assert len(first_attempts) == 1
    assert manager.persistence_error_count >= 1
    assert json.loads(state_path.read_text())["pending_page"] == "onset"

    pages = []
    replacement = RerankIncidentManager(
        state_path=state_path,
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=pages.append,
        retry_backoff_s=0.01,
    )
    assert replacement.wait_idle(timeout=1)

    assert len(pages) == 1
    assert json.loads(state_path.read_text())["pending_page"] is None


def test_concurrent_direct_observers_claim_one_transition(tmp_path):
    pages = []

    def slow_alert(message):
        time.sleep(0.01)
        pages.append(message)

    manager = RerankIncidentManager(
        state_path=tmp_path / "state.json",
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=slow_alert,
    )
    metadata = {"status": "failure", "failure_class": "timeout", "effective_arm": "off"}

    with ThreadPoolExecutor(max_workers=16) as callers:
        list(callers.map(manager.observe, [metadata] * 32))
    assert manager.wait_idle()

    assert len(pages) == 1


def test_singleton_rejects_conflicting_live_config(tmp_path):
    state_path = tmp_path / "singleton.json"
    first = get_rerank_incident_manager(
        state_path=state_path, latency_budget_ms=100, failure_threshold=3
    )
    assert get_rerank_incident_manager(
        state_path=state_path, latency_budget_ms=100, failure_threshold=3
    ) is first

    with pytest.raises(RuntimeError, match="gateway restart required"):
        get_rerank_incident_manager(
            state_path=state_path, latency_budget_ms=101, failure_threshold=3
        )


def test_default_page_uses_required_discord_alert_destination(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.send_cmd",
        SimpleNamespace(_load_hermes_env=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.send_message_tool",
        SimpleNamespace(
            send_message_tool=lambda args: calls.append(args) or json.dumps({"success": True})
        ),
    )

    _send_page("incident")

    assert calls == [{
        "action": "send",
        "target": "discord:1480528231286181948",
        "message": "incident",
    }]


@pytest.mark.parametrize("response", [{}, {"success": False}, "not-json"])
def test_default_page_requires_positive_delivery_ack(monkeypatch, response):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.send_cmd",
        SimpleNamespace(_load_hermes_env=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.send_message_tool",
        SimpleNamespace(send_message_tool=lambda args: response),
    )

    with pytest.raises(RuntimeError, match="positive success acknowledgement"):
        _send_page("incident")


def test_initial_hung_delivery_is_bounded_serialized_and_self_retries(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    attempts = []

    def blocking_alert(message):
        attempts.append(message)
        entered.set()
        release.wait(1)
    state_path = tmp_path / "state.json"
    manager = RerankIncidentManager(
        state_path=state_path,
        latency_budget_ms=100,
        failure_threshold=1,
        alert_fn=blocking_alert,
        delivery_timeout_s=0.02,
        retry_backoff_s=0.01,
    )
    manager.observe({"status": "failure", "failure_class": "first", "effective_arm": "off"})
    assert entered.wait(0.2)
    time.sleep(0.08)

    manager.observe({"status": "failure", "failure_class": "second", "effective_arm": "off"})
    deadline = time.monotonic() + 0.2
    state = json.loads(state_path.read_text())
    in_memory = {}
    while time.monotonic() < deadline:
        with manager._lock:
            in_memory = dict(manager._state)
        if in_memory.get("failure_class") == "second":
            break
        time.sleep(0.005)

    assert len(attempts) == 1
    assert state["pending_page"] == "onset"
    assert in_memory["pending_page"] == "onset"
    assert in_memory["failure_class"] == "second"
    release.set()
    assert manager.wait_idle(timeout=1)
    assert len(attempts) == 1
    assert json.loads(state_path.read_text())["pending_page"] is None
