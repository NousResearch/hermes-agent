"""Standalone cron delivery retries transient DNS/connect failures inside the tick."""

import pytest

import cron.scheduler_delivery as sd


class _Transport:
    def __init__(self, is_relay=False):
        self.is_relay = is_relay


def _target(is_relay=False):
    fields = {name: None for name in sd._TargetDelivery.__dataclass_fields__}
    fields.update(job={"id": "job1"}, platform_name="telegram", chat_id="-100", mirror_text="",
                  mirror_this_target=False, transport=_Transport(is_relay))
    return sd._TargetDelivery(**fields)


@pytest.fixture
def fast(monkeypatch):
    sleeps = []
    monkeypatch.setattr(sd.time, "sleep", sleeps.append)
    monkeypatch.setattr(sd, "_get_delivery_retry_attempts", lambda: 3)
    monkeypatch.setattr(sd, "_get_delivery_retry_backoff_seconds", lambda: 7.0)
    monkeypatch.setattr(sd, "_queue_for_live_reconnect", lambda *a, **k: None)
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)
    return sleeps


def _script(monkeypatch, outcomes):
    calls = []

    def fake(t, content, media):
        calls.append(content)
        return outcomes[min(len(calls), len(outcomes)) - 1]

    monkeypatch.setattr(sd, "_standalone_send", fake)
    return calls


def test_dns_blip_is_retried_then_delivered(monkeypatch, fast):
    calls = _script(monkeypatch, [(None, "ConnectError [Errno 11001] getaddrinfo failed"), ({"success": True}, None)])
    target_errors, delivery_errors = [], []
    sd._deliver_standalone(_target(), "report", [], target_errors, delivery_errors)
    assert len(calls) == 2 and fast == [7.0]
    assert target_errors == [] and delivery_errors == []


def test_persistent_dns_failure_stops_at_attempt_cap_and_records_error(monkeypatch, fast):
    calls = _script(monkeypatch, [(None, "getaddrinfo failed")])
    target_errors, delivery_errors = [], []
    sd._deliver_standalone(_target(), "report", [], target_errors, delivery_errors)
    assert len(calls) == 3 and fast == [7.0, 7.0]
    assert delivery_errors and "getaddrinfo" in delivery_errors[0]


def test_result_dict_error_is_also_retried(monkeypatch, fast):
    calls = _script(monkeypatch, [({"error": "Connect error: name or service not known"}, None),
                                  ({"success": True}, None)])
    errs = []
    sd._deliver_standalone(_target(), "report", [], errs, errs)
    assert len(calls) == 2 and errs == []


def test_non_transient_failure_is_not_retried(monkeypatch, fast):
    calls = _script(monkeypatch, [(None, "chat not found")])
    target_errors, delivery_errors = [], []
    sd._deliver_standalone(_target(), "report", [], target_errors, delivery_errors)
    assert len(calls) == 1 and fast == []
    assert any("chat not found" in e for e in delivery_errors)


def test_post_send_timeout_is_never_retried(monkeypatch, fast):
    calls = _script(monkeypatch, [(None, "Timed out")])
    sd._deliver_standalone(_target(), "report", [], [], [])
    assert len(calls) == 1


def test_attempts_one_disables_retry(monkeypatch, fast):
    monkeypatch.setattr(sd, "_get_delivery_retry_attempts", lambda: 1)
    calls = _script(monkeypatch, [(None, "getaddrinfo failed")])
    sd._deliver_standalone(_target(), "report", [], [], [])
    assert len(calls) == 1 and fast == []


def test_relay_target_still_fails_closed(monkeypatch, fast):
    calls = _script(monkeypatch, [({"success": True}, None)])
    errs = []
    sd._deliver_standalone(_target(is_relay=True), "report", [], errs, errs)
    assert calls == [] and errs
