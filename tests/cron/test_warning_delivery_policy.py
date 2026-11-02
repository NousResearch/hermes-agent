"""Failure classification survives durable delivery and drain-time policy changes."""
from unittest.mock import patch

import pytest

from cron import delivery_queue, scheduler
from cron.scheduler_delivery import _deliver_result


@pytest.fixture(autouse=True)
def configured_transport():
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    with patch("gateway.config.load_gateway_config", return_value=config):
        yield


@pytest.mark.parametrize("suppress", [False, True])
def test_failure_queue_settles_without_claiming_a_suppressed_send(tmp_path, monkeypatch, suppress):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "queue.db")
    (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: false}\n")
    job = {"id": "fixture", "name": "fixture", "deliver": "telegram:chat", "execution_id": "run"}
    delivery_queue.enqueue("run", job, "arbitrary diagnostic", for_failure=True)
    (tmp_path / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(suppress).lower()}}}\n")
    with patch("cron.scheduler_delivery._deliver_standalone") as send:
        assert scheduler.drain_delivery_queue({}, None) == 1
    row = delivery_queue.get_status("run")
    assert row["status"] == ("suppressed" if suppress else "delivered")
    assert send.call_count == (0 if suppress else 1)
    assert scheduler.drain_delivery_queue({}, None) == 0


def test_success_content_and_explicit_destination_override_survive(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("display:\n  suppress_warning_notifications: true\n  platforms:\n    telegram:\n      suppress_warning_notifications: false\n")
    job = {"id": "fixture", "deliver": "telegram:chat"}
    with patch("cron.scheduler_delivery._deliver_standalone") as send:
        assert _deliver_result(job, "warning quoted in requested result", for_failure=False) is None
        assert _deliver_result(job, "diagnostic", for_failure=True) is None
        assert send.call_count == 2


def test_mixed_target_queue_is_not_reported_wholly_suppressed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "queue.db")
    (tmp_path / "config.yaml").write_text("display:\n  suppress_warning_notifications: true\n  platforms:\n    telegram:\n      suppress_warning_notifications: false\n")
    job = {"id": "fixture", "deliver": ["slack:muted", "telegram:allowed"]}
    delivery_queue.enqueue("mixed", job, "diagnostic", for_failure=True)
    with patch("cron.scheduler_delivery._deliver_standalone") as send:
        assert scheduler.drain_delivery_queue({}, None) == 1
    assert send.call_count == 1
    assert delivery_queue.get_status("mixed")["status"] == "delivered"
