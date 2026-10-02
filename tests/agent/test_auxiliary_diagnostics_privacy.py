"""Auxiliary notices must not serialize provider credentials or exception bodies."""
from types import SimpleNamespace

from agent import auxiliary_reasoning_floor as floor
from agent.background_review import _warn_review_routing_fallback


def test_reasoning_floor_notice_omits_route_credentials_and_request_metadata(monkeypatch, caplog):
    route = "https://synthetic-user:synthetic-password@example.invalid/v1?key=synthetic-key"
    model = "synthetic-private-model"
    monkeypatch.setattr(floor, "_FLOORED_ROUTES", set())
    floor.remember_reasoning_floor("custom", route, {"model": model}, ValueError("synthetic-error"))
    with caplog.at_level("INFO", logger=floor.logger.name):
        result = floor.known_reasoning_floor(
            {"enabled": False}, "custom", route, model, task="synthetic-private-task",
        )
    assert result == {"enabled": True, "effort": floor.REASONING_FLOOR_EFFORT}
    assert "cannot disable reasoning" in caplog.text
    for value in ("synthetic-user", "synthetic-password", "synthetic-key", model, "synthetic-private-task"):
        assert value not in caplog.text
        assert all(value not in str(record.args) for record in caplog.records)


def test_routing_failure_keeps_actionable_once_only_notice_without_exception_data(caplog):
    notices = []
    agent = SimpleNamespace(provider="openai-codex", model="gpt-6-luna",
                            _emit_warning=notices.append)
    error = ValueError("Authorization: Bearer synthetic-token\npassword=synthetic-password")
    for _ in range(2):
        _warn_review_routing_fallback(agent, "synthetic-review-provider", "synthetic-review-model", error)
    assert len(notices) == 1
    assert "ValueError" in notices[0]
    assert "could not be routed" in notices[0] and "falling back to" in notices[0]
    assert "hermes doctor" in notices[0]
    # The already resolved main route remains visible for cost awareness.
    # Only the failed auxiliary settings and raw exception are private here.
    assert "openai-codex/gpt-6-luna" in notices[0]
    records = [record for record in caplog.records if record.name == "agent.background_review"]
    assert len(records) == 2
    output = caplog.text + notices[0]
    for value in ("synthetic-token", "synthetic-password",
                  "synthetic-review-provider", "synthetic-review-model"):
        assert value not in output
