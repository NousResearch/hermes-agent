"""Tests for the configurable kanban ping reason clip (kanban.notify_reason_max_chars)."""

from types import SimpleNamespace

import pytest

from gateway.kanban_watchers_common import notify_reason_max_chars
from gateway.kanban_watchers_notifier import _EVENT_FORMATTERS, _clip, _safe_review_reason


def _cfg(cfg):
    return {"kanban": cfg}


@pytest.fixture
def no_config(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", dict)


def test_reason_limit_defaults_to_160(no_config):
    assert notify_reason_max_chars() == 160


def test_reason_limit_reads_config_value(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg({"notify_reason_max_chars": 1000}))
    assert notify_reason_max_chars() == 1000


def test_reason_limit_zero_means_unclipped(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg({"notify_reason_max_chars": 0}))
    assert notify_reason_max_chars() == 0


def test_reason_limit_negative_clamps_to_zero(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg({"notify_reason_max_chars": -5}))
    assert notify_reason_max_chars() == 0


def test_reason_limit_fails_safe_on_bad_value(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg({"notify_reason_max_chars": "a lot"}))
    assert notify_reason_max_chars() == 160


def test_reason_limit_fails_safe_when_config_unreadable(monkeypatch):
    def boom():
        raise OSError("no config")

    monkeypatch.setattr("hermes_cli.config.load_config", boom)
    assert notify_reason_max_chars() == 160


def test_clip_zero_limit_keeps_full_value():
    ev = SimpleNamespace(payload={"reason": "x" * 5000})
    out = _clip(ev, "reason", "gateway.kanban.ping.reason_suffix", 0)
    assert len(out) > 4000  # the {value} placeholder carried the full 5000 chars


def test_clip_positive_limit_still_clips():
    ev = SimpleNamespace(payload={"reason": "x" * 500})
    out = _clip(ev, "reason", "gateway.kanban.ping.reason_suffix", 160)
    assert "x" * 160 in out
    assert "x" * 161 not in out


def test_blocked_formatter_uses_configured_limit(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg({"notify_reason_max_chars": 200}))
    ev = SimpleNamespace(payload={"reason": "B" * 1000})
    n = SimpleNamespace(head="[t1] head")
    msg, _, _ = _EVENT_FORMATTERS["blocked"](ev, n)
    assert "B" * 200 in msg
    assert "B" * 201 not in msg


def test_blocked_formatter_unclipped_at_zero(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg({"notify_reason_max_chars": 0}))
    ev = SimpleNamespace(payload={"reason": "B" * 3000})
    n = SimpleNamespace(head="[t1] head")
    msg, _, _ = _EVENT_FORMATTERS["blocked"](ev, n)
    assert "B" * 3000 in msg


def test_safe_review_reason_zero_limit_keeps_full_value():
    assert _safe_review_reason("z" * 1000, 0) == "z" * 1000
    assert _safe_review_reason("z" * 1000, 160).endswith("…")
