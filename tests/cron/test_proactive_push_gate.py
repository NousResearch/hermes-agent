"""Cron delivery honours the per-platform proactive-push opt-out gate."""

import pytest

from cron.scheduler import _filter_proactive_push_optout, _resolve_delivery_targets


def _cfg(mp, config):
    mp.setattr("hermes_cli.config.load_config", lambda: config)


# -- filter helper (unit) ----------------------------------------------------

def test_optout_platform_dropped_others_kept(monkeypatch):
    _cfg(monkeypatch, {"display": {"platforms": {"line": {"proactive_push": False}}}})
    targets = [
        {"platform": "line", "chat_id": "U1", "thread_id": None},
        {"platform": "telegram", "chat_id": "T1", "thread_id": None},
    ]
    kept = _filter_proactive_push_optout(targets, {"id": "j1"})
    assert [t["platform"] for t in kept] == ["telegram"]  # line dropped, telegram kept


def test_deliver_all_still_respects_optout(monkeypatch):
    # An explicit multi-target (deliver=all expands to many) still skips opt-out.
    _cfg(monkeypatch, {"display": {"platforms": {"line": {"proactive_push": False}}}})
    targets = [
        {"platform": "line", "chat_id": "U1"},
        {"platform": "discord", "chat_id": "D1"},
        {"platform": "slack", "chat_id": "S1"},
    ]
    kept = _filter_proactive_push_optout(targets, {"id": "j2"})
    assert {t["platform"] for t in kept} == {"discord", "slack"}


def test_global_optout_drops_all(monkeypatch):
    _cfg(monkeypatch, {"display": {"proactive_push": False}})
    targets = [{"platform": "line", "chat_id": "U1"}, {"platform": "telegram", "chat_id": "T1"}]
    assert _filter_proactive_push_optout(targets, {"id": "j3"}) == []


def test_no_optout_keeps_everything(monkeypatch):
    _cfg(monkeypatch, {"display": {}})
    targets = [{"platform": "line", "chat_id": "U1"}, {"platform": "telegram", "chat_id": "T1"}]
    assert len(_filter_proactive_push_optout(targets, {"id": "j4"})) == 2


def test_fail_open_on_config_error(monkeypatch):
    def boom():
        raise RuntimeError("config unreadable")
    monkeypatch.setattr("hermes_cli.config.load_config", boom)
    targets = [{"platform": "line", "chat_id": "U1"}]
    # Fail-open: deliver as before rather than silently dropping.
    assert _filter_proactive_push_optout(targets, {"id": "j5"}) == targets


def test_skip_is_logged(monkeypatch, caplog):
    import logging
    _cfg(monkeypatch, {"display": {"platforms": {"line": {"proactive_push": False}}}})
    with caplog.at_level(logging.INFO):
        _filter_proactive_push_optout([{"platform": "line", "chat_id": "U1"}], {"id": "jlog"})
    assert any("opted out of proactive push" in r.message and "jlog" in r.message
               for r in caplog.records)


# -- integration through _resolve_delivery_targets ---------------------------

def test_resolve_targets_filters_optout_origin(monkeypatch):
    _cfg(monkeypatch, {"display": {"platforms": {"line": {"proactive_push": False}}}})
    job = {
        "id": "sync", "deliver": "origin",
        "origin": {"platform": "line", "chat_id": "Uddf", "thread_id": None},
    }
    # deliver=origin resolves to line:Uddf, then the gate drops it → no targets.
    assert _resolve_delivery_targets(job) == []


def test_resolve_targets_keeps_non_optout_origin(monkeypatch):
    _cfg(monkeypatch, {"display": {"platforms": {"line": {"proactive_push": False}}}})
    job = {
        "id": "tg", "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "T1", "thread_id": None},
    }
    targets = _resolve_delivery_targets(job)
    assert [t["platform"] for t in targets] == ["telegram"]
