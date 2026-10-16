"""Low-disk auto-prune prompt: asks only while undecided, says when pruning will not help."""

import shutil
import time
from collections import namedtuple
from pathlib import Path

import pytest

from hermes_cli import prune_advisor

_Usage = namedtuple("_Usage", "total used free")


@pytest.fixture
def home(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    db = SessionDB(db_path=h / "state.db")
    db.create_session("recent", source="cli")
    db.append_message("recent", role="user", content="hi")
    db.end_session("recent", "user_exit")
    db.close()
    return h


def _disk(monkeypatch, free_gb, total_gb=100):
    gib = prune_advisor.GIB
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage(total_gb * gib, (total_gb - free_gb) * gib, free_gb * gib))


def test_asks_only_when_low_and_undecided_and_either_answer_ends_it(home, monkeypatch):
    _disk(monkeypatch, free_gb=50)
    assert prune_advisor.should_ask(home) is None  # plenty of space: silent
    _disk(monkeypatch, free_gb=1)
    assert prune_advisor.should_ask(home) is not None
    prune_advisor.snooze(home)
    assert prune_advisor.should_ask(home) is None  # "later" holds for the window
    prune_advisor._snooze_path(home).unlink()
    for answer in (False, True):
        assert prune_advisor.decide(answer, home)
        assert (home / "config.yaml").read_text(encoding="utf-8-sig").count("auto_prune") == 1
        assert prune_advisor.should_ask(home) is None  # an explicit key ends the question


def test_prompt_says_pruning_will_not_help_when_nothing_is_old_enough(home, monkeypatch):
    _disk(monkeypatch, free_gb=1)
    advice = prune_advisor.advise(home)
    assert advice is not None and advice.total_sessions == 1 and advice.prunable_sessions == 0
    assert not advice.helps
    assert "Pruning will not help" in "\n".join(prune_advisor.advice_lines(advice))
    # Backdate the session past retention: now it is prunable and the prompt offers it.
    import sqlite3
    old = time.time() - 200 * 86400
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("UPDATE sessions SET started_at=?, ended_at=?, last_activity_at=?", (old, old, old))
        conn.execute("UPDATE messages SET timestamp=?", (old,))
    advice = prune_advisor.advise(home)
    assert advice.prunable_sessions == 1
    assert "will not help" not in "\n".join(prune_advisor.advice_lines(advice))
    # Retention is read the way maintenance reads it (effective config, int()): a quoted or
    # env-referenced 0 must not be described as the 90-day default.
    monkeypatch.setenv("KEEP_DAYS", "0")
    for value in ("0", '"0"', "${KEEP_DAYS}"):
        (home / "config.yaml").write_text(f"sessions:\n  retention_days: {value}\n", encoding="utf-8")
        assert prune_advisor.advise(home).retention_days == 0, value


def test_admin_managed_setting_is_never_asked_nor_overwritten(home, monkeypatch, tmp_path):
    """An administrator-pinned auto_prune is already decided: no prompt, and /autoprune off must
    not claim to have turned pruning off while the managed true still applies."""
    from hermes_cli import managed_scope
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("sessions:\n  auto_prune: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    _disk(monkeypatch, free_gb=1)
    try:
        assert prune_advisor.should_ask(home) is None
        reply = prune_advisor.run_command("off", home)
        assert "administrator" in reply
        cfg = home / "config.yaml"
        assert not cfg.exists() or "auto_prune" not in cfg.read_text(encoding="utf-8-sig")
    finally:
        managed_scope.invalidate_managed_cache()


def test_later_re_arms_the_local_notice_after_the_snooze_window(home, monkeypatch):
    _disk(monkeypatch, free_gb=1)
    shown: dict = {}
    assert prune_advisor.notice_due(home, shown)
    shown[str(home)] = time.time()
    assert not prune_advisor.notice_due(home, shown)  # same process, same window: once
    shown[str(home)] = time.time() - prune_advisor.SNOOZE_SECONDS - 1
    assert prune_advisor.notice_due(home, shown)  # a week later it asks again
