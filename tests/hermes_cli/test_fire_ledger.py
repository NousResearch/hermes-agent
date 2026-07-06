"""Durability + safety tests for the AGGREGATED plugin fire ledger (v2).

Same properties the dual-review demanded, retargeted to the counter model:
- no outbound network on the flush path (socket-deny),
- concurrent multi-process flush never corrupts and never loses counts,
- prune only deletes CLOSED day-files, never a live one,
- corrupt/partial day-files are skipped with a count, not crashed on,
- aggregation actually collapses volume (N fires -> 1 counter row).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import hermes_cli.fire_ledger as fl


def _cfg(tmp_path: Path, **over) -> fl.FireLedgerConfig:
    base = dict(enabled=True, dir=tmp_path / "ledger", retention_days=30, max_mb=25, min_observation_days=30)
    base.update(over)
    return fl.FireLedgerConfig(**base)


def _fresh_agg():
    """Reset the module-global aggregator so tests don't bleed counts."""
    fl._AGG = fl._Aggregator()


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------

def test_disabled_config_records_nothing(tmp_path):
    _fresh_agg()
    cfg = _cfg(tmp_path, enabled=False)
    assert fl.record_hook_fire(plugin_name="p", hook_name="post_tool_call", kwargs={}, cfg=cfg) is False
    fl.flush(cfg)
    assert not (tmp_path / "ledger").exists() or not list((tmp_path / "ledger").glob("*.json"))


def test_load_config_failopen_disabled_on_garbage():
    assert fl.load_fire_ledger_config(cfg={"plugins": "not-a-dict"}).enabled is False


def test_load_config_clamps_bad_ints():
    cfg = fl.load_fire_ledger_config(cfg={"plugins": {"fire_ledger": {
        "enabled": True, "retention_days": "oops", "max_mb": -5,
    }}})
    assert cfg.enabled is True
    assert cfg.retention_days == fl.DEFAULT_RETENTION_DAYS
    assert cfg.max_mb == 1


# ---------------------------------------------------------------------------
# Aggregation: many fires -> one counter row
# ---------------------------------------------------------------------------

def test_many_fires_collapse_to_one_counter(tmp_path):
    _fresh_agg()
    cfg = _cfg(tmp_path)
    for _ in range(1000):
        fl.record_hook_fire(plugin_name="chatty-guard", hook_name="post_tool_call",
                            kwargs={"tool_name": "terminal"}, cfg=cfg)
    fl.flush(cfg)
    res = fl.read_counters(cfg=cfg)
    rows = [r for r in res.counters if r["plugin_name"] == "chatty-guard"]
    assert len(rows) == 1, "1000 identical fires must collapse to ONE counter row"
    assert rows[0]["count"] == 1000
    assert rows[0]["decision"] == "noop"
    assert "terminal" in rows[0]["tool_classes"]
    # the whole day-file for one pid should be tiny (bytes, not MB)
    files = list(cfg.dir.glob("*.json"))
    assert len(files) == 1
    assert files[0].stat().st_size < 4096


def test_distinct_decisions_split_counters(tmp_path):
    _fresh_agg()
    cfg = _cfg(tmp_path)
    for _ in range(5):
        fl.record_hook_fire(plugin_name="g", hook_name="post_tool_call", kwargs={}, result={"action": "block"}, cfg=cfg)
    for _ in range(3):
        fl.record_hook_fire(plugin_name="g", hook_name="post_tool_call", kwargs={}, cfg=cfg)
    fl.flush(cfg)
    by_decision = {r["decision"]: r["count"] for r in fl.read_counters(cfg=cfg).counters if r["plugin_name"] == "g"}
    assert by_decision == {"block": 5, "noop": 3}


# ---------------------------------------------------------------------------
# Safe metadata — no raw command leaks (counters carry only tool-class)
# ---------------------------------------------------------------------------

def test_no_raw_command_text_in_files(tmp_path):
    _fresh_agg()
    cfg = _cfg(tmp_path)
    fl.record_middleware_fire(plugin_name="p", kind="tool_request",
        kwargs={"tool_name": "terminal", "args": {"command": "git push --force secret-branch"}}, cfg=cfg)
    fl.flush(cfg)
    blob = (list(cfg.dir.glob("*.json"))[0]).read_text()
    assert "secret-branch" not in blob
    assert "--force" not in blob
    assert "terminal" in blob  # tool-class is fine


# ---------------------------------------------------------------------------
# No outbound network on the flush path
# ---------------------------------------------------------------------------

def test_flush_succeeds_under_socket_deny(tmp_path, monkeypatch):
    _fresh_agg()
    def deny(*a, **k):
        raise AssertionError("ledger attempted network I/O")
    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    cfg = _cfg(tmp_path)
    fl.record_hook_fire(plugin_name="p", hook_name="post_tool_call", kwargs={}, cfg=cfg)
    assert fl.flush(cfg) is True
    assert fl.read_counters(cfg=cfg).counters


# ---------------------------------------------------------------------------
# Fail-open: a bad flush never raises into dispatch
# ---------------------------------------------------------------------------

def test_record_never_raises_on_flush_error(tmp_path, monkeypatch):
    _fresh_agg()
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    # record forces a flush via the atexit path only; explicit flush must swallow
    fl.record_hook_fire(plugin_name="p", hook_name="post_tool_call", kwargs={}, cfg=cfg)
    assert fl.flush(cfg) is False  # swallowed, not raised


# ---------------------------------------------------------------------------
# Corruption tolerance
# ---------------------------------------------------------------------------

def test_reader_skips_corrupt_day_file_with_count(tmp_path):
    _fresh_agg()
    cfg = _cfg(tmp_path)
    fl.record_hook_fire(plugin_name="good", hook_name="post_tool_call", kwargs={}, cfg=cfg)
    fl.flush(cfg)
    _fresh_agg()  # drop in-memory so we read purely from disk
    # a partial/garbage day-file + a wrong-schema one
    (cfg.dir / "2020-01-01.111.json").write_text('{"schema":"hermes.plugin_fire.v2","day":"2020-01-01","rows":[TRUNC')
    (cfg.dir / "2020-01-02.222.json").write_text('{"schema":"wrong","rows":[]}')
    res = fl.read_counters(cfg=cfg)
    assert res.skipped_corrupt == 2
    assert any(r["plugin_name"] == "good" for r in res.counters)


# ---------------------------------------------------------------------------
# Prune only closed files
# ---------------------------------------------------------------------------

def test_prune_deletes_closed_not_live(tmp_path):
    _fresh_agg()
    cfg = _cfg(tmp_path)
    fl.record_hook_fire(plugin_name="live", hook_name="post_tool_call", kwargs={}, cfg=cfg)
    fl.flush(cfg)
    live = list(cfg.dir.glob("*.json"))[0]
    old = cfg.dir / "2020-01-01.999999.json"
    old.write_text('{"schema":"hermes.plugin_fire.v2","day":"2020-01-01","pid":999999,"rows":[]}')
    out = fl.prune_closed_files(cfg)
    assert live.exists(), "live file must NOT be pruned"
    assert not old.exists(), "closed old-day file should be pruned"
    assert out["deleted"] == 1


# ---------------------------------------------------------------------------
# Concurrency: N processes each flush their OWN file, no loss, no corruption
# ---------------------------------------------------------------------------

def _worker(dir_str, n, idx):
    cfg = fl.FireLedgerConfig(enabled=True, dir=Path(dir_str), retention_days=30, max_mb=25, min_observation_days=30)
    for _ in range(n):
        fl.record_hook_fire(plugin_name=f"proc{idx}", hook_name="post_tool_call", kwargs={}, cfg=cfg)
    fl.flush(cfg)


def _pruner(dir_str, rounds):
    cfg = fl.FireLedgerConfig(enabled=True, dir=Path(dir_str), retention_days=30, max_mb=25, min_observation_days=30)
    for _ in range(rounds):
        fl.prune_closed_files(cfg)


def test_concurrent_flush_plus_pruner_no_loss(tmp_path):
    _fresh_agg()
    cfg = _cfg(tmp_path)
    cfg.dir.mkdir(parents=True, exist_ok=True)
    n_procs, per_proc = 4, 500
    procs = [mp.Process(target=_worker, args=(str(cfg.dir), per_proc, i)) for i in range(n_procs)]
    procs.append(mp.Process(target=_pruner, args=(str(cfg.dir), 20)))
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    _fresh_agg()  # read purely from the flushed files
    res = fl.read_counters(cfg=cfg)
    total = sum(r["count"] for r in res.counters)
    assert total == n_procs * per_proc, f"count loss! got {total} want {n_procs*per_proc}"
    assert res.skipped_corrupt == 0
    # each proc wrote its own file (per-pid isolation)
    assert len(list(cfg.dir.glob("*.json"))) == n_procs


# ---------------------------------------------------------------------------
# summarize_window shape
# ---------------------------------------------------------------------------

def test_summarize_window_aggregates_per_plugin(tmp_path):
    _fresh_agg()
    cfg = _cfg(tmp_path)
    for _ in range(10):
        fl.record_hook_fire(plugin_name="a", hook_name="post_tool_call", kwargs={}, cfg=cfg)
    for _ in range(2):
        fl.record_hook_fire(plugin_name="a", hook_name="pre_llm_call", kwargs={}, result={"context": "x"}, cfg=cfg)
    fl.flush(cfg)
    summ = fl.summarize_window(days=30, cfg=cfg)
    assert summ["plugins"]["a"]["fires"] == 12
    assert summ["plugins"]["a"]["decisions"] == {"noop": 10, "context": 2}
    assert set(summ["plugins"]["a"]["channels"]) == {"post_tool_call", "pre_llm_call"}


# ---------------------------------------------------------------------------
# Classifier unit coverage (unchanged semantics)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hook,result,err,expected", [
    ("post_tool_call", {"action": "block"}, None, "block"),
    ("transform_tool_result", "rewritten", None, "transform"),
    ("pre_llm_call", {"context": "x"}, None, "context"),
    ("pre_llm_call", "recalled", None, "context"),
    ("post_tool_call", None, None, "noop"),
    ("post_tool_call", None, RuntimeError("x"), "error"),
])
def test_classify_hook_decision(hook, result, err, expected):
    assert fl.classify_hook_decision(hook, result, err) == expected


@pytest.mark.parametrize("result,err,expected", [
    ({"action": "block"}, None, "block"),
    ({"args": {"command": "ls"}}, None, "transform"),
    ({"action": "allow"}, None, "allow"),
    (None, None, "noop"),
    (None, ValueError("x"), "error"),
])
def test_classify_middleware_decision(result, err, expected):
    assert fl.classify_middleware_decision("tool_request", {}, result, err) == expected
