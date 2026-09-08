"""Regression tests for `hermes sessions` model-pin audit and repair.

Covers the CLI half of a model/provider switch (issue #96745):
- `sessions list --model/--provider` — audit filters over the EFFECTIVE
  identity (the model_config JSON wins on resume), so they catch the desync
  class where the model column looks current but the JSON still routes to an
  outdated provider.
- `sessions repair --model TARGET [--provider TARGET]` — bulk re-pin of
  per-session overrides with backup-first, lineage preservation, and
  cron-run/aux-model safety.
- `cron list` shows the per-job model/provider pin so drift is visible from
  the CLI instead of requiring a jobs.json read.
"""

import json
import time
from argparse import Namespace

import pytest

import hermes_cli.sessions_cmd as sc

TARGET_MODEL = "deepseek/deepseek-v4-flash-0731"
TARGET_PROVIDER = "nous"


def _args(action, **kw):
    base = dict(
        sessions_action=action,
        session_id=None, title=None, yes=True, source=None, path=None,
        from_source=None, dry_run=False, older_than=None, newer_than=None,
        before=None, after=None, limit=50, workspace=None,
        model=None, provider=None, all=False,
        check_only=False, no_backup=False,
    )
    base.update(kw)
    return Namespace(**base)


def _init_db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB

    return SessionDB()  # resolves to tmp_path/state.db via DEFAULT_DB_PATH


def _seed(db, sid, *, model=None, billing_provider=None, mc=None, source="cli"):
    db._conn.execute(
        "INSERT INTO sessions (id, source, started_at, title, model, "
        "billing_provider, billing_base_url, model_config) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            sid, source, time.time(), f"title-{sid}", model,
            billing_provider, None,
            json.dumps(mc) if mc is not None else None,
        ),
    )
    db._conn.commit()


def _row(db, sid):
    cur = db._conn.execute(
        "SELECT id, model, billing_provider, billing_base_url, model_config "
        "FROM sessions WHERE id=?",
        (sid,),
    )
    return cur.fetchone()


# ── sessions list --model/--provider audit filters ──────────────────


def test_list_model_filter_matches_json_pinned_model(tmp_path, monkeypatch, capsys):
    db = _init_db(tmp_path, monkeypatch)
    _seed(db, "alpha", model="old-model",
          mc={"model": "glm-5.3-flash", "provider": "empero"})
    _seed(db, "beta", model=TARGET_MODEL,
          mc={"model": TARGET_MODEL, "provider": "nous"})
    db.close()

    rc = sc.cmd_sessions(_args("list", model="glm-5.3"))
    out = capsys.readouterr().out
    assert rc in (None, 0)
    assert "alpha" in out
    assert "beta" not in out


def test_list_provider_filter_catches_desync_class(tmp_path, monkeypatch, capsys):
    db = _init_db(tmp_path, monkeypatch)
    # model column says TARGET (looks current) but the JSON still pins
    # empero — the desync class that resume would honor.
    _seed(db, "alpha", model=TARGET_MODEL, billing_provider="nous",
          mc={"model": "glm-5.3-flash", "provider": "empero",
              "base_url": "https://free.empero.org/v1"})
    _seed(db, "beta", model=TARGET_MODEL, billing_provider="nous",
          mc={"model": TARGET_MODEL, "provider": "nous"})
    db.close()

    rc = sc.cmd_sessions(_args("list", provider="empero"))
    out = capsys.readouterr().out
    assert rc in (None, 0)
    assert "alpha" in out
    assert "beta" not in out

    capsys.readouterr()
    rc = sc.cmd_sessions(_args("list", provider="nous"))
    out = capsys.readouterr().out
    assert rc in (None, 0)
    assert "beta" in out
    assert "alpha" not in out


# ── sessions repair --model/--provider (bulk re-pin) ─────────────────


def test_repair_rewrites_deviating_rows_preserves_lineage(
    tmp_path, monkeypatch, capsys
):
    db = _init_db(tmp_path, monkeypatch)
    _seed(db, "alpha", model="hy3-free", billing_provider="opencode-free",
          mc={"model": "hy3-free", "provider": "opencode-free"})
    _seed(db, "beta", model=TARGET_MODEL, billing_provider="nous",
          mc={"model": "glm-5.3-flash", "provider": "empero",
              "base_url": "https://free.empero.org/v1",
              "api_mode": "chat_completions",
              "reasoning_config": {"enabled": True, "effort": "high"},
              "_branched_from": "alpha"})
    _seed(db, "cron_123", model="old", billing_provider="x",
          mc={"model": "old"})  # cron run record — must be untouched
    _seed(db, "gamma", model=TARGET_MODEL, billing_provider="nous",
          mc={"model": TARGET_MODEL, "provider": "nous"})  # already on target
    db.close()

    rc = sc.cmd_sessions(
        _args("repair", model=TARGET_MODEL, provider=TARGET_PROVIDER)
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "backup:" in out
    assert "rewritten:   2" in out

    db = _init_db(tmp_path, monkeypatch)
    try:
        a = _row(db, "alpha")
        b = _row(db, "beta")
        c = _row(db, "cron_123")
        g = _row(db, "gamma")
    finally:
        db.close()

    assert a["model"] == TARGET_MODEL
    assert a["billing_provider"] == TARGET_PROVIDER
    mc_a = json.loads(a["model_config"])
    assert mc_a["model"] == TARGET_MODEL and mc_a["provider"] == TARGET_PROVIDER

    assert b["model"] == TARGET_MODEL
    assert b["billing_provider"] == TARGET_PROVIDER
    mc_b = json.loads(b["model_config"])
    assert mc_b["model"] == TARGET_MODEL and mc_b["provider"] == TARGET_PROVIDER
    assert "base_url" not in mc_b and "api_mode" not in mc_b
    assert mc_b["_branched_from"] == "alpha"
    assert mc_b["reasoning_config"] == {"enabled": True, "effort": "high"}

    # cron run record untouched
    assert c["model"] == "old"
    # already-on-target row untouched
    assert g["model"] == TARGET_MODEL
    assert json.loads(g["model_config"]) == {
        "model": TARGET_MODEL, "provider": "nous"
    }


def test_repair_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    db = _init_db(tmp_path, monkeypatch)
    _seed(db, "alpha", model="hy3-free", billing_provider="opencode-free",
          mc={"model": "hy3-free", "provider": "opencode-free"})
    db.close()

    rc = sc.cmd_sessions(
        _args("repair", model=TARGET_MODEL, provider=TARGET_PROVIDER,
              dry_run=True)
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry run" in out
    assert "would rewrite alpha" in out

    db = _init_db(tmp_path, monkeypatch)
    try:
        a = _row(db, "alpha")
    finally:
        db.close()
    assert a["model"] == "hy3-free"
    assert json.loads(a["model_config"])["provider"] == "opencode-free"


def test_repair_default_skips_null_and_matching(tmp_path, monkeypatch, capsys):
    db = _init_db(tmp_path, monkeypatch)
    _seed(db, "alpha", model=None, billing_provider=None, mc=None)
    _seed(db, "beta", model=TARGET_MODEL, billing_provider="nous",
          mc={"model": TARGET_MODEL, "provider": "nous"})
    db.close()

    rc = sc.cmd_sessions(
        _args("repair", model=TARGET_MODEL, provider=TARGET_PROVIDER)
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "nothing to repair" in out

    db = _init_db(tmp_path, monkeypatch)
    try:
        a = _row(db, "alpha")
        b = _row(db, "beta")
    finally:
        db.close()
    assert a["model"] is None and a["model_config"] is None
    assert b["model"] == TARGET_MODEL


def test_repair_all_pins_null_rows(tmp_path, monkeypatch, capsys):
    db = _init_db(tmp_path, monkeypatch)
    _seed(db, "alpha", model=None, billing_provider=None, mc=None)
    db.close()

    rc = sc.cmd_sessions(
        _args("repair", model=TARGET_MODEL, provider=TARGET_PROVIDER, all=True)
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "rewritten:   1" in out

    db = _init_db(tmp_path, monkeypatch)
    try:
        a = _row(db, "alpha")
    finally:
        db.close()
    assert a["model"] == TARGET_MODEL
    assert json.loads(a["model_config"])["provider"] == TARGET_PROVIDER


def test_repair_model_only_keeps_provider_routing(tmp_path, monkeypatch, capsys):
    db = _init_db(tmp_path, monkeypatch)
    _seed(db, "alpha", model="hy3-free", billing_provider="opencode-free",
          mc={"model": "hy3-free", "provider": "opencode-free"})
    db.close()

    rc = sc.cmd_sessions(_args("repair", model=TARGET_MODEL))
    out = capsys.readouterr().out
    assert rc == 0

    db = _init_db(tmp_path, monkeypatch)
    try:
        a = _row(db, "alpha")
    finally:
        db.close()
    assert a["model"] == TARGET_MODEL
    mc_a = json.loads(a["model_config"])
    assert mc_a["model"] == TARGET_MODEL
    assert mc_a["provider"] == "opencode-free"  # provider routing untouched
    assert a["billing_provider"] == "opencode-free"


def test_repair_no_backup_skips_copy(tmp_path, monkeypatch, capsys):
    db = _init_db(tmp_path, monkeypatch)
    _seed(db, "alpha", model="hy3-free", billing_provider="opencode-free",
          mc={"model": "hy3-free", "provider": "opencode-free"})
    db.close()

    rc = sc.cmd_sessions(
        _args("repair", model=TARGET_MODEL, provider=TARGET_PROVIDER,
              no_backup=True)
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "backup:" not in out
    assert not list(tmp_path.glob("state.db.model-reset-backup-*"))


# ── cron list model/provider pin display ────────────────────────────


def test_cron_list_shows_model_provider_pins(tmp_path, monkeypatch, capsys):
    jobs_dir = tmp_path / "cron"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = jobs_dir / "jobs.json"
    jobs_file.write_text(json.dumps({"jobs": [
        {
            "id": "pinned1", "name": "pinned job", "enabled": True,
            "schedule": {"kind": "cron", "expr": "0 9 * * *"},
            "model": TARGET_MODEL, "provider": TARGET_PROVIDER,
            "deliver": ["local"],
        },
        {
            "id": "loose1", "name": "loose job", "enabled": True,
            "schedule": {"kind": "interval", "minutes": 30},
            "model": None, "provider": None,
            "deliver": ["local"],
        },
    ]}))
    monkeypatch.setattr("cron.jobs.JOBS_FILE", jobs_file)

    from hermes_cli.cron import cron_list

    cron_list(show_all=True)
    out = capsys.readouterr().out
    assert f"Model:     {TARGET_MODEL}" in out
    assert f"Provider: {TARGET_PROVIDER}" in out
    assert "Model:     (profile default)" in out
    assert "Provider: (profile default)" in out
