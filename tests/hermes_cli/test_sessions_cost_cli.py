"""CLI integration tests for `hermes sessions cost`."""

import sys

import pytest

from hermes_cli.session_cost import CACHE_HIT_WARN_THRESHOLD_PCT

_SESSIONS = [
    {
        "id": "20260701_000000_aaaaaa",
        "source": "cli",
        "model": "deepseek-v4-flash",
        "title": "api debugging",
        "input_tokens": 1_000_000,
        "cache_read_tokens": 900_000,
        "cache_write_tokens": 100_000,
        "output_tokens": 50_000,
        "estimated_cost_usd": 0.31,
    },
    {
        "id": "20260702_000000_bbbbbb",
        "source": "cli",
        "model": "deepseek-v4-flash",
        "title": "cache busted",
        "input_tokens": 1_000_000,
        "cache_read_tokens": 100_000,
        "cache_write_tokens": 500_000,
        "output_tokens": 50_000,
        "estimated_cost_usd": 0.45,
    },
    {
        "id": "20260703_000000_cccccc",
        "source": "telegram",
        "model": "unknown",
        "title": "no usage",
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": None,
    },
]


def _run(monkeypatch, capsys, argv_tail, rows=_SESSIONS, detail=None):
    """Run `hermes sessions cost <argv_tail>` against a FakeDB."""
    import hermes_cli.main as main_mod
    import hermes_state
    from hermes_cli import config as cfg_mod

    class FakeDB:
        def list_sessions_rich(self, source=None, exclude_sources=None, limit=20):
            return list(rows)[:limit]

        def resolve_session_id(self, session_id):
            for s in rows:
                if s["id"].startswith(session_id):
                    return s["id"]
            return None

        def get_session_rich_row(self, session_id):
            for s in rows:
                if s["id"] == session_id:
                    return dict(s)
            return None

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(cfg_mod, "load_config_readonly", lambda: {"cost": {"cache_hit_ratio": 0.10}})
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "cost", *argv_tail])
    main_mod.main()
    return capsys.readouterr().out


def test_cost_table_columns_and_values(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, [])
    assert "ID" in out
    assert "Hit%" in out
    assert "Input" in out
    assert "CRead" in out
    assert "CWrite" in out
    assert "Output" in out
    assert "Cost" in out
    assert "No-Cache" in out
    assert "Saved" in out
    # Session 1: hit = 900k / 2M = 45.0%
    assert "45.0%" in out
    assert "20260701_000000_aaaaaa" in out
    # Session 2: hit = 100k / 1.6M = 6.2%
    assert "6.2%" in out
    assert "20260702_000000_bbbbbb" in out
    # Session 3 has no usage -> dash placeholders
    assert "20260703_000000_cccccc" in out


def test_cost_table_flags_low_cache_hit(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, [])
    # Both data sessions are below the 70% warn threshold and must carry the
    # marker; the no-usage session must not.
    assert out.count("⚠") == 2
    assert "20260702_000000_bbbbbb" in out


def test_cost_table_counterfactual_footer(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, [])
    assert "Counterfactual: cache reads billed at 10%" in out
    assert "cost.cache_hit_ratio" in out


def test_cost_table_no_sessions(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, [], rows=[])
    assert "No sessions found." in out


def test_cost_detail_single_session(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, ["--session", "20260702_000000_bbbbbb"])
    assert "Session 20260702_000000_bbbbbb" in out
    assert "Title:" in out
    assert "cache busted" in out
    assert "Source:" in out
    assert "Model:" in out
    assert "deepseek-v4-flash" in out
    assert "Input tokens:" in out
    assert "1,000,000" in out
    assert "Cache read:" in out
    assert "100,000" in out
    assert "Cache write:" in out
    assert "500,000" in out
    assert "Output tokens:" in out
    assert "Cache hit:" in out
    assert "6.2%" in out
    assert "Estimated cost:" in out
    assert "$0.45" in out
    assert "Cost if 0% cache:" in out
    assert "Estimated savings:" in out
    assert "Counterfactual assumes cache reads billed at 10%" in out


def test_cost_detail_resolves_unique_prefix(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, ["--session", "20260702"])
    assert "Session 20260702_000000_bbbbbb" in out


def test_cost_detail_unknown_session(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, ["--session", "missing-prefix"])
    assert "Session 'missing-prefix' not found." in out


def test_cost_detail_warn_note_on_low_hit(monkeypatch, capsys):
    out = _run(monkeypatch, capsys, ["--session", "20260702"])
    assert "⚠" in out
    assert f"below {CACHE_HIT_WARN_THRESHOLD_PCT:.0f}%" in out
    assert "prompt prefix may have been invalidated" in out
