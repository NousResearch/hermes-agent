"""``hermes sessions cost`` over a real store: hit %, savings, and unpriced sessions."""

from argparse import Namespace

import pytest

from hermes_cli import sessions_cmd
from hermes_cli.session_cost import cache_hit_ratio_from_config, session_cost_breakdown


@pytest.fixture
def db(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", dict)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("priced", "cli")
    db.update_token_counts("priced", input_tokens=100_000, cache_read_tokens=800_000,
                           cache_write_tokens=100_000, output_tokens=0, estimated_cost_usd=1.0,
                           cost_status="estimated")
    db.create_session("unpriced", "cli")
    # Unknown pricing: the writer persists amount None as estimated_cost_usd = 0.
    db.update_token_counts("unpriced", input_tokens=50_000, output_tokens=1_000,
                           estimated_cost_usd=None, cost_status="unknown")
    yield db
    db.close()


def _cost(db, capsys, **kw):
    sessions_cmd._cmd_cost(db, Namespace(session=None, limit=20, source=None, **kw))
    return {line.split()[0]: line.split() for line in capsys.readouterr().out.splitlines()
            if line.startswith(("priced", "unpriced"))}


def test_counterfactual_reprices_cache_reads_at_the_cold_rate(db, capsys):
    rows = _cost(db, capsys)
    # 80% hit; all cost is input-side; savings = 1.0 * 0.8 * (1/0.1 - 1) = 7.20
    assert rows["priced"][1:2] == ["80.0%"]
    assert rows["priced"][6:9] == ["$1.00", "$8.20", "$7.20"]


def test_unpriced_session_is_not_reported_as_free(db, capsys):
    assert db.get_session("unpriced")["estimated_cost_usd"] == 0  # what the store holds
    rows = _cost(db, capsys)
    assert rows["unpriced"][6:9] == ["n/a", "n/a", "n/a"]
    b = session_cost_breakdown({"cache_read_tokens": 10, "estimated_cost_usd": 0.0, "cost_status": "included"})
    assert (b["estimated_cost"], b["savings"]) == (0.0, 0.0)  # a subscription zero stays a real zero
    assert cache_hit_ratio_from_config({"cost": {"cache_hit_ratio": 5}}) == 0.10
