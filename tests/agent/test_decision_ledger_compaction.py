from types import SimpleNamespace
from agent.conversation_compression import _fold_decision_ledger


def test_fold_decision_ledger_replaces_stale_handoff_with_verbatim_entries():
    agent = SimpleNamespace(session_id="session", _session_db=SimpleNamespace(get_decision_ledger_entries=lambda _: [{"kind": "denial", "text": "Do not delete production data.", "turn_id": "turn-7"}, {"kind": "correction", "text": "Use Postgres instead.", "turn_id": "turn-8"}]))
    compressed = [{"role": "user", "content": "[DECISION LEDGER — VERBATIM]\n- stale\n[END DECISION LEDGER]", "_decision_ledger_synthetic": True}]
    _fold_decision_ledger(agent, compressed)
    assert compressed == [{"role": "user", "content": "[DECISION LEDGER — VERBATIM]\n- denial (turn-7): Do not delete production data.\n- correction (turn-8): Use Postgres instead.\n[END DECISION LEDGER]", "_decision_ledger_synthetic": True}]
