"""The serve handler ledger has one row per live serve handler and is regenerated with the code."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _generator():
    spec = importlib.util.spec_from_file_location("gen_serve_handler_ledger",
                                                  ROOT / "scripts" / "gen_serve_handler_ledger.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_ledger_lists_every_live_handler():
    """A route or RPC method added or removed without regenerating the ledger is named here."""
    gen = _generator()
    facts = gen._facts()
    live = {row["handler"] for row in gen.rows(facts)}
    committed = gen.OUT.read_text(encoding="utf-8-sig")
    listed = [row["handler"] for row in json.loads(committed)["handlers"]]
    assert len(listed) == len(set(listed))
    assert sorted(live - set(listed)) == [], "handlers with no ledger row: run scripts/gen_serve_handler_ledger.py"
    assert sorted(set(listed) - live) == [], "ledger rows for removed handlers: run scripts/gen_serve_handler_ledger.py"
    assert committed == gen.render(facts), "serve-handler-ledger.json is stale: run scripts/gen_serve_handler_ledger.py"


def test_rpc_rows_follow_the_ws_dispatch_tables():
    """Served verbs are authority-owned; a legacy-only verb is a fallback or refused, never decided."""
    gen = _generator()
    facts = {"routes": [], "legacy": ["pet.get", "session.title", "session.list"], "served": ["session.list"],
             "process_global": [], "session_namespaces": ["session."], "session_fallback_allowed": []}
    by_name = {row["handler"]: row for row in gen.rows(facts)}
    assert by_name["session.list"]["decision"] == "authority"
    assert by_name["pet.get"]["current_path"].startswith("legacy fallback")
    assert by_name["session.title"]["current_path"].startswith("authority refuses (-32601)")
    assert {by_name[m]["decision"] for m in ("pet.get", "session.title")} == {"undecided"}
    assert {row["decision"] for row in by_name.values()} <= set(gen.DECISION_VALUES)
