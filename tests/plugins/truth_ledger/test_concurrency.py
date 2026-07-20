from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import types
from pathlib import Path


def _load_ledger_module():
    repo_root = Path(__file__).resolve().parents[3]
    plugin_dir = repo_root / "plugins" / "truth-ledger"
    mod_path = plugin_dir / "ledger.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.truth_ledger.ledger",
        mod_path,
        submodule_search_locations=[str(plugin_dir)],
    )
    if "hermes_plugins" not in __import__("sys").modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        __import__("sys").modules["hermes_plugins"] = ns
    if "hermes_plugins.truth_ledger" not in __import__("sys").modules:
        pkg = types.ModuleType("hermes_plugins.truth_ledger")
        pkg.__path__ = [str(plugin_dir)]
        __import__("sys").modules["hermes_plugins.truth_ledger"] = pkg
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _event(event_id: str, key: str, value) -> dict:
    return {
        "schema_version": 1,
        "event_id": event_id,
        "occurred_at": "2026-07-17T20:00:00Z",
        "operation": "assert",
        "fact_id": f"fact_{event_id.removeprefix('evt_')}",
        "supersedes": None,
        "fact": {"scope": "user", "kind": "preference", "subject": "u1", "key": key, "value": value},
        "evidence": {
            "type": "user_stated", "profile": "default", "platform": "cli",
            "session_id": "s1", "turn_id": "t1", "task_id": None,
            "speaker_id": "u1", "conversation_id": None, "thread_id": None,
        },
        "extraction": {
            "schema_name": "truth-ledger.fact-candidates.v1",
            "provider": "test", "model": "test", "prompt_version": 1,
        },
    }


def _worker(root: str, start: int, count: int, q):
    ledger_mod = _load_ledger_module()
    store = ledger_mod.LedgerStore(Path(root))
    indexed = 0
    duplicate = 0
    for i in range(start, start + count):
        event = _event(f"evt_{i}", f"k{i}", i)
        out = store.append_event(event=event, event_key=f"k-{i}")
        if out["status"] == "indexed":
            indexed += 1
        elif out["status"] == "duplicate":
            duplicate += 1

    race = _event("evt_race", "race", "same")
    out = store.append_event(event=race, event_key="race-key")
    q.put({"indexed": indexed, "duplicate": duplicate, "race_status": out["status"]})


def test_multiprocess_append_is_non_corrupt_and_idempotent(tmp_path):
    root = tmp_path
    q = mp.Queue()
    procs = [
        mp.Process(target=_worker, args=(str(root), 0, 50, q)),
        mp.Process(target=_worker, args=(str(root), 50, 50, q)),
        mp.Process(target=_worker, args=(str(root), 100, 50, q)),
        mp.Process(target=_worker, args=(str(root), 150, 50, q)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=20)
        assert p.exitcode == 0

    rows = [q.get(timeout=2) for _ in procs]
    race_indexed = sum(1 for r in rows if r["race_status"] == "indexed")
    assert race_indexed == 1

    ledger_file = root / "ledger" / "2026-07.jsonl"
    lines = [x for x in ledger_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 201

    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == 201
