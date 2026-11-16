"""Tests: spawn_tree.* JSON-RPC handlers (tui_gateway/methods_session.py).

A parseable-but-non-object snapshot file must not wedge ``spawn_tree.list``
(the legacy per-file scan called ``raw.get`` outside its suppress guard) and
must not satisfy the ``spawn_tree.load`` result contract.
"""

import json

import tui_gateway.server as srv


def test_spawn_tree_list_survives_non_dict_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = srv._spawn_tree_session_dir("sess-x")
    (d / "bad.json").write_text('"not a snapshot"', encoding="utf-8")
    good = d / "good.json"
    good.write_text(json.dumps({"session_id": "sess-x", "label": "ok"}), encoding="utf-8")

    envelope = srv._methods["spawn_tree.list"](1, {"session_id": "sess-x"})
    assert "error" not in envelope, envelope
    entries = envelope["result"]["entries"]
    labels = {e["label"] for e in entries}
    assert "ok" in labels and len(entries) == 2  # bad file degrades to a fallback entry


def test_spawn_tree_load_rejects_non_dict_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = srv._spawn_tree_session_dir("sess-y")
    bad = d / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")

    envelope = srv._methods["spawn_tree.load"](1, {"path": str(bad)})
    assert "error" in envelope
