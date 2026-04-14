from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from agent.knowledge_lanes import KnowledgeLaneStore


def _load_script(name: str):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_knowledge_lane_governance_bridge_script_exports_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = KnowledgeLaneStore()
    draft = store.add_draft(
        title="Draft bridge note",
        body="Bridge this into the governed promotion flow.",
        source="chat:user",
        provenance={"session_id": "sess-2"},
    )
    module = _load_script("knowledge_lane_governance_bridge")

    exit_code = module.main(
        [
            "--id",
            draft["id"],
            "--lane",
            "draft",
            "--target-surface",
            "memory",
            "--target-path",
            "memory/governed-knowledge.md",
            "--repo-root",
            "/home/user/.hermes/hermes-agent",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["promotion_id"].startswith("prom_")
    assert payload["ingestion_package_path"].endswith(".json")
