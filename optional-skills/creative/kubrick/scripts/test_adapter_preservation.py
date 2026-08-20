#!/usr/bin/env python3
"""Provider semantic preservation and injected-loss contract tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from adapt_provider import (  # noqa: E402
    PRESERVATION_FIELDS,
    adapt,
    apply_preservation_policy,
    preservation_report,
)

SCHEMA = json.loads(
    (ROOT / "schemas/provider-preservation-report.schema.json").read_text(encoding="utf-8")
)


def packet() -> dict:
    negative = ["no named esoterica", "no continuity reset"]
    return {
        "schema_version": "1.0.0",
        "adapter_id": "kubrick-generic-adapter-v1",
        "provider": "generic",
        "source_graph_id": "graph-preservation-test",
        "mode": "storyboard",
        "shared_constraints": {
            "visual_identity": ["doorway threshold geometry", "cold metal badge"],
            "continuity_rules": ["preserve graph identity", "preserve residue"],
            "prohibited_resets": ["badge ownership without transfer"],
            "negative_constraints": negative,
        },
        "frames": [
            {
                "frame_id": "frame-001",
                "prompt": "doorway threshold geometry; cold metal badge; motif ownership badge: supervisor; residue persists: badge scratch; required object: cracked badge",
                "state_constraints": ["node badge: held", "node door: closed"],
                "continuity_from_previous": [],
                "provider_parameters": {},
            },
            {
                "frame_id": "frame-002",
                "prompt": "doorway threshold geometry; cold metal badge; motif ownership badge: junior; residue persists: badge scratch; required object: cracked badge",
                "state_constraints": ["node badge: transferred", "node door: open"],
                "continuity_from_previous": [
                    "preserve graph identity graph-preservation-test",
                    "preserve prior motif ownership badge: supervisor",
                    "do not remove residue: badge scratch",
                ],
                "provider_parameters": {},
            },
        ],
        "private_state_policy": {
            "graph_mutation_allowed": False,
            "pattern_links_exposed": False,
            "lexicon_links_exposed": False,
        },
        "validation": {"status": "VALID", "errors": [], "warnings": []},
    }


def validate_schema(report: dict) -> None:
    assert set(SCHEMA["required"]) <= set(report)
    assert set(report) <= set(SCHEMA["properties"])
    try:
        import jsonschema
    except ImportError:
        return
    jsonschema.validate(report, SCHEMA)


def remove_text(result: dict, text: str) -> None:
    for frame in result["frames"]:
        frame["prompt"] = frame["prompt"].replace(text, "")


def test_all_providers_preserve_critical_invariants() -> None:
    source = packet()
    before = copy.deepcopy(source)
    for provider in ("generic", "grok-imagine", "flux", "sd3", "midjourney"):
        result = adapt(source, provider)
        assert result["validation"]["status"] == "VALID", (provider, result)
        report = result["preservation_report"]
        validate_schema(report)
        assert report["critical_invariants_preserved"] is True
        assert report["losses"] == []
        assert all(report[field] is True for field in PRESERVATION_FIELDS)
        assert preservation_report(source, result) == report
    assert source == before, "provider adaptation mutated the neutral packet"


def test_injected_losses_are_detected() -> None:
    source = packet()
    baseline = adapt(source, "flux")
    mutations = {
        "identity_preserved": lambda result: result.__setitem__("source_graph_id", "wrong-graph"),
        "required_objects_preserved": lambda result: remove_text(result, "required object: cracked badge"),
        "ownership_preserved": lambda result: remove_text(result, "motif ownership badge: supervisor"),
        "geometry_preserved": lambda result: remove_text(result, "doorway threshold geometry"),
        "state_change_preserved": lambda result: result["frames"][1].__setitem__("state_constraints", []),
        "residue_preserved": lambda result: remove_text(result, "residue persists: badge scratch"),
        "continuity_preserved": lambda result: result["frames"][1].__setitem__("continuity_lock", []),
        "negative_constraints_preserved": lambda result: result["frames"][0].__setitem__("negative_constraints", []),
    }
    for field, mutate in mutations.items():
        damaged = copy.deepcopy(baseline)
        mutate(damaged)
        report = preservation_report(source, damaged)
        assert report[field] is False, (field, report)
        assert report["critical_invariants_preserved"] is False
        assert report["losses"]
        governed = apply_preservation_policy(source, damaged)
        assert governed["validation"]["status"] == "INVALID"
        assert any(error.startswith("PROVIDER_LOSS:") for error in governed["validation"]["errors"])


def main() -> None:
    test_all_providers_preserve_critical_invariants()
    test_injected_losses_are_detected()
    print("provider semantic preservation: PASS")


if __name__ == "__main__":
    main()
