from __future__ import annotations

import json

from traceguard_tool import traceguard_validate


def test_traceguard_tool_accepts_supported_parent_claims() -> None:
    result = json.loads(
        traceguard_validate(
            evidence_manifest=[
                {
                    "fact_id": "TG-001",
                    "evidence_chunk_id": "traceguard.txt:1-2",
                    "quoted_evidence": "FACT:TG-001 retained child evidence.",
                }
            ],
            parent_synthesis={
                "result": {
                    "observed_facts": [
                        {
                            "fact_id": "TG-001",
                            "evidence_chunk_id": "traceguard.txt:1-2",
                            "text": "retained child evidence",
                        }
                    ]
                }
            },
        )
    )

    assert result["success"] is True
    assert result["traceguard"]["accepted"] is True
    assert result["traceguard"]["unsupported_claim_rate"] == 0.0


def test_traceguard_tool_rejects_unsupported_parent_claims() -> None:
    result = json.loads(
        traceguard_validate(
            evidence_manifest=[
                {
                    "fact_id": "TG-001",
                    "chunk_id": "traceguard.txt:1-2",
                    "text": "FACT:TG-001 retained child evidence.",
                }
            ],
            parent_synthesis={
                "result": {
                    "facts": [
                        {
                            "fact_id": "TG-002",
                            "evidence_chunk_id": "traceguard.txt:3-4",
                            "text": "omitted evidence",
                        }
                    ]
                }
            },
        )
    )

    assert result["success"] is True
    assert result["traceguard"]["accepted"] is False
    assert result["traceguard"]["rejected_claims"][0]["reason"] == "unsupported_fact_id"


def test_traceguard_tool_reports_bad_payload_shape() -> None:
    result = json.loads(
        traceguard_validate(evidence_manifest={}, parent_synthesis=[])
    )

    assert result["success"] is False
    assert (
        "evidence_manifest" in result["error"]
        or "parent_synthesis" in result["error"]
    )
