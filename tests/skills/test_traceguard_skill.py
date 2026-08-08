from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "traceguard"
    / "scripts"
    / "traceguard.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("traceguard_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture() -> dict[str, object]:
    return {
        "expected_retained_facts": [
            {
                "fact_id": "TG-001",
                "evidence_chunk_id": "traceguard.txt:1-2",
                "text": "FACT:TG-001 retained child evidence.",
            }
        ],
        "expected_omitted_facts": [
            {
                "fact_id": "TG-002",
                "evidence_chunk_id": "traceguard.txt:3-4",
                "text": "FACT:TG-002 omitted evidence.",
            }
        ],
    }


def test_traceguard_accepts_claims_backed_by_manifest() -> None:
    mod = load_module()
    manifest = mod.build_manifest_from_fixture(_fixture())
    parent = {
        "result": {
            "retained_facts": [
                {
                    "fact_id": "TG-001",
                    "text": "retained child evidence",
                    "evidence_chunk_id": "traceguard.txt:1-2",
                }
            ]
        },
        "evidence_references": [
            {
                "chunk_id": "traceguard.txt:1-2",
                "supports_fact_ids": ["TG-001"],
                "quoted_evidence": "FACT:TG-001 retained child evidence.",
            }
        ],
    }

    result = mod.validate_parent_synthesis(
        evidence_manifest=manifest,
        parent_synthesis=parent,
    )

    assert result.accepted is True
    assert result.unsupported_claim_rate == 0.0
    assert len(result.accepted_claims) == 2
    assert result.rejected_claims == ()


def test_traceguard_rejects_omitted_fact_claims() -> None:
    mod = load_module()
    manifest = mod.build_manifest_from_fixture(_fixture())
    parent = {
        "result": {
            "observed_facts": [
                {
                    "fact_id": "TG-002",
                    "text": "omitted evidence",
                    "evidence_chunk_id": "traceguard.txt:3-4",
                }
            ]
        }
    }

    result = mod.validate_parent_synthesis(
        evidence_manifest=manifest,
        parent_synthesis=parent,
    )

    assert result.accepted is False
    assert result.unsupported_claim_rate == 1.0
    assert result.rejected_claims[0].reason == "unsupported_fact_id"


def test_traceguard_rejects_chunk_handles_without_fact_evidence() -> None:
    mod = load_module()
    manifest = mod.build_manifest_from_fixture(_fixture())
    parent = {
        "result": {"summary": "chunk was read"},
        "evidence_references": [
            {"chunk_id": "traceguard.txt:1-2", "claim": "read traceguard.txt:1-2"}
        ],
    }

    result = mod.validate_parent_synthesis(
        evidence_manifest=manifest,
        parent_synthesis=parent,
    )

    assert result.accepted is False
    assert result.rejected_claims[0].reason == "chunk_handle_without_fact"


def test_traceguard_rejects_evidence_handle_mismatch() -> None:
    mod = load_module()
    manifest = [
        {
            "fact_id": "TG-001",
            "chunk_id": "traceguard.txt:1-2",
            "text": "FACT:TG-001 retained child evidence.",
        }
    ]
    parent = {
        "result": {
            "facts": [
                {
                    "fact_id": "TG-001",
                    "text": "retained child evidence",
                    "evidence_chunk_id": "traceguard.txt:9-10",
                }
            ]
        }
    }

    result = mod.validate_parent_synthesis(
        evidence_manifest=manifest,
        parent_synthesis=parent,
    )

    assert result.accepted is False
    assert result.rejected_claims[0].reason == "evidence_handle_mismatch"


def test_allowed_evidence_manifest_normalization_is_canonical() -> None:
    mod = load_module()
    manifest = (
        {
            "evidence_chunk_id": "traceguard.txt:9-10",
            "fact_id": "TG-003",
            "extra": "ignored",
        },
        mod.TraceGuardEvidence(
            fact_id="TG-001",
            chunk_id="traceguard.txt:1-2",
            text="FACT:TG-001 retained child evidence.",
        ),
        {
            "fact_id": "TG-002",
            "chunk_id": "traceguard.txt:3-4",
            "text": "FACT:TG-002 retained child evidence.",
            "child_call_id": "child_0002",
        },
    )

    normalized = mod.normalize_allowed_evidence_manifest(manifest)

    assert normalized == (
        {
            "fact_id": "TG-001",
            "chunk_id": "traceguard.txt:1-2",
            "text": "FACT:TG-001 retained child evidence.",
            "child_call_id": None,
        },
        {
            "fact_id": "TG-002",
            "chunk_id": "traceguard.txt:3-4",
            "text": "FACT:TG-002 retained child evidence.",
            "child_call_id": "child_0002",
        },
        {
            "fact_id": "TG-003",
            "chunk_id": "traceguard.txt:9-10",
            "text": "",
            "child_call_id": None,
        },
    )


def test_validate_payload_accepts_supported_parent_claims() -> None:
    mod = load_module()
    result = mod.validate_payload(
        {
            "evidence_manifest": [
                {
                    "fact_id": "TG-001",
                    "evidence_chunk_id": "traceguard.txt:1-2",
                    "quoted_evidence": "FACT:TG-001 retained child evidence.",
                }
            ],
            "parent_synthesis": {
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
        }
    )

    assert result["success"] is True
    assert result["traceguard"]["accepted"] is True
    assert result["traceguard"]["unsupported_claim_rate"] == 0.0


def test_validate_payload_reports_bad_payload_shape() -> None:
    mod = load_module()

    assert mod.validate_payload([])["success"] is False
    assert (
        "evidence_manifest"
        in mod.validate_payload({"evidence_manifest": {}, "parent_synthesis": {}})["error"]
    )
    assert (
        "parent_synthesis"
        in mod.validate_payload({"evidence_manifest": [], "parent_synthesis": []})["error"]
    )


def test_cli_accepts_payload_file_and_exits_zero(tmp_path, capsys) -> None:
    mod = load_module()
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "evidence_manifest": [
                    {
                        "fact_id": "TG-001",
                        "chunk_id": "traceguard.txt:1-2",
                        "text": "FACT:TG-001 retained child evidence.",
                    }
                ],
                "parent_synthesis": {
                    "result": {
                        "facts": [
                            {
                                "fact_id": "TG-001",
                                "chunk_id": "traceguard.txt:1-2",
                                "text": "retained child evidence",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = mod.main(["--input", str(payload_path)])

    verdict = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert verdict["traceguard"]["accepted"] is True


def test_cli_exits_one_on_rejected_claims(tmp_path, capsys) -> None:
    mod = load_module()
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "evidence_manifest": [],
                "parent_synthesis": {
                    "result": {
                        "facts": [
                            {
                                "fact_id": "TG-404",
                                "chunk_id": "traceguard.txt:1-2",
                                "text": "unsupported claim",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = mod.main(["--input", str(payload_path)])

    verdict = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert verdict["traceguard"]["rejected_claims"][0]["reason"] == "unsupported_fact_id"


def test_cli_exits_two_on_malformed_payload(tmp_path, capsys) -> None:
    mod = load_module()
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("{not json", encoding="utf-8")

    exit_code = mod.main(["--input", str(payload_path)])

    verdict = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert verdict["success"] is False
