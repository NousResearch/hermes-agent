"""Discord-specific semantic guard for the canonical 42-row parity ledger."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
sys.path.insert(0, str(SCRIPTS_CI))

from validate_feature_parity_ledger import validate_ledger  # noqa: E402

LEDGER_PATH = (
    REPO_ROOT / "docs" / "architecture" / "feature-parity" / "discord.json"
)


def _ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _by_id(document: dict) -> dict[str, dict]:
    return {row["id"]: row for row in document["capabilities"]}


def test_discord_ledger_passes_generic_contract() -> None:
    assert validate_ledger(_ledger()) == []


def test_discord_contract_has_exactly_42_canonical_rows() -> None:
    document = _ledger()
    assert len(document["capabilities"]) == 42
    assert [row["id"] for row in document["capabilities"]] == [
        "M1", "M2", "M3", "M4", "M5", "M6", "M7",
        "T1", "T2", "T3", "T4", "T5",
        "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9",
        "V1", "V2", "V3", "V4", "V5", "V6",
        "A1", "A2", "A3", "A4", "A5", "A6",
        "W1", "W2", "W3", "W4", "W5",
        "R1", "R2", "R3", "R4",
    ]


def test_w_lane_cannot_be_semantically_reassigned() -> None:
    rows = _by_id(_ledger())
    assert (rows["W1"]["name"], rows["W1"]["product_state"]) == (
        "Discord-native webhook operations",
        "rejected",
    )
    assert rows["W2"]["name"] == "Generic Hermes webhook → Discord delivery"
    assert rows["W3"]["name"] == "Multiplex routing acceptance matrix"
    assert rows["W4"]["name"] == "Proactive/home/cron delivery"
    assert (rows["W5"]["name"], rows["W5"]["product_state"]) == (
        "OAuth2 authorization-code flow",
        "rejected",
    )


def test_rejected_native_webhook_row_has_no_code_path() -> None:
    w1 = _by_id(_ledger())["W1"]
    assert w1["delivery_state"] == "gap"
    assert w1["implementation_paths"] == []
    assert all("webhooks.py" not in path for path in w1["implementation_paths"])


def test_main_snapshot_does_not_claim_packet_delivery() -> None:
    document = _ledger()
    assert document["snapshot"]["tools_discord_api_on_main"] is False
    assert document["snapshot"]["discord_adapter_bytes"] == 475891
    assert not any(
        row["delivery_state"] in {"on_main_unverified", "released"}
        for row in document["capabilities"]
    )


def test_every_active_candidate_names_its_gap_class() -> None:
    for row in _ledger()["capabilities"]:
        if row["delivery_state"] == "candidate_blocked":
            assert row.get("blocker")
        if row["delivery_state"] == "candidate_unwired":
            assert row.get("wiring_gap")


def test_only_one_authoritative_m3_publication_remains() -> None:
    m3 = _by_id(_ledger())["M3"]
    authoritative = [
        publication
        for publication in m3["publications"]
        if publication["role"] == "authoritative"
    ]
    superseded = [
        publication
        for publication in m3["publications"]
        if publication["role"] == "superseded"
    ]
    assert len(authoritative) == 1
    assert authoritative[0]["kind"] == "pull_request"
    assert authoritative[0]["number"] == 89405
    assert authoritative[0]["role"] == "authoritative"
    assert authoritative[0]["state"] == "open"
    assert any(
        publication["number"] == 86419
        and publication["role"] == "superseded"
        and publication["state"] == "closed"
        for publication in superseded
    )


def test_status_counts_are_an_explicit_non_completion_receipt() -> None:
    counts = Counter(
        row["delivery_state"] for row in _ledger()["capabilities"]
    )
    assert counts["released"] == 0
    assert counts["on_main_unverified"] == 0
    assert counts["candidate_open"] == 0
    assert counts["candidate_unwired"] == 1
    assert counts["candidate_blocked"] == 10
    assert counts["gap"] == 31
    assert sum(counts.values()) == 42


def test_closed_packet_authorities_are_historical_not_active() -> None:
    document = _ledger()
    closed = set(document["snapshot"]["closed_packet_authority_prs"])
    for row in document["capabilities"]:
        for publication in row.get("publications", []):
            if publication.get("number") in closed:
                assert publication.get("role") != "authoritative"
                assert publication.get("state") == "closed"


def test_live_successors_own_i4_and_r3() -> None:
    rows = _by_id(_ledger())
    i4 = [p for p in rows["I4"]["publications"] if p["role"] == "authoritative"]
    r3 = [p for p in rows["R3"]["publications"] if p["role"] == "authoritative"]
    assert i4 == [{"kind": "pull_request", "number": 81388, "role": "authoritative", "state": "open"}]
    assert r3 == [{"kind": "pull_request", "number": 87700, "role": "authoritative", "state": "open"}]
