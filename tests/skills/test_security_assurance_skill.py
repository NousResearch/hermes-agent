"""Tests for the security-assurance skill.

The cases follow the app.mithril.fund assurance, mobile, and operations
review tests. The script must keep the same decisions.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "security" / "security-assurance"
SCRIPT = SKILL_DIR / "scripts" / "review.py"


@pytest.fixture(scope="module")
def review_mod():
    spec = importlib.util.spec_from_file_location("security_assurance_review", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _yes_transfer(review_mod) -> dict:
    return {control["id"]: "yes" for control in review_mod.TRANSFER_CONTROLS}


def test_ppap_is_replaced_even_when_all_answers_are_positive(review_mod) -> None:
    yes = _yes_transfer(review_mod)
    assert review_mod.review_transfer("password-zip", yes)["decision"] == "replace-ppap"
    assert review_mod.review_transfer("managed-link", yes)["decision"] == "planned-only"
    incomplete = dict(yes)
    incomplete["authentication"] = "unknown"
    assert review_mod.review_transfer("managed-link", incomplete)["decision"] == "incomplete-plan"
    assert review_mod.review_transfer("email-attachment", yes)["decision"] == "review-attachment"


def test_password_zip_stays_rejected_when_answers_are_unknown(review_mod) -> None:
    assert review_mod.review_transfer("password-zip", {})["decision"] == "replace-ppap"


def test_inventory_has_an_evidence_floor(review_mod) -> None:
    now = 1000000
    roster = [
        {"id": "laptop-1", "platform": "macos", "owner": "IT"},
        {"id": "phone-2", "platform": "ios", "owner": "IT"},
    ]
    missing_report = review_mod.reconcile_devices(roster, [], now, 1000)
    missing_roster = review_mod.reconcile_devices(
        [],
        [{"id": "laptop-1", "platform": "macos", "reported_at": now}],
        now,
        1000,
    )
    assert missing_report["state"] == "unmeasured"
    assert missing_report["reason"] == "reports-not-provided"
    assert missing_report["coverage"] is None
    assert missing_roster["reason"] == "roster-not-provided"


def test_inventory_distinguishes_missing_stale_unexpected_and_boundary(review_mod) -> None:
    result = review_mod.reconcile_devices(
        [
            {"id": "a", "platform": "macos"},
            {"id": "b", "platform": "ios"},
            {"id": "c", "platform": "linux"},
        ],
        [
            {"id": "a", "platform": "macos", "reported_at": 900},
            {"id": "b", "platform": "ios", "reported_at": 899},
            {"id": "x", "platform": "windows", "reported_at": 1000},
        ],
        1000,
        100,
    )
    assert result["state"] == "reconciled"
    assert result["counts"] == {"matched": 1, "stale": 1, "missing": 1, "unexpected": 1}
    assert result["coverage"] == pytest.approx(1 / 3)


def test_inventory_rejects_ambiguous_identities(review_mod) -> None:
    with pytest.raises(ValueError, match="duplicate device id"):
        review_mod.reconcile_devices(
            [{"id": "a", "platform": "macos"}, {"id": "a", "platform": "macos"}],
            [{"id": "a", "platform": "macos", "reported_at": 1000}],
            1000,
            100,
        )
    mismatch = review_mod.reconcile_devices(
        [{"id": "a", "platform": "macos"}],
        [{"id": "a", "platform": "linux", "reported_at": 1000}],
        1000,
        100,
    )
    assert mismatch["rows"][0]["status"] == "platform-mismatch"
    invalid = review_mod.reconcile_devices(
        [{"id": "a", "platform": "macos"}],
        [{"id": "a", "platform": "macos", "reported_at": None}],
        1000,
        100,
    )
    assert invalid["rows"][0]["status"] == "invalid-report"


def test_unknown_mobile_evidence_never_becomes_compliant(review_mod) -> None:
    result = review_mod.review_mobile("mtd", "ios", {})
    assert result["unmeasured"] == 2
    assert result["reported_present"] == 0
    assert result["conclusion"] == "insufficient-evidence"


def test_management_console_and_device_are_independent_evidence(review_mod) -> None:
    one_sided = review_mod.review_mobile("mtd", "android", {"mtd-managed-android": "observed"})
    missing = review_mod.review_mobile(
        "mtd",
        "android",
        {"mtd-managed-android": "observed", "mtd-device-android": "missing"},
    )
    both = review_mod.review_mobile(
        "mtd",
        "android",
        {"mtd-managed-android": "observed", "mtd-device-android": "observed"},
    )
    assert one_sided["conclusion"] == "insufficient-evidence"
    assert missing["conclusion"] == "reported-finding"
    assert both["conclusion"] == "self-reported-only"
    assert both["results"][0]["rule"] == "GOOG-16-013400"


def test_unsupported_mobile_evidence_is_refused(review_mod) -> None:
    with pytest.raises(ValueError, match="unsupported mobile security review"):
        review_mod.review_mobile("mtd", "windows", {})
    with pytest.raises(ValueError, match="invalid evidence answer"):
        review_mod.review_mobile("mdm", "ios", {"mdm-command-audit": "not-applicable"})


def test_unanswered_guidance_is_unmeasured(review_mod) -> None:
    result = review_mod.review_assurance("sbom", {})
    assert result["unmeasured"] == 4
    assert result["conclusion"] == "insufficient-evidence"


def test_missing_and_self_reported_evidence_stay_distinct(review_mod) -> None:
    missing = review_mod.review_assurance(
        "sase",
        {"sase-identity": "observed", "sase-connector": "missing"},
    )
    all_reported = review_mod.review_assurance(
        "sbom",
        {control["id"]: "observed" for control in review_mod.ASSURANCE_CONTROLS["sbom"]},
    )
    assert missing["reported_gaps"] == 1
    assert missing["conclusion"] == "reported-gap"
    assert all_reported["unmeasured"] == 0
    assert all_reported["conclusion"] == "self-reported-only"


def test_unsupported_or_invalid_assurance_input_refuses(review_mod) -> None:
    with pytest.raises(ValueError, match="unsupported assurance review"):
        review_mod.review_assurance("unknown", {})
    with pytest.raises(ValueError, match="invalid evidence answer"):
        review_mod.review_assurance("sbom", {"sbom-components": "not-applicable"})


def test_supply_chain_gaps_produce_bounded_plan_input(review_mod) -> None:
    unknown = review_mod.review_assurance("scrm", {})
    gap = review_mod.review_assurance("scrm", {"scrm-criticality": "missing"})
    assert unknown["unmeasured"] == 5
    assert unknown["conclusion"] == "insufficient-evidence"
    assert gap["reported_gaps"] == 1
    assert gap["results"][0]["id"] == "scrm-criticality"
    assert gap["results"][0]["status"] == "reported-gap"
    assert isinstance(gap["results"][0]["next_step"], str)


def test_cli_review_reads_an_input_file(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"surface": "transfer", "method": "password-zip", "answers": {}}),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "review", "--input", str(request)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["decision"] == "replace-ppap"


def test_cli_rejects_invalid_input() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "review", "mobile", "--view", "mtd", "--platform", "windows"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "unsupported mobile security review" in completed.stdout
