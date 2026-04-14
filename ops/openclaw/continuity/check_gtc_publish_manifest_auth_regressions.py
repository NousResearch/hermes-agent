#!/usr/bin/env python3
"""Canonical dual-mode publish-manifest authenticity regression harness.

Focus:
- Compatibility HMAC verification remains valid.
- Default Ed25519 verification remains valid.
- Signature tamper fails close for both schemes.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

from check_gtc_latest_schema_regressions import (
    assert_manifest_auth_signature_tamper,
    run_negative_case,
    scenario_manifest_auth_signature_tamper,
    scenario_manifest_auth_signature_tamper_ed25519,
    scenario_publish_manifest_auth_mode_compat_hmac_valid,
    scenario_publish_manifest_auth_mode_default_ed25519_valid,
    scenario_seed_schema_fixture_contract,
)
from strict_required_check_contracts import (
    required_check_contract_inputs,
    required_check_provenance,
    strict_required_check_contract,
)

CONTRACT = strict_required_check_contract("gtc_publish_manifest_auth_dual_mode")
CHECK_ID = CONTRACT.check_id
HARNESS_ID = CONTRACT.harness
SUMMARY_SCHEMA_VERSION = CONTRACT.summary_schema_version
SUMMARY_SOURCE = CONTRACT.summary_source


def _print_result_line(row: dict[str, Any]) -> None:
    name = str(row.get("name") or "unknown")
    if bool(row.get("ok")):
        print(f"PASS {name}")
        return
    detail = str(row.get("detail") or row.get("error") or "")
    if detail:
        print(f"FAIL {name}: {detail}")
    else:
        print(f"FAIL {name}")


SCENARIOS: list[tuple[str, Callable[[], dict[str, Any]]]] = [
    (
        "publish_manifest_auth_mode_compat_hmac_valid",
        scenario_publish_manifest_auth_mode_compat_hmac_valid,
    ),
    (
        "publish_manifest_auth_mode_default_ed25519_valid",
        scenario_publish_manifest_auth_mode_default_ed25519_valid,
    ),
    (
        "publish_manifest_auth_signature_tamper",
        lambda: run_negative_case(
            "publish_manifest_auth_signature_tamper",
            scenario_manifest_auth_signature_tamper,
            expected_surface="publish_manifest_authenticity",
            extra_assert=lambda payload: assert_manifest_auth_signature_tamper(payload, expected_scheme="hmac-sha256"),
        ),
    ),
    (
        "publish_manifest_auth_signature_tamper_ed25519",
        lambda: run_negative_case(
            "publish_manifest_auth_signature_tamper_ed25519",
            scenario_manifest_auth_signature_tamper_ed25519,
            expected_surface="publish_manifest_authenticity",
            extra_assert=lambda payload: assert_manifest_auth_signature_tamper(payload, expected_scheme="ed25519-sha256"),
        ),
    ),
]

_IMPLEMENTED_SCENARIO_NAMES = ["seed_schema_fixture_contract", *[name for name, _ in SCENARIOS]]
if _IMPLEMENTED_SCENARIO_NAMES != list(CONTRACT.scenario_names):
    raise RuntimeError(
        "required-check scenario contract mismatch for "
        f"{CHECK_ID}: implemented={_IMPLEMENTED_SCENARIO_NAMES} expected={list(CONTRACT.scenario_names)}"
    )


def _required_check_contract_inputs() -> dict[str, Any]:
    return required_check_contract_inputs(CHECK_ID)


def _required_check_provenance() -> dict[str, Any]:
    return required_check_provenance(CHECK_ID)


def main() -> int:
    results: list[dict[str, Any]] = []
    failed = 0

    sentinel = scenario_seed_schema_fixture_contract()
    results.append(sentinel)
    _print_result_line(sentinel)
    if not bool(sentinel.get("ok")):
        failed += 1
        summary = {
            "ok": False,
            "total": len(results),
            "failed": failed,
            "results": results,
            "harness": HARNESS_ID,
            "source": SUMMARY_SOURCE,
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "required_check_provenance": _required_check_provenance(),
            "fail_fast": {
                "triggered": True,
                "reason": "seed_schema_fixture_contract",
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1

    for name, fn in SCENARIOS:
        try:
            row = fn()
            results.append(row)
            _print_result_line(row)
            if not bool(row.get("ok")):
                failed += 1
        except Exception as exc:
            failed += 1
            row = {"name": name, "ok": False, "error": str(exc)}
            results.append(row)
            _print_result_line(row)

    summary = {
        "ok": failed == 0,
        "total": len(results),
        "failed": failed,
        "results": results,
        "harness": HARNESS_ID,
        "source": SUMMARY_SOURCE,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "required_check_provenance": _required_check_provenance(),
        "fail_fast": {
            "triggered": False,
            "reason": None,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
