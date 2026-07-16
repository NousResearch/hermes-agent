"""Phase 9 RED contracts for Context Health update survival.

These tests intentionally describe the expected update-survival smoke/regression
surface before implementing it. They must fail RED while the smoke script and
post-update checklist are absent.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "context_health_smoke.py"
UPDATE_SURVIVAL_CHECKLIST = REPO_ROOT / "references" / "context-health-update-survival.md"

REQUIRED_SMOKE_COVERAGE = {
    "policy loading": ("policy loading", "context_health_policy"),
    "pre-turn intake hook presence": ("pre-turn intake", "intake hook"),
    "WCP provider payload enforcement": ("working context packet", "provider payload"),
    "Task Boundary Firewall default-new behavior": ("task boundary firewall", "default-new"),
    "closed task exclusion": ("closed task", "exclusion"),
    "retrieval scope enforcement": ("retrieval scope", "session_search"),
    "compact failure fallback": ("compact failure", "safe hold"),
    "threshold does not revert to 85%-only path": ("85", "threshold"),
    "same-window rehydrate path or HOLD if not enabled": ("same-window rehydrate", "hold"),
    "update-aftercare checklist": ("aftercare", "post-update"),
}

SAFE_EXECUTION_CONTRACT = {
    "dry-run/default safe mode": ("dry-run", "safe mode"),
    "temp HERMES_HOME/tmp_path based": ("tmp_path", "HERMES_HOME"),
    "synthetic sentinel only": ("synthetic", "sentinel"),
    "nonzero on missing hooks/regressions": ("nonzero", "missing hook"),
    "no raw/private/secret/token/password material in report": (
        "raw/private/secret/token/password",
        "report",
    ),
    "machine-readable summary or clear PASS/FAIL report": (
        "machine-readable",
        "PASS/FAIL",
    ),
}

FORBIDDEN_RUNTIME_SURFACES = {
    "real ~/.hermes/state.db",
    "live provider",
    "network",
    "secrets",
    "tmux/session",
    "profile/systemd/cron/wrapper/env/credential",
    "gateway restart/deploy/activation",
    "CLI slash command activation",
    "runtime config/profile activation",
}

PHASE9_SCOPE_STATEMENT = (
    "Phase 9 is update survival smoke/regression detection scope, not "
    "gateway runtime implementation, command activation, or runtime config/profile activation."
)


def _read_lower(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower()


def _assert_terms_present(text: str, contract: dict[str, tuple[str, ...]]) -> None:
    missing = []
    for behavior, terms in contract.items():
        if not any(term.lower() in text for term in terms):
            missing.append(f"{behavior}: expected one of {terms!r}")
    assert not missing, "Missing smoke contract coverage:\n" + "\n".join(missing)


def test_update_survival_smoke_script_exists_and_declares_required_coverage() -> None:
    """Phase 9 smoke script must enumerate every governance invariant it protects."""
    assert SMOKE_SCRIPT.is_file(), (
        "Phase 9 RED: expected scripts/context_health_smoke.py to exist and declare "
        "smoke coverage for policy loading, pre-turn intake, WCP provider payload, "
        "Task Boundary Firewall default-new behavior, closed task exclusion, retrieval "
        "scope enforcement, compact fallback, 85%-threshold regression, same-window "
        "rehydrate/HOLD, and update-aftercare checklist."
    )

    text = _read_lower(SMOKE_SCRIPT)
    _assert_terms_present(text, REQUIRED_SMOKE_COVERAGE)


def test_update_survival_checklist_document_exists_for_post_update_aftercare() -> None:
    """Phase 9 must document the post-update aftercare command and HOLD boundary."""
    assert UPDATE_SURVIVAL_CHECKLIST.is_file(), (
        "Phase 9 RED: expected references/context-health-update-survival.md to exist "
        "with a post-update aftercare checklist, smoke command, expected output, and "
        "HOLD guidance when governance hooks are missing after update."
    )

    text = _read_lower(UPDATE_SURVIVAL_CHECKLIST)
    required_doc_terms = {
        "post-update aftercare checklist": ("post-update", "aftercare"),
        "smoke command": ("context_health_smoke.py", "smoke command"),
        "expected PASS/FAIL output": ("pass/fail", "expected output"),
        "HOLD on missing hooks": ("hold", "missing hook"),
        "no live side effects": ("no live", "no real state"),
    }
    _assert_terms_present(text, required_doc_terms)


def test_smoke_script_contract_requires_dry_run_safe_mode_and_no_live_side_effects() -> None:
    """The smoke runner must be safe even if executed after an update."""
    assert SMOKE_SCRIPT.is_file(), (
        "Phase 9 RED: expected scripts/context_health_smoke.py before validating its "
        "dry-run/default-safe-mode, temp HERMES_HOME/tmp_path, synthetic-sentinel, "
        "nonzero-on-regression, no-secret-report, and machine-readable summary contracts."
    )

    text = _read_lower(SMOKE_SCRIPT)
    _assert_terms_present(text, SAFE_EXECUTION_CONTRACT)
    for forbidden_surface in FORBIDDEN_RUNTIME_SURFACES:
        assert forbidden_surface.lower() in text, (
            "Smoke script contract must explicitly forbid or guard this live/runtime "
            f"surface: {forbidden_surface}"
        )


def test_phase9_scope_is_detection_not_gateway_or_command_runtime_activation() -> None:
    """Phase 9 can detect gateway/update drift without implementing runtime activation."""
    assert SMOKE_SCRIPT.is_file(), (
        "Phase 9 RED: expected scripts/context_health_smoke.py to encode detection-only "
        "scope for gateway/update survival, while excluding gateway restart/deploy, "
        "CLI slash command activation, and runtime config/profile activation. "
        f"Contract: {PHASE9_SCOPE_STATEMENT}"
    )

    text = _read_lower(SMOKE_SCRIPT)
    expected_scope_terms = {
        "gateway/update detection scope": ("gateway", "detection"),
        "not gateway runtime implementation": ("not gateway runtime", "no gateway restart"),
        "not command activation": ("not command activation", "no slash command"),
        "not runtime config/profile activation": (
            "not runtime config",
            "no profile activation",
        ),
    }
    _assert_terms_present(text, expected_scope_terms)
