"""Main repair loop: detect → fix → verify.

Loop: stream diagnostics → group by file → N file-sticky workers; mutating
formatters first serially. Workers fix root cause; no project-wide check rerun;
no diagnostic suppression. Verify = same suite; success iff empty remaining.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.cleanse.detector import detect_issues
from agent.cleanse.parser import Diagnostic
from agent.cleanse.worker import repair_issues

logger = logging.getLogger("hermes.cleanse.loop")


@dataclass
class CleanseResult:
    """Result of a full cleanse run."""

    initial_count: int
    fixed_count: int
    remaining_diagnostics: list[Diagnostic] = field(default_factory=list)
    verification_evidence: dict[str, Any] = field(default_factory=dict)
    skipped_tools: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Success = no remaining diagnostics."""
        return len(self.remaining_diagnostics) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "initial_count": self.initial_count,
            "fixed_count": self.fixed_count,
            "remaining_count": len(self.remaining_diagnostics),
            "remaining_diagnostics": [d.to_dict() for d in self.remaining_diagnostics],
            "verification_evidence": self.verification_evidence,
            "skipped_tools": self.skipped_tools,
        }


def _write_verification_evidence(root: Path, result: CleanseResult) -> None:
    """Write verification_evidence.json to project root."""
    evidence_path = root / "verification_evidence.json"
    try:
        with evidence_path.open("w", encoding="utf-8") as f:
            json.dump(result.verification_evidence, f, indent=2)
        logger.info(f"Wrote verification evidence to {evidence_path}")
    except OSError as exc:
        logger.warning(f"Failed to write verification evidence: {exc}")


def run_cleanse(
    root: Path,
    max_workers: int = 2,
    run_tests: bool = False,
) -> CleanseResult:
    """Run the full cleanse loop: detect → fix → verify.

    Process:
    1. Detect all issues via available checkers
    2. Stream diagnostics, group by file
    3. Apply mutating formatters serially first
    4. Spawn N file-sticky workers to fix remaining issues
    5. Re-run same suite to verify (success = empty remaining)
    6. Write verification_evidence.json

    Args:
        root: Project root directory
        max_workers: Maximum parallel workers (default 2)
        run_tests: Whether to run project tests after fixes (default False)

    Returns:
        CleanseResult with initial/fixed counts and remaining diagnostics
    """
    root = Path(root).resolve()
    logger.info(f"Starting cleanse at {root}")

    # Step 1: Detect issues
    logger.info("Detecting issues...")
    detection = detect_issues(root)
    initial_count = len(detection.diagnostics)
    logger.info(f"Found {initial_count} diagnostics across {len(set(d.file for d in detection.diagnostics))} files")

    if detection.skipped_tools:
        logger.info(f"Skipped tools (not on PATH): {', '.join(detection.skipped_tools)}")

    if initial_count == 0:
        logger.info("No issues found; project is clean")
        evidence = {
            "status": "clean",
            "initial_count": 0,
            "skipped_tools": detection.skipped_tools,
        }
        result = CleanseResult(
            initial_count=0,
            fixed_count=0,
            remaining_diagnostics=[],
            verification_evidence=evidence,
            skipped_tools=detection.skipped_tools,
        )
        _write_verification_evidence(root, result)
        return result

    # Step 2: Repair issues
    logger.info(f"Repairing issues with {max_workers} workers...")
    repair_results = repair_issues(root, detection.diagnostics, max_workers=max_workers)

    # Count fixed vs remaining
    remaining_diagnostics: list[Diagnostic] = []
    for file_path, repair_result in repair_results.items():
        remaining_diagnostics.extend(repair_result.remaining)

    fixed_count = initial_count - len(remaining_diagnostics)
    logger.info(f"Fixed {fixed_count}/{initial_count} diagnostics")

    # Step 3: Verify by re-running same suite
    logger.info("Verifying fixes...")
    verification = detect_issues(root)
    final_diagnostics = verification.diagnostics

    # Step 4: Run tests if requested
    test_status = None
    if run_tests and detection.recipe:
        logger.info("Running project tests...")
        from agent.verify.runner import run_verify

        verify_result = run_verify(
            root,
            detection.recipe,
            phases=("test",),
            stop_on_failure=False,
        )
        test_status = "passed" if verify_result.ok else "failed"
        logger.info(f"Tests {test_status}")

    # Build verification evidence
    evidence = {
        "status": "clean" if len(final_diagnostics) == 0 else "incomplete",
        "initial_count": initial_count,
        "fixed_count": fixed_count,
        "remaining_count": len(final_diagnostics),
        "skipped_tools": detection.skipped_tools,
    }
    if test_status:
        evidence["test_status"] = test_status

    result = CleanseResult(
        initial_count=initial_count,
        fixed_count=fixed_count,
        remaining_diagnostics=final_diagnostics,
        verification_evidence=evidence,
        skipped_tools=detection.skipped_tools,
    )

    # Write evidence
    _write_verification_evidence(root, result)

    if result.ok:
        logger.info("✓ Cleanse complete: all issues resolved")
    else:
        logger.warning(f"⚠ Cleanse incomplete: {len(final_diagnostics)} issues remain")

    return result
