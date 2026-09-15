#!/usr/bin/env python3
"""Demo script for project-checker repair loop.

Demonstrates the cleanse functionality without requiring full hermes CLI setup.
"""

import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.cleanse import detect_issues, run_cleanse


def demo_cleanse(project_path: str = "."):
    """Run a demo cleanse on the specified project."""
    root = Path(project_path).resolve()
    print(f"Demo: Running cleanse on {root}")
    print("=" * 60)

    # Step 1: Detect issues
    print("\n1. Detecting issues...")
    detection = detect_issues(root)
    print(f"   Found {len(detection.diagnostics)} diagnostics")
    if detection.skipped_tools:
        print(f"   Skipped tools: {', '.join(detection.skipped_tools)}")

    # Show sample diagnostics
    if detection.diagnostics:
        print("\n   Sample diagnostics:")
        for diag in detection.diagnostics[:3]:
            print(f"   - {diag.file}:{diag.line} [{diag.code}] {diag.message}")
        if len(detection.diagnostics) > 3:
            print(f"   ... and {len(detection.diagnostics) - 3} more")

    # Step 2: Run full cleanse
    print("\n2. Running full cleanse (detect → fix → verify)...")
    result = run_cleanse(root, max_workers=2, run_tests=False)

    # Print results
    print("\n" + "=" * 60)
    print("Cleanse Results:")
    print("=" * 60)
    print(f"Initial issues:   {result.initial_count}")
    print(f"Fixed:            {result.fixed_count}")
    print(f"Remaining:        {len(result.remaining_diagnostics)}")
    print(f"Status:           {'✓ Clean' if result.ok else '⚠ Incomplete'}")

    if result.remaining_diagnostics:
        print("\nRemaining issues (first 5):")
        for diag in result.remaining_diagnostics[:5]:
            print(f"  {diag.file}:{diag.line}:{diag.col or 1} [{diag.code}] {diag.message}")

    evidence_path = root / "verification_evidence.json"
    print(f"\nEvidence written to: {evidence_path}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(demo_cleanse(path))
