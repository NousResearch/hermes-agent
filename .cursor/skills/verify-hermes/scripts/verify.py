#!/usr/bin/env python3
"""
Hermes Verification Control CLI

Systematic feature verification across CLI, TUI, and gateway surfaces.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class VerificationSession:
    """Manages a verification session with isolated workspace."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.workspace = Path(f"/tmp/verify-hermes-{self.session_id}")
        self.evidence_dir = self.workspace / "evidence"
        self.archive_dir = Path.home() / ".cursor" / "verify-hermes" / "archives"
        self.session_file = self.workspace / "session.json"

    def create(self, feature: str, surfaces: List[str]):
        """Initialize verification workspace."""
        print(f"🚀 Launching verification session: {self.session_id}")
        print(f"   Feature: {feature}")
        print(f"   Surfaces: {', '.join(surfaces)}")

        self.workspace.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(exist_ok=True)

        session_data = {
            "session_id": self.session_id,
            "feature": feature,
            "surfaces": surfaces,
            "created_at": datetime.now().isoformat(),
            "status": "launched",
            "phases": {
                "launch": {"status": "complete", "timestamp": datetime.now().isoformat()},
                "doctor": {"status": "pending"},
                "drive": {"status": "pending", "surfaces": {}},
                "evidence": {"status": "pending"},
                "cleanup": {"status": "pending"},
            },
        }

        self.save_session(session_data)
        print(f"✓ Workspace created: {self.workspace}")
        return session_data

    def load_session(self) -> Dict:
        """Load existing session data."""
        if not self.session_file.exists():
            raise FileNotFoundError(f"No session found at {self.workspace}")
        with open(self.session_file) as f:
            return json.load(f)

    def save_session(self, data: Dict):
        """Save session data."""
        with open(self.session_file, "w") as f:
            json.dump(data, f, indent=2)

    def update_phase(self, phase: str, status: str, **kwargs):
        """Update phase status."""
        data = self.load_session()
        data["phases"][phase].update({"status": status, "timestamp": datetime.now().isoformat()})
        data["phases"][phase].update(kwargs)
        self.save_session(data)


class DoctorPhase:
    """Environment health checks."""

    def __init__(self, session: VerificationSession):
        self.session = session

    def run(self) -> bool:
        """Run all health checks."""
        print("🩺 Running Doctor phase...")
        print()

        checks = [
            self._check_python,
            self._check_pytest,
            self._check_hermes_home,
            self._check_git,
        ]

        results = []
        for check in checks:
            name, passed, details = check()
            results.append((name, passed, details))
            status = "✓" if passed else "✗"
            print(f"{status} {name}: {details}")

        print()
        all_passed = all(passed for _, passed, _ in results)

        self.session.update_phase(
            "doctor",
            "complete" if all_passed else "failed",
            checks={name: {"passed": passed, "details": details} for name, passed, details in results},
        )

        if all_passed:
            print("✓ Doctor phase PASSED - environment ready")
        else:
            print("✗ Doctor phase FAILED - fix issues before proceeding")

        return all_passed

    def _check_python(self) -> Tuple[str, bool, str]:
        """Check Python version."""
        version = sys.version.split()[0]
        major, minor = map(int, version.split(".")[:2])
        passed = major == 3 and minor >= 11
        return ("Python version", passed, f"{version} (requires 3.11+)")

    def _check_pytest(self) -> Tuple[str, bool, str]:
        """Check pytest installation."""
        try:
            result = subprocess.run(
                ["pytest", "--version"], capture_output=True, text=True, timeout=5
            )
            passed = result.returncode == 0
            version = result.stdout.split()[1] if passed else "not found"
            return ("pytest", passed, f"version {version}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ("pytest", False, "not found")

    def _check_hermes_home(self) -> Tuple[str, bool, str]:
        """Check HERMES_HOME setup."""
        hermes_home = os.environ.get("HERMES_HOME", Path.home() / ".hermes")
        hermes_path = Path(hermes_home)
        passed = hermes_path.exists()
        return ("HERMES_HOME", passed, f"{hermes_home} ({'exists' if passed else 'missing'})")

    def _check_git(self) -> Tuple[str, bool, str]:
        """Check git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            passed = result.returncode == 0
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
            return ("Git repository", passed, f"branch: {branch}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ("Git repository", False, "git not found")


class DrivePhase:
    """Test execution driver."""

    def __init__(self, session: VerificationSession):
        self.session = session

    def run(self, phase: str) -> bool:
        """Run tests for a specific surface."""
        print(f"🚗 Driving {phase} verification...")
        print()

        session_data = self.session.load_session()
        surfaces = session_data["surfaces"]

        if phase not in surfaces and phase != "all":
            print(f"✗ Surface '{phase}' not in session surfaces: {surfaces}")
            return False

        test_surfaces = surfaces if phase == "all" else [phase]
        results = {}

        for surface in test_surfaces:
            print(f"\n{'=' * 60}")
            print(f"Surface: {surface.upper()}")
            print('=' * 60)

            passed, output = self._run_surface_tests(surface)
            results[surface] = {
                "passed": passed,
                "output": output,
                "timestamp": datetime.now().isoformat(),
            }

            # Save output
            output_file = self.session.evidence_dir / f"{surface}-output.txt"
            with open(output_file, "w") as f:
                f.write(output)

            status = "PASSED ✓" if passed else "FAILED ✗"
            print(f"\n{surface.upper()}: {status}")

        # Update session
        session_data["phases"]["drive"]["surfaces"] = results
        all_passed = all(r["passed"] for r in results.values())
        session_data["phases"]["drive"]["status"] = "complete" if all_passed else "failed"
        self.session.save_session(session_data)

        print(f"\n{'=' * 60}")
        print(f"Drive phase: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
        print('=' * 60)

        return all_passed

    def _run_surface_tests(self, surface: str) -> Tuple[bool, str]:
        """Run tests for a specific surface."""
        test_commands = {
            "cli": ["pytest", "tests/hermes_cli/", "-v", "--tb=short"],
            "tui": ["pytest", "tests/tui_gateway/", "-v", "--tb=short"],
            "gateway": ["pytest", "tests/gateway/", "-v", "--tb=short"],
        }

        if surface not in test_commands:
            return False, f"Unknown surface: {surface}"

        cmd = test_commands[surface]
        print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
            )
            return result.returncode == 0, result.stdout + "\n\n" + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out (5 minutes)"
        except FileNotFoundError:
            return False, f"Command not found: {cmd[0]}"


class EvidencePhase:
    """Evidence collection and report generation."""

    def __init__(self, session: VerificationSession):
        self.session = session

    def run(self):
        """Collect evidence and generate report."""
        print("📊 Collecting evidence...")
        print()

        session_data = self.session.load_session()
        drive_results = session_data["phases"]["drive"].get("surfaces", {})

        # Determine verdict
        verdict = self._determine_verdict(drive_results)

        # Generate summary
        summary = {
            "session_id": self.session.session_id,
            "feature": session_data["feature"],
            "verdict": verdict["status"],
            "timestamp": datetime.now().isoformat(),
            "surfaces": {},
            "blockers": verdict.get("blockers", []),
        }

        for surface, result in drive_results.items():
            summary["surfaces"][surface] = {
                "status": "PASS" if result["passed"] else "FAIL",
                "output_file": f"{surface}-output.txt",
            }

        # Save summary
        summary_file = self.session.evidence_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print report
        self._print_report(summary)

        # Update session
        self.session.update_phase("evidence", "complete", summary=summary)

        return summary

    def _determine_verdict(self, results: Dict) -> Dict:
        """Determine overall verdict."""
        if not results:
            return {"status": "INCONCLUSIVE", "blockers": ["No tests executed"]}

        all_passed = all(r["passed"] for r in results.values())
        any_failed = any(not r["passed"] for r in results.values())

        if all_passed:
            return {"status": "PASS"}
        elif any_failed:
            blockers = [
                f"{surface} tests failed" for surface, r in results.items() if not r["passed"]
            ]
            return {"status": "FAIL", "blockers": blockers}
        else:
            return {"status": "INCONCLUSIVE", "blockers": ["Unable to determine status"]}

    def _print_report(self, summary: Dict):
        """Print evidence report."""
        print("=" * 60)
        print("VERIFICATION EVIDENCE REPORT")
        print("=" * 60)
        print()
        print(f"Session:  {summary['session_id']}")
        print(f"Feature:  {summary['feature']}")
        print(f"Verdict:  {summary['verdict']}")
        print(f"Time:     {summary['timestamp']}")
        print()
        print("Surfaces:")
        for surface, result in summary["surfaces"].items():
            status = result["status"]
            symbol = "✓" if status == "PASS" else "✗"
            print(f"  {symbol} {surface.upper()}: {status}")
        print()

        if summary["blockers"]:
            print("Blockers:")
            for blocker in summary["blockers"]:
                print(f"  - {blocker}")
            print()

        print(f"Evidence location: {self.session.evidence_dir}")
        print("=" * 60)


class CleanupPhase:
    """Cleanup and archival."""

    def __init__(self, session: VerificationSession):
        self.session = session

    def run(self):
        """Archive evidence and cleanup workspace."""
        print("🧹 Cleanup phase...")
        print()

        # Create archive directory
        archive_path = self.session.archive_dir / self.session.session_id
        archive_path.mkdir(parents=True, exist_ok=True)

        # Archive evidence
        if self.session.evidence_dir.exists():
            for file in self.session.evidence_dir.iterdir():
                shutil.copy2(file, archive_path)
            print(f"✓ Archived evidence to: {archive_path}")

        # Copy session file
        if self.session.session_file.exists():
            shutil.copy2(self.session.session_file, archive_path / "session.json")

        # Remove workspace
        if self.session.workspace.exists():
            shutil.rmtree(self.session.workspace)
            print(f"✓ Removed workspace: {self.session.workspace}")

        # Update session (in archive)
        session_file = archive_path / "session.json"
        if session_file.exists():
            with open(session_file) as f:
                data = json.load(f)
            data["phases"]["cleanup"] = {
                "status": "complete",
                "timestamp": datetime.now().isoformat(),
            }
            with open(session_file, "w") as f:
                json.dump(data, f, indent=2)

        print()
        print(f"✓ Cleanup complete")
        print(f"Evidence preserved at: {archive_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hermes Verification Control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Verification commands")

    # Launch command
    launch_parser = subparsers.add_parser("launch", help="Launch new verification session")
    launch_parser.add_argument("--feature", required=True, help="Feature name to verify")
    launch_parser.add_argument(
        "--surfaces",
        default="cli,tui,gateway",
        help="Comma-separated surfaces (default: cli,tui,gateway)",
    )
    launch_parser.add_argument("--session-id", help="Custom session ID")

    # Doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Run environment health checks")
    doctor_parser.add_argument("--session-id", help="Session ID (defaults to latest)")

    # Drive command
    drive_parser = subparsers.add_parser("drive", help="Drive test execution")
    drive_parser.add_argument(
        "--phase",
        choices=["cli", "tui", "gateway", "all"],
        default="all",
        help="Surface to test",
    )
    drive_parser.add_argument("--session-id", help="Session ID (defaults to latest)")

    # Evidence command
    evidence_parser = subparsers.add_parser("evidence", help="Collect evidence and generate report")
    evidence_parser.add_argument("--session-id", help="Session ID (defaults to latest)")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Archive evidence and cleanup")
    cleanup_parser.add_argument("--session-id", help="Session ID (defaults to latest)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Find latest session if not specified
    session_id = args.session_id
    if not session_id and args.command != "launch":
        # Look for latest session in /tmp
        tmp_sessions = sorted(Path("/tmp").glob("verify-hermes-*"))
        if tmp_sessions:
            session_id = tmp_sessions[-1].name.replace("verify-hermes-", "")
        else:
            print("✗ No active session found. Run 'verify.py launch' first.")
            sys.exit(1)

    try:
        if args.command == "launch":
            session = VerificationSession(session_id=args.session_id)
            surfaces = [s.strip() for s in args.surfaces.split(",")]
            session.create(args.feature, surfaces)
            print()
            print(f"Next: python {__file__} doctor --session-id {session.session_id}")

        elif args.command == "doctor":
            session = VerificationSession(session_id=session_id)
            phase = DoctorPhase(session)
            success = phase.run()
            if not success:
                sys.exit(1)
            print()
            print(f"Next: python {__file__} drive --phase cli --session-id {session.session_id}")

        elif args.command == "drive":
            session = VerificationSession(session_id=session_id)
            phase = DrivePhase(session)
            success = phase.run(args.phase)
            if not success:
                sys.exit(1)
            print()
            print(f"Next: python {__file__} evidence --session-id {session.session_id}")

        elif args.command == "evidence":
            session = VerificationSession(session_id=session_id)
            phase = EvidencePhase(session)
            phase.run()
            print()
            print(f"Next: python {__file__} cleanup --session-id {session.session_id}")

        elif args.command == "cleanup":
            session = VerificationSession(session_id=session_id)
            phase = CleanupPhase(session)
            phase.run()

    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
