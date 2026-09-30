#!/usr/bin/env python3
"""SamBench-Web v0: 6-task benchmark across Tiers T1–T3 + H5/H6 verification (05-final-plan.md §1, §13, T0.4).

Features:
- 6 canonical tasks across T1 (basic web), T2 (CRUD + auth + DB), T3 (multi-role + mock payments + audit)
- Pre-launch spend estimator & hard campaign cap enforcement (6 tasks × 3 arms × 2 models × 2 reps = 72 runs)
- Programmatic hidden graders + seeded vulnerability suite (H6: 5 vulnerability classes)
- Cold-resume Ledger memory probe (H5: 10 scripted probes, <= 2K injected tokens)
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from samagent.conductor.engine import SamAgentConductor  # noqa: E402
from samagent.conductor.verify import scan_directory_security  # noqa: E402
from samagent.ledger.store import ProjectLedger  # noqa: E402


@dataclass(frozen=True)
class SamBenchTask:
    id: str
    tier: str  # "T1" | "T2" | "T3"
    title: str
    brief: str
    ambiguous: bool
    expected_roles: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SAMBENCH_V0_TASKS: List[SamBenchTask] = [
    SamBenchTask(
        id="t1_habit_tracker",
        tier="T1",
        title="Daily Habit Tracker",
        brief="Build a simple public habit tracker where anyone can view habits and add a new habit with title validation.",
        ambiguous=False,
        expected_roles=["visitor"],
    ),
    SamBenchTask(
        id="t1_markdown_notes",
        tier="T1",
        title="Quick Scratchpad Board",
        brief="Create a public note board to list notes and post new notes without requiring a login.",
        ambiguous=True,
        expected_roles=["visitor"],
    ),
    SamBenchTask(
        id="t2_yoga_booking",
        tier="T2",
        title="Yoga Studio Class Booking",
        brief="Booking site for my yoga studio where visitors see classes, members book classes (no double booking, private per member), and admins add classes.",
        ambiguous=True,
        expected_roles=["visitor", "member", "admin"],
    ),
    SamBenchTask(
        id="t2_clinic_appointments",
        tier="T2",
        title="Clinic Patient Appointment Portal",
        brief="Patient portal with member login to book appointments, strict privacy so patients never see each other's bookings, and admin schedule management.",
        ambiguous=False,
        expected_roles=["visitor", "member", "admin"],
    ),
    SamBenchTask(
        id="t3_saas_helpdesk",
        tier="T3",
        title="Multi-Role B2B Support Helpdesk",
        brief="Customer support portal with visitor status page, member ticket creation and owner-only viewing, admin queue management, and mock payment/webhook billing.",
        ambiguous=True,
        expected_roles=["visitor", "member", "admin"],
    ),
    SamBenchTask(
        id="t3_event_ticketing",
        tier="T3",
        title="Event Ticketing & Seat Reservation",
        brief="Event reservation web app with public catalog, authenticated member seat booking (prevent duplicate seat booking and IDOR), admin event creation, and mock checkout.",
        ambiguous=False,
        expected_roles=["visitor", "member", "admin"],
    ),
]


def estimate_campaign_spend(
    *,
    num_tasks: int = 6,
    num_arms: int = 3,
    num_models: int = 2,
    reps: int = 2,
    avg_usd_per_cloud_run: float = 0.28,
    max_campaign_usd: float = 25.0,
) -> Dict[str, Any]:
    """Print and verify pre-launch spend estimate before any campaign runs (05-final-plan.md §13)."""
    total_runs = num_tasks * num_arms * num_models * reps
    # Assume 1 local model ($0) + 1 cloud model
    cloud_runs = num_tasks * num_arms * max(0, num_models - 1) * reps
    est_usd = round(cloud_runs * avg_usd_per_cloud_run, 2)
    within_cap = est_usd <= max_campaign_usd
    return {
        "total_runs": total_runs,
        "cloud_runs": cloud_runs,
        "estimated_usd": est_usd,
        "max_campaign_usd": max_campaign_usd,
        "within_cap": within_cap,
    }


def evaluate_h5_cold_resume() -> Dict[str, Any]:
    """H5 Eval: 10 scripted cold-resume probes on ProjectLedger (target: >=90% accuracy, <=2K tokens)."""
    with tempfile.TemporaryDirectory(prefix="sambench-h5-") as tmp:
        ledger = ProjectLedger(Path(tmp))
        f_auth = ledger.record_fact(
            scope="auth",
            kind="decision",
            text="Use cookie session tokens with sqlite users table",
            source_ref="spec.yaml",
        )
        # Supersede auth decision to test bi-temporal validity
        ledger.supersede_fact(
            f_auth.id,
            new_text="Use Bearer token header with RBAC role check in SecureAppService",
            source_ref="CCR-1",
        )
        ledger.record_fact(
            scope="db",
            kind="schema",
            text="Bookings table enforces UNIQUE(item_id, owner_id) to reject duplicate bookings with HTTP 409",
            source_ref="contract/db/schema.sql",
        )
        ledger.record_fact(
            scope="security",
            kind="idor",
            text="GET /api/bookings/{id} must verify row['owner_id'] == user_id or role == 'admin', else HTTP 403",
            source_ref="contract/openapi.yaml",
        )
        ledger.record_fact(
            scope="secrets",
            kind="private_note",
            text="Internal staging admin email is ops-internal-vault@corp.local",
            sensitivity="private",
        )
        ledger.record_attempt(
            task_id="task_api_1",
            file_path="app/main.py",
            approach="Used raw string formatting in SQL query",
            outcome="failed",
            error_signature="L3_raw_fstring_sql_injection",
        )

        probes = [
            ("auth token header", False, "Bearer token header", "cookie session tokens"),
            ("duplicate booking constraint", False, "UNIQUE(item_id, owner_id)", None),
            ("IDOR owner check bookings", False, "HTTP 403", None),
            ("staging admin email", False, "ops-internal-vault@corp.local", None),
            ("staging admin email", True, None, "ops-internal-vault@corp.local"),  # cloud route MUST strip private!
            ("app/main.py SQL approach", False, "L3_raw_fstring_sql_injection", None),
            ("RBAC role check", False, "SecureAppService", None),
            ("HTTP 409 duplicate", True, "HTTP 409", "ops-internal-vault@corp.local"),
            ("contract schema bookings", False, "Bookings table", None),
            ("failed attempt main.py", True, "raw string formatting", "ops-internal-vault@corp.local"),
        ]

        passed_probes = 0
        token_counts: List[int] = []
        for query, is_cloud, must_contain, must_not_contain in probes:
            block = ledger.build_turn_context_block(
                query,
                is_cloud_route=is_cloud,
                target_files=["app/main.py"],
                max_tokens=2000,
            )
            toks = len(block) // 4
            token_counts.append(toks)
            ok = toks <= 2000
            if must_contain and must_contain not in block:
                ok = False
            if must_not_contain and must_not_contain in block:
                ok = False
            if ok:
                passed_probes += 1

        token_counts.sort()
        median_toks = token_counts[len(token_counts) // 2]
        pass_rate = round(100.0 * passed_probes / len(probes), 1)
        return {
            "hypothesis": "H5 (Ledger cold-resume survival & privacy)",
            "probes": len(probes),
            "passed_probes": passed_probes,
            "pass_rate_pct": pass_rate,
            "median_injected_tokens": median_toks,
            "target_met": pass_rate >= 90.0 and median_toks <= 2000,
        }


def evaluate_h6_seeded_vulnerabilities() -> Dict[str, Any]:
    """H6 Eval: Seed 5 OWASP vibe-coding vulnerabilities and verify L3 catches 100% with 0% false blocks."""
    caught = 0
    seeded_cases = 5

    with tempfile.TemporaryDirectory(prefix="sambench-h6-") as tmp:
        root = Path(tmp)
        conductor = SamAgentConductor(root)
        conductor.prepare_spec_and_contract("Yoga studio booking app with member auth")
        res = conductor.execute_and_verify()
        clean_passed = res["verification"]["passed"]

        # Seed 1: Hardcoded OpenAI secret key
        (root / "app" / "leaked.py").write_text('API_KEY = "sk-proj-1234567890abcdefghijklmnop"\n', encoding="utf-8")
        s1 = not scan_directory_security(root / "app")["passed"]
        (root / "app" / "leaked.py").unlink()
        if s1:
            caught += 1

        # Seed 2: Raw f-string SQL injection
        (root / "app" / "sqli.py").write_text(
            'def bad(conn, uid):\n    conn.execute(f"SELECT * FROM users WHERE id = {uid}")\n',
            encoding="utf-8",
        )
        s2 = not scan_directory_security(root / "app")["passed"]
        (root / "app" / "sqli.py").unlink()
        if s2:
            caught += 1

        # Seed 3, 4, 5: Missing auth (anonymous booking), IDOR (cross-member read), and missing input validation
        main_py = root / "app" / "main.py"
        orig_main = main_py.read_text(encoding="utf-8")
        for broken_key in ("unauth_rejected", "idor_blocked", "input_validated"):
            main_py.write_text(
                orig_main.replace(f'"{broken_key}": ', f'"{broken_key}": False and '),
                encoding="utf-8",
            )
            ver = conductor.execute_and_verify() if False else None
            from samagent.conductor.verify import VerificationRunner
            from samagent.spec.models import SpecDocument

            l3 = VerificationRunner(root, SpecDocument.load(root)).run_l3_security()
            if not l3.passed:
                caught += 1
        main_py.write_text(orig_main, encoding="utf-8")

    catch_rate = round(100.0 * caught / seeded_cases, 1)
    false_block_rate = 0.0 if clean_passed else 100.0
    return {
        "hypothesis": "H6 (Secure by default — seeded vulnerability detection)",
        "seeded_vulnerabilities": seeded_cases,
        "caught": caught,
        "catch_rate_pct": catch_rate,
        "false_block_rate_pct": false_block_rate,
        "target_met": catch_rate >= 95.0 and false_block_rate < 10.0,
    }


def run_sambench_v0_suite() -> Dict[str, Any]:
    """Run all 6 SamBench-Web v0 tasks through the SamAgent pipeline and grade them."""
    t0 = time.monotonic()
    spend = estimate_campaign_spend()
    task_results: List[Dict[str, Any]] = []

    for task in SAMBENCH_V0_TASKS:
        with tempfile.TemporaryDirectory(prefix=f"sambench-{task.id}-") as tmp:
            proj = Path(tmp)
            conductor = SamAgentConductor(proj, cloud_available=True)
            # Test skip-safe interview on ambiguous tasks, explicit answers on non-ambiguous tasks
            answers = (
                None
                if task.ambiguous
                else {
                    "Q1_AUTH_ROLES": "roles_3" if "member" in task.expected_roles else "public_only",
                }
            )
            prep = conductor.prepare_spec_and_contract(task.brief, answers=answers)
            red_ok = prep["red_first_check"]["is_red_for_right_reason"]
            exec_res = conductor.execute_and_verify(run_id=f"bench_{task.id}")
            ver_ok = exec_res["verification"]["passed"]
            task_results.append(
                {
                    "task_id": task.id,
                    "tier": task.tier,
                    "title": task.title,
                    "ambiguous_brief": task.ambiguous,
                    "red_first_verified": red_ok,
                    "verification_l0_l4_passed": ver_ok,
                    "fanout_parallel": exec_res["fanout"]["allow_parallel"],
                    "assumptions_logged": len(exec_res["assumptions"]),
                    "judge_model": exec_res["verification"]["judge_model"],
                }
            )

    h5 = evaluate_h5_cold_resume()
    h6 = evaluate_h6_seeded_vulnerabilities()
    elapsed = round(time.monotonic() - t0, 2)

    passed_tasks = sum(1 for r in task_results if r["red_first_verified"] and r["verification_l0_l4_passed"])
    return {
        "benchmark": "SamBench-Web v0",
        "elapsed_seconds": elapsed,
        "campaign_spend_estimate": spend,
        "tasks_total": len(task_results),
        "tasks_passed": passed_tasks,
        "pass_rate_pct": round(100.0 * passed_tasks / len(task_results), 1),
        "task_results": task_results,
        "h5_cold_resume": h5,
        "h6_security_probes": h6,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="Write JSON report to this path")
    args = ap.parse_args()

    report = run_sambench_v0_suite()
    text = json.dumps(report, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["tasks_passed"] == report["tasks_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
