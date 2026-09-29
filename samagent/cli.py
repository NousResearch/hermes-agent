#!/usr/bin/env python3
"""SamAgent CLI Entrypoint (`python -m samagent.cli`).

Commands:
- interview <brief>        : Generate <=5 adaptive interview questions with recommended defaults
- plan <brief>             : Freeze .samagent/{spec.yaml,brief.md,contract/*,acceptance/*} and print Plan Card
- build [brief]            : Execute scaffold, git-worktree module workers, and L0–L4 verification
- verify                   : Run L0–L4 verification & security probes on current project
- ledger                   : List active and superseded bi-temporal facts in .samagent/ledger.db
- profiles                 : Generate lean Hermes profiles (samagent-worker/orchestrator/verifier/judge)
- bench                    : Run SamBench-Web v0 + H1–H7 ablation suite
- ui [--port 8080]         : Launch the Mission Control web server
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from evals.samagent_bench.ablation_runner import run_ablation_arms
from evals.samagent_bench.sambench_v0 import run_sambench_v0_suite
from samagent.conductor.engine import SamAgentConductor
from samagent.conductor.verify import VerificationRunner
from samagent.ledger.store import ProjectLedger
from samagent.profiles import generate_samagent_profiles
from samagent.spec.interview import generate_interview_questions
from samagent.spec.models import SpecDocument


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="samagent", description="SamAgent spec-first coding agent CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_int = sub.add_parser("interview", help="Generate <=5 adaptive interview questions")
    p_int.add_argument("brief", help="Project brief")

    p_plan = sub.add_parser("plan", help="Freeze spec, contract, red-first tests, and Plan Card")
    p_plan.add_argument("brief", help="Project brief")
    p_plan.add_argument("--dir", default=".", help="Project directory")
    p_plan.add_argument("--local-strict", action="store_true", help="Force 100%% local model routing")

    p_build = sub.add_parser("build", help="Execute scaffold, worktree workers, and L0–L4 verification")
    p_build.add_argument("brief", nargs="?", default="Yoga studio booking web application", help="Project brief")
    p_build.add_argument("--dir", default=".", help="Project directory")
    p_build.add_argument("--worktrees", action="store_true", help="Run module workers in isolated git worktrees")

    p_ver = sub.add_parser("verify", help="Run L0–L4 verification & security gates")
    p_ver.add_argument("--dir", default=".", help="Project directory")

    p_led = sub.add_parser("ledger", help="Inspect Project Ledger facts")
    p_led.add_argument("--dir", default=".", help="Project directory")

    p_prof = sub.add_parser("profiles", help="Materialize lean Hermes profiles under HERMES_HOME")
    p_prof.add_argument("--hermes-home", default=None, help="Override HERMES_HOME directory")
    p_prof.add_argument("--local-strict", action="store_true", help="Configure profiles for 100%% local routing")

    sub.add_parser("bench", help="Run SamBench-Web v0 + H1–H7 ablation suite")

    p_ui = sub.add_parser("ui", help="Start SamAgent Mission Control UI server")
    p_ui.add_argument("--host", default="0.0.0.0")
    p_ui.add_argument("--port", type=int, default=8080)

    args = ap.parse_args(argv)

    if args.cmd == "interview":
        qs = [q.to_dict() for q in generate_interview_questions(args.brief)]
        print(json.dumps(qs, indent=2))
        return 0

    if args.cmd == "plan":
        proj = Path(args.dir).resolve()
        proj.mkdir(parents=True, exist_ok=True)
        ans = {"Q4_PRIVACY_ROUTING": "local_strict"} if args.local_strict else None
        cond = SamAgentConductor(proj, cloud_available=(not args.local_strict))
        res = cond.prepare_spec_and_contract(args.brief, answers=ans)
        print(json.dumps(res["plan_card"], indent=2))
        return 0

    if args.cmd == "build":
        proj = Path(args.dir).resolve()
        proj.mkdir(parents=True, exist_ok=True)
        cond = SamAgentConductor(proj)
        if not (proj / ".samagent" / "spec.yaml").exists():
            cond.prepare_spec_and_contract(args.brief)
        res = cond.execute_and_verify(use_worktrees=bool(args.worktrees))
        print(json.dumps(res, indent=2))
        return 0 if res["verification"]["passed"] else 1

    if args.cmd == "verify":
        proj = Path(args.dir).resolve()
        spec = SpecDocument.load(proj)
        rep = VerificationRunner(proj, spec).run_all()
        print(json.dumps(rep.to_dict(), indent=2))
        return 0 if rep.passed else 1

    if args.cmd == "ledger":
        proj = Path(args.dir).resolve()
        ledger = ProjectLedger(proj)
        facts = [f.to_dict() for f in ledger.list_facts(active_only=False)]
        print(json.dumps({"facts": facts}, indent=2))
        return 0

    if args.cmd == "profiles":
        home = Path(args.hermes_home) if args.hermes_home else None
        policy = "local_strict" if args.local_strict else "default"
        written = generate_samagent_profiles(home, router_policy=policy, cloud_available=(not args.local_strict))
        print(json.dumps({k: str(v) for k, v in written.items()}, indent=2))
        return 0

    if args.cmd == "bench":
        v0 = run_sambench_v0_suite()
        abl = run_ablation_arms()
        print(json.dumps({"sambench_v0": v0, "ablation": abl}, indent=2))
        return 0

    if args.cmd == "ui":
        import uvicorn
        from samagent.ui_server import app

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
