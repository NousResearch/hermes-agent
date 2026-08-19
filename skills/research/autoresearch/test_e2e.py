#!/usr/bin/env python3
"""End-to-end test for autoresearch helper scripts.

Tests Mode 2 (Knowledge Research) flow:
  init -> plan -> branch -> work(echo) -> evaluate -> merge -> branch -> evaluate(revert) -> report

Returns JSON with pass/fail per step.
"""
import json, os, sys, shutil, subprocess, tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent / "scripts"
PASS, FAIL = [], []

def run(cmd, cwd=None, input_json=None):
    """Run a python script command, return parsed JSON output."""
    env = {**os.environ, "PYTHONPATH": str(SCRIPTS_DIR)}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}\nstderr: {r.stderr}")
    out = r.stdout.strip()
    # Collect all lines from first { to last } for multi-line JSON
    first, last = out.find("{"), out.rfind("}")
    if first == -1 or last == -1:
        raise RuntimeError(f"No JSON output from: {' '.join(cmd)}\nstdout: {out}")
    json_str = out[first:last+1]
    return json.loads(json_str)

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS: {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL: {name} -- {detail}")

def main():
    workdir = Path(tempfile.mkdtemp(prefix="autoresearch_test_"))
    run_dir = str(workdir / "research")
    ws_dir = run_dir + "/workspace"

    print(f"Test dir: {workdir}")
    print(f"Scripts:  {SCRIPTS_DIR}")
    print()

    # --- STEP 1: state.py init ---
    print("=== 1. state.py init ===")
    r = run(["python3", str(SCRIPTS_DIR/"state.py"), "init", run_dir,
             "Test AI market", "market", "AI infrastructure", "Quick", "5"])
    check("init status", r.get("status") == "initialized")
    check("workspace dir", Path(r["workspace"]).exists(), "workspace not created")

    # --- STEP 2: workspace.py init ---
    print("\n=== 2. workspace.py init ===")
    r = run(["python3", str(SCRIPTS_DIR/"workspace.py"), "init", ws_dir])
    check("init outputs commands", "commands" in r and len(r["commands"]) >= 2)
    # Execute the commands
    for cmd in r["commands"]:
        subprocess.run(cmd, shell=True, capture_output=True)
    check("workspace is git repo", Path(ws_dir + "/.git").exists())
    # Create initial doc
    (Path(ws_dir) / "research.md").write_text("# AI Infrastructure Market\n\n## Overview\n\n## Players\n\n## Pricing\n\n## Trends\n")
    subprocess.run(f"cd {ws_dir} && git add -A && git commit -m 'initial skeleton'", shell=True, capture_output=True)
    check("initial commit", True)

    # --- STEP 3: plan.py write ---
    print("\n=== 3. plan.py write + read ---")
    experiments = json.dumps([
        {"id": 1, "type": "investigate", "hypothesis": "Find top AI infra providers", "target_section": "Players"},
        {"id": 2, "type": "investigate", "hypothesis": "Compare pricing models", "target_section": "Pricing"},
        {"id": 3, "type": "deepen", "hypothesis": "Deep dive on trending tools", "target_section": "Trends"},
    ])
    r = run(["python3", str(SCRIPTS_DIR/"plan.py"), "write", run_dir, experiments])
    check("plan written", r.get("status") == "plan_written", f"got {r}")
    check("3 experiments", r.get("count") == 3, f"count={r.get('count')}")

    r = run(["python3", str(SCRIPTS_DIR/"plan.py"), "read", run_dir])
    check("plan readable", len(r.get("experiments", [])) == 3)

    # --- STEP 4: next-pending ---
    print("\n=== 4. plan.py next-pending ===")
    r = run(["python3", str(SCRIPTS_DIR/"plan.py"), "next-pending", run_dir])
    check("first pending is id=1", r.get("id") == 1, f"got id={r.get('id')}")

    # --- STEP 5: workspace.py branch + merge (happy path) ---
    print("\n=== 5. workspace.py branch -> merge ===")
    # Branch
    r = run(["python3", str(SCRIPTS_DIR/"workspace.py"), "branch", ws_dir, "1", "ai-providers"])
    check("branch commands generated", "commands" in r)
    for cmd in r["commands"]:
        subprocess.run(cmd, shell=True, capture_output=True)

    # Simulate work: edit the file
    content = Path(ws_dir + "/research.md").read_text()
    content = content.replace("## Players\n", "## Players\n\nMajor providers: AWS, GCP, Azure, Modal, Replicate, Lambda Labs.\n")
    Path(ws_dir + "/research.md").write_text(content)

    # Evaluate
    r = run(["python3", str(SCRIPTS_DIR/"evaluate.py"), "score", "4", "5", "3", "5", "4"])
    check("merge decision", r.get("decision") == "MERGE", f"decision={r.get('decision')} scores: {r.get('scores')}")

    # Merge
    r = run(["python3", str(SCRIPTS_DIR/"workspace.py"), "merge", ws_dir, "1", "ai-providers", "exp 1: added providers"])
    check("merge commands generated", "commands" in r)
    for cmd in r["commands"]:
        subprocess.run(cmd, shell=True, capture_output=True)

    # Update plan
    run(["python3", str(SCRIPTS_DIR/"plan.py"), "update-experiment", run_dir, "1", "merged", "--reason", "data found"])

    # Log result
    r = run(["python3", str(SCRIPTS_DIR/"evaluate.py"), "log-result", run_dir, "1",
             "Found providers", "investigate", "Players", "MERGE", "Specific data found",
             "--scores", "E=4,A=5,D=3,R=5,N=4"])
    check("result logged", True)

    # --- STEP 6: Branch -> REVERT path ---
    print("\n=== 6. workspace.py branch -> revert (bad experiment) ===")
    r = run(["python3", str(SCRIPTS_DIR/"workspace.py"), "branch", ws_dir, "2", "bad-pricing"])
    for cmd in r["commands"]:
        subprocess.run(cmd, shell=True, capture_output=True)
    run(["python3", str(SCRIPTS_DIR/"plan.py"), "update-experiment", run_dir, "2", "in_progress"])

    # Simulate bad work: nonsense section
    content = Path(ws_dir + "/research.md").read_text()
    content += "\n\n## Pricing\n\nThis section was vandalized with nonsense.\n"
    Path(ws_dir + "/research.md").write_text(content)

    # Evaluate
    r = run(["python3", str(SCRIPTS_DIR/"evaluate.py"), "score", "1", "1", "1", "1", "1"])
    check("revert decision (all 1s)", r.get("decision") == "REVERT", f"decision={r.get('decision')}")

    # Revert
    r = run(["python3", str(SCRIPTS_DIR/"workspace.py"), "revert", ws_dir, "2", "bad-pricing"])
    check("revert commands generated", "commands" in r)
    for cmd in r["commands"]:
        subprocess.run(cmd, shell=True, capture_output=True)

    run(["python3", str(SCRIPTS_DIR/"plan.py"), "update-experiment", run_dir, "2", "reverted", "--reason", "nonsense"])
    run(["python3", str(SCRIPTS_DIR/"evaluate.py"), "log-result", run_dir, "2",
             "Bad pricing edit", "investigate", "Pricing", "REVERT", "Low quality",
             "--scores", "E=1,A=1,D=1,R=1,N=1"])

    # --- STEP 7: state updates ---
    print("\n=== 7. state.py update-status + checkpoint ===")
    r = run(["python3", str(SCRIPTS_DIR/"state.py"), "update-status", run_dir, "executing",
             "--experiments-done", "2", "--experiments-total", "3", "--experiments-merged", "1", "--experiments-reverted", "1"])
    check("status updated", r.get("phase") == "executing", f"phase={r.get('phase')}")

    r = run(["python3", str(SCRIPTS_DIR/"state.py"), "checkpoint", run_dir, "1", "3"])
    check("checkpoint written", r.get("last_completed") == 1)

    # --- STEP 8: evaluate stats ---
    print("\n=== 8. evaluate.py stats + read-results ===")
    r = run(["python3", str(SCRIPTS_DIR/"evaluate.py"), "stats", run_dir])
    check("stats merged=1", r.get("merged") == 1, f"merged={r.get('merged')}")
    check("stats reverted=1", r.get("reverted") == 1, f"reverted={r.get('reverted')}")

    # read-results outputs plain text, not JSON
    env = {**os.environ, "PYTHONPATH": str(SCRIPTS_DIR)}
    r = subprocess.run(["python3", str(SCRIPTS_DIR/"evaluate.py"), "read-results", run_dir, "--last", "5"],
                       capture_output=True, text=True, env=env)
    check("results readable", r.returncode == 0 and "Experiment" in r.stdout)

    # --- STEP 9: budget check ---
    print("\n=== 9. state.py check-budget ===")
    r = run(["python3", str(SCRIPTS_DIR/"state.py"), "check-budget", run_dir, "--tokens", "100"])
    check("budget not exceeded", not r.get("exceeded"), f"violations={r.get('violations')}")

    # --- STEP 10: workspace log ---
    print("\n=== 10. workspace.py log + status ===")
    r = run(["python3", str(SCRIPTS_DIR/"workspace.py"), "log", ws_dir, "--oneline"])
    check("log works", "commands" in r)

    r = run(["python3", str(SCRIPTS_DIR/"state.py"), "status", run_dir])
    check("status readable", r.get("phase") == "executing")

    # --- STEP 11: report.py ---
    print("\n=== 11. report.py generate + summary ===")
    r = run(["python3", str(SCRIPTS_DIR/"report.py"), "generate", run_dir])
    check("report generated", r.get("status") == "generated", f"got {r}")
    check("report file exists", Path(r["path"]).exists())

    report = Path(r["path"]).read_text()
    check("report has title", "# Research Report:" in report)
    check("report has experiments", "Merged Experiments" in report)
    check("report has reverted", "Reverted Experiments" in report)

    r = run(["python3", str(SCRIPTS_DIR/"report.py"), "summary", run_dir])
    check("summary generated", "summary" in r)

    # --- STEP 12: usage.py track ---
    print("\n=== 12. usage.py track + summary ===")
    r = run(["python3", str(SCRIPTS_DIR/"usage.py"), "track", run_dir, "1", "5000", "3000"])
    check("usage tracked", r.get("status") == "tracked")

    r = run(["python3", str(SCRIPTS_DIR/"usage.py"), "summary", run_dir])
    check("usage summary works", True)

    # --- STEP 13: state.py control ---
    print("\n=== 13. state.py control + read-control ===")
    run(["python3", str(SCRIPTS_DIR/"state.py"), "control", run_dir, "--action", "pause"])
    r = run(["python3", str(SCRIPTS_DIR/"state.py"), "read-control", run_dir])
    check("control shows pause", r.get("action") == "pause", f"action={r.get('action')}")

    # Reset
    run(["python3", str(SCRIPTS_DIR/"state.py"), "control", run_dir, "--action", "none"])

    # --- SUMMARY ---
    print(f"\n{'='*50}")
    print(f"Results: {len(PASS)} passed, {len(FAIL)} failed out of {len(PASS)+len(FAIL)}")
    if FAIL:
        print(f"FAILED: {', '.join(FAIL)}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")

    # Cleanup
    shutil.rmtree(workdir, ignore_errors=True)

if __name__ == "__main__":
    main()
