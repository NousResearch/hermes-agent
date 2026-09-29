"""Run the read-tool eval through the REAL Hermes AIAgent.

For each task: fresh temp HERMES_HOME, fresh fixture workspace, real
AIAgent with the file+terminal+search toolsets, real provider API. Collects
accuracy plus efficiency metrics (API turns, tool calls, read_file calls,
prompt/completion tokens, wall time).

Usage:
  python3 evals/readtool/runner.py --model anthropic/claude-opus-4.8 \\
      --provider nous --reps 3 --label baseline
  python3 evals/readtool/runner.py --model qwen/qwen3.8-max \\
      --provider openrouter --reps 3 --label baseline --tasks fifo_hang

Results land in evals/readtool/results/<label>/<model-slug>/rep<N>.json.
Compare two labels with report.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fixtures import build_workspace  # noqa: E402
from tasks import TASKS, TASKS_BY_ID  # noqa: E402

SYSTEM_SUFFIX = (
    "You are working inside the project directory {ws}. All paths in the "
    "task are relative to it. Work autonomously; do not ask questions. "
    "When done, state your final answer plainly."
)


def _count_metrics(messages: list) -> dict:
    api_turns = 0
    tool_calls = 0
    read_calls = 0
    read_errors = 0
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            api_turns += 1
            for tc in m.get("tool_calls") or []:
                tool_calls += 1
                fn = (tc.get("function") or {}).get("name", "")
                if fn == "read_file":
                    read_calls += 1
        elif role == "tool":
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if '"error"' in content or "File not found" in content:
                read_errors += 1
    return {
        "api_turns": api_turns,
        "tool_calls": tool_calls,
        "read_file_calls": read_calls,
        "tool_error_results": read_errors,
    }


def run_task(task, model: str, provider: str, timeout_mult: float,
             toolsets: list[str]) -> dict:
    timeout = task.timeout_s * timeout_mult
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("task timeout must be finite and positive")
    t0 = time.monotonic()
    # The parent owns cleanup: a timed-out worker must be stopped before its
    # workspace disappears. A thread timeout would leave the agent running.
    root = Path(tempfile.mkdtemp(prefix="readtool-"))
    process = None
    try:
        env = dict(os.environ)
        env["HERMES_HOME"] = str(root / ".hermes")
        env["TERMINAL_CWD"] = str(root / "workspace")
        for var in list(env):
            if var.endswith("_API_KEY") and var != "OPENROUTER_API_KEY":
                env.pop(var)
        output = root / "result.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   task.task_id, model, provider, json.dumps(toolsets), str(output)]
        process = subprocess.Popen(command, env=env, start_new_session=os.name != "nt")
        try:
            try:
                process.wait(timeout=max(0, timeout - (time.monotonic() - t0)))
            except subprocess.TimeoutExpired:
                return {"task_id": task.task_id, "capability": task.capability,
                        "final_response": "", "score": 0.0,
                        "wall_s": round(time.monotonic() - t0, 1),
                        "error": f"TimeoutError: task exceeded {timeout:g}s"}
        finally:
            if process.poll() is None:
                from agent.deadline import kill_process_tree
                kill_process_tree(process.pid)
                process.wait(timeout=20)
        # A crashed/misconfigured worker is a harness failure, not model evidence.
        if process.returncode != 0:
            raise SystemExit(f"ABORT: readtool worker exited {process.returncode}")
        envelope = json.loads(output.read_text(encoding="utf-8"))
        if "abort" in envelope:
            raise SystemExit(envelope["abort"])
        return envelope["result"]
    finally:
        if process is None or process.poll() is not None:
            shutil.rmtree(root, ignore_errors=True)
        else:
            # Failed termination must abort, not erase a live worker's state.
            print(f"Worker still alive; preserved eval directory: {root}", file=sys.stderr)


def _run_task_worker(task, model: str, provider: str, toolsets: list[str]) -> dict:
    ws = Path(os.environ["TERMINAL_CWD"])
    Path(os.environ["HERMES_HOME"]).mkdir(parents=True)
    result: dict = {"task_id": task.task_id, "capability": task.capability}
    t0 = time.monotonic()
    try:
        build_workspace(ws)
        # Import inside the env so profile-aware paths bind to the temp home.
        from run_agent import AIAgent  # noqa: PLC0415

        agent = AIAgent(
            model=model,
            provider=provider,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=toolsets,
            max_iterations=40,
        )
        convo = agent.run_conversation(
            SYSTEM_SUFFIX.format(ws=ws) + "\n\nTask: " + task.prompt,
        )
        final = convo.get("final_response") or ""
        messages = convo.get("messages") or []
        result.update(_count_metrics(messages))
        result.update(
            {
                "final_response": final,
                "score": task.grade(final),
                "prompt_tokens": getattr(agent, "session_prompt_tokens", 0),
                "completion_tokens": getattr(agent, "session_completion_tokens", 0),
                "total_tokens": getattr(agent, "session_total_tokens", 0),
                "wall_s": round(time.monotonic() - t0, 1),
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        if "No LLM provider configured" in str(exc) or "authentication" in str(exc).lower():
            # Harness misconfiguration, not a model result. Abort the whole
            # run rather than writing poisoned zero-score records.
            raise SystemExit(f"ABORT (harness config error, not a result): {msg}")
        result.update(
            {
                "final_response": "",
                "score": 0.0,
                "wall_s": round(time.monotonic() - t0, 1),
                "error": msg,
            }
        )
    return result


def _worker_main(argv: list[str]) -> int:
    task_id, model, provider, toolsets, output = argv
    try:
        envelope = {"result": _run_task_worker(
            TASKS_BY_ID[task_id], model, provider, json.loads(toolsets))}
    except SystemExit as exc:
        envelope = {"abort": str(exc)}
    Path(output).write_text(json.dumps(envelope), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", required=True, help="e.g. baseline, feat-fifo-guard")
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default all)")
    ap.add_argument("--timeout-mult", type=float, default=1.0)
    ap.add_argument(
        "--toolsets",
        default="file,terminal,search",
        help=(
            "Comma-separated toolsets. Use 'file' alone for the "
            "discriminative arm (no terminal escape hatch — the read tool "
            "must handle the hostile file itself)."
        ),
    )
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY not in environment. Run: set -a; "
            "source ~/.hermes/.env; set +a  — then relaunch."
        )

    slate = (
        [TASKS_BY_ID[t] for t in args.tasks.split(",") if t]
        if args.tasks
        else TASKS
    )
    slug = args.model.replace("/", "_")
    out_dir = EVAL_DIR / "results" / args.label / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    for rep in range(1, args.reps + 1):
        rep_path = out_dir / f"rep{rep}.json"
        if rep_path.exists():
            print(f"rep{rep} exists, skipping")
            continue
        records = []
        for task in slate:
            print(f"[rep{rep}] {task.task_id} ...", flush=True)
            rec = run_task(task, args.model, args.provider, args.timeout_mult,
                           [t for t in args.toolsets.split(",") if t])
            print(
                f"[rep{rep}] {task.task_id}: score={rec['score']:.2f} "
                f"turns={rec.get('api_turns', '?')} tok={rec.get('total_tokens', '?')} "
                f"wall={rec['wall_s']}s err={rec.get('error')}",
                flush=True,
            )
            records.append(rec)
        rep_path.write_text(
            json.dumps(
                {"model": args.model, "provider": args.provider, "label": args.label,
                 "rep": rep, "records": records},
                indent=2,
            )
        )
        print(f"wrote {rep_path}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        raise SystemExit(_worker_main(sys.argv[2:]))
    raise SystemExit(main())
