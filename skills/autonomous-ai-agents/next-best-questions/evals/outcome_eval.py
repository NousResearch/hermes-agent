"""outcome_eval.py — the objective-outcome eval (P3-P6): does asking the skill's questions
actually make the final artifact better?

Protocol (ClarifyGPT / AmbigSWE / arXiv:2606.03135): ambiguous-but-executable tasks
(evals/outcome_bank.py) + a STRICT simulated user who answers a question only if the hidden
spec genuinely resolves it (generic fishing gets "The spec doesn't say.") + paired arms per
task, outcome = fraction of hidden tests passed.

Arms:
  baseline      solve the ambiguous prompt directly, no questions.
  nbq           run the skill, ask its top-K bucket questions, fold Q&A into the solve.
  zeroshot      one naive call for "the K best clarifying questions" — no EVSI machinery (P4).
  prompt-evsi   the ENTIRE framework carried by one prompt instead of the script — the
                prompt-vs-script question, made falsifiable.
  nbq-derive    nbq with --auto-derive on (derive-or-ask exercised end-to-end).

Analysis (printed + saved): per-task paired Δpass vs baseline, sign test, mean Δ; the P6
anchor = Spearman(mean q_value of asked questions, per-task Δpass) across nbq tasks — the
first objective calibration of the skill's value scores (no extra model calls).

Usage (host):
  OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes \\
  python3 evals/outcome_eval.py --arms baseline nbq zeroshot prompt-evsi --k 3 \\
      --out ~/.hermes/outcome_eval.json
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import infogain  # noqa: E402
import pipeline  # noqa: E402
import outcome_bank  # noqa: E402
from validate_evsi import preflight_model  # noqa: E402
from analyze_evsi import spearman  # noqa: E402

NO_ANSWER = "The spec doesn't say."


# ── strict user simulator ─────────────────────────────────────────────────────

def simulator_prompt(hidden_spec, question):
    return (
        "You are the USER in a requirements dialogue. Your COMPLETE private spec for the task "
        "is below. An assistant asks you ONE clarifying question.\n\n"
        f"YOUR PRIVATE SPEC:\n{hidden_spec}\n\n"
        f"ASSISTANT'S QUESTION: {question}\n\n"
        "Rules (strict): if your spec genuinely resolves the question — or resolves PART of a "
        "compound question — answer that part briefly using ONLY information from the spec "
        "(a question offering options, e.g. 'X, Y, or Z?', IS resolved if the spec picks one). "
        "If the spec addresses none of it, or the question is generic fishing, reply with "
        "exactly:\n"
        f"{NO_ANSWER}\n"
        "Never invent details beyond the spec. Reply with the answer only."
    )


def simulate_user(hidden_spec, question, model, timeout=90):
    out = pipeline.raw_chat(model, simulator_prompt(hidden_spec, question),
                            timeout=timeout, num_predict=150)
    ans = (out.get("content") or "").strip()
    revealed = bool(ans) and NO_ANSWER.lower().rstrip(".") not in ans.lower()
    return {"question": question, "answer": ans or NO_ANSWER, "revealed": revealed}


# ── solving + objective scoring ───────────────────────────────────────────────

def solve_prompt(task, qa):
    block = ""
    if qa:
        block = ("\nYou asked the user clarifying questions; their answers:\n"
                 + "\n".join(f"- Q: {x['question']}\n  A: {x['answer']}" for x in qa) + "\n")
    return (
        f"Write the Python function described below.\n\nTASK: {task['ambiguous_prompt']}\n"
        f"{block}\n"
        f"Reply with ONLY a Python code block defining `{task['func']}` (stdlib only, no I/O)."
    )


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text or "", re.DOTALL)
    if m:
        return m.group(1)
    return text or ""


def run_tests(code, tests, timeout=10):
    """Execute solution+tests in a subprocess; each test scored independently. Returns
    (frac_passed, per_test_bools). Any crash in the solution body fails all tests."""
    lines = [code, ""]
    for i, t in enumerate(tests):
        lines += [f"try:",
                  f"    assert {t}",
                  f"    print('PASS {i}')",
                  f"except Exception:",
                  f"    print('FAIL {i}')"]
    try:
        r = subprocess.run([sys.executable, "-I", "-c", "\n".join(lines)],
                           capture_output=True, text=True, timeout=timeout)
        out = r.stdout
    except subprocess.TimeoutExpired:
        out = ""
    results = [f"PASS {i}" in out for i in range(len(tests))]
    return (sum(results) / len(tests) if tests else 0.0), results


def solve_and_score(task, qa, solver_model, timeout=240):
    out = pipeline.raw_chat(solver_model, solve_prompt(task, qa),
                            timeout=timeout, num_predict=900)
    code = extract_code(out.get("content"))
    frac, per = run_tests(code, task["tests"])
    return {"code": code, "frac": frac, "per_test": per}


# ── question sources (the arms' only difference) ─────────────────────────────

def questions_nbq(task, k, skill_model, auto_derive=False, max_rounds=None):
    cfg = dict(infogain.DEFAULTS)
    for key in infogain._MODEL_KEYS:
        cfg[key] = skill_model
    cfg["derive_model"] = ""            # -> value_judge_model
    cfg["families"] = infogain.families_cfg(families_model=skill_model)
    if max_rounds:
        cfg["max_rounds"] = max_rounds
    if auto_derive:
        cfg["auto_derive"] = "on"
    result = infogain.run(task["ambiguous_prompt"], cfg)
    qs = [r["question"] for r in result["bucket"][:k]]
    meta = {"q_values": [round(r.get("value", 0.0), 3) for r in result["bucket"][:k]],
            "derived": result.get("derived", []),
            "usage": result.get("usage", {})}
    return qs, meta


_NUMBERED = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")


def _parse_numbered(text, k):
    qs = [m.group(1) for line in (text or "").splitlines() if (m := _NUMBERED.match(line))]
    return qs[:k]


def questions_zeroshot(task, k, model, timeout=120):
    p = (f"You are about to implement this task:\n\nTASK: {task['ambiguous_prompt']}\n\n"
         f"Before writing any code, ask the user the {k} best clarifying questions. "
         f"Reply with ONLY the {k} questions, numbered 1..{k}.")
    out = pipeline.raw_chat(model, p, timeout=timeout, num_predict=400)
    return _parse_numbered(out.get("content"), k), {"raw": out.get("content", "")}


def questions_prompt_evsi(task, k, model, timeout=180):
    """The whole framework in ONE prompt — no staged calls, no exact arithmetic, no
    role-routing. If this matches the script on outcomes, the ideas suffice in-prompt."""
    p = (
        "You rank clarifying questions by Expected Value of Sample Information before "
        "implementing a task. Method — apply it INTERNALLY, showing only the final output:\n"
        "1. Frame the task: goal, response type, and the baseline you'd build right now.\n"
        "2. Propose candidate-question FAMILIES across distinct regions of the unknowns, plus "
        "one family challenging the approach itself and one hunting failure modes.\n"
        "3. For each candidate question, project 2-4 plausible answers with probabilities; "
        "estimate how much each answer would CHANGE your implementation (0-1) and the STAKES "
        "of guessing wrong (0-1); estimate derivable_prob = could you already infer the answer "
        "from the task or general knowledge?\n"
        "4. Weight each question by sqrt(uncertainty × Σ P(answer)·change·stakes), where "
        "uncertainty is answer-entropy discounted by derivable_prob. DERIVE-don't-ask: a "
        "question you can already answer is evidence, not a question — never ask it.\n"
        "5. Discard low-value and redundant questions; keep only genuinely user-specific "
        "unknowns whose answers change the code.\n\n"
        f"TASK: {task['ambiguous_prompt']}\n\n"
        f"Output ONLY your top {k} questions by weight, numbered 1..{k}."
    )
    out = pipeline.raw_chat(model, p, timeout=timeout, num_predict=700)
    return _parse_numbered(out.get("content"), k), {"raw": out.get("content", "")}


# ── one task × one arm ───────────────────────────────────────────────────────

def run_cell(task, arm, k, models, max_rounds=None):
    t0 = time.time()
    qs, meta = [], {}
    if arm == "nbq":
        qs, meta = questions_nbq(task, k, models["skill"], max_rounds=max_rounds)
    elif arm == "nbq-derive":
        qs, meta = questions_nbq(task, k, models["skill"], auto_derive=True,
                                 max_rounds=max_rounds)
    elif arm == "zeroshot":
        qs, meta = questions_zeroshot(task, k, models["skill"])
    elif arm == "prompt-evsi":
        qs, meta = questions_prompt_evsi(task, k, models["skill"])
    elif arm != "baseline":
        raise ValueError(f"unknown arm {arm!r}")

    qa = [simulate_user(task["hidden_spec"], q, models["sim"]) for q in qs]
    # nbq-derive: tombstoned derivations are established facts the solver should also see
    for d in (meta.get("derived") or []):
        qa.append({"question": d["question"], "answer": d["answer"] + " (derived)",
                   "revealed": True})
    solved = solve_and_score(task, qa, models["solver"])
    return {
        "task": task["id"], "arm": arm, "k": k,
        "questions": qs, "qa": qa,
        "revealed": sum(1 for x in qa if x["revealed"]),
        "unanswerable": sum(1 for x in qa if not x["revealed"]),
        "frac": solved["frac"], "per_test": solved["per_test"], "code": solved["code"],
        "meta": {kk: vv for kk, vv in meta.items() if kk != "derived"},
        "elapsed_s": round(time.time() - t0, 1),
    }


# ── analysis ──────────────────────────────────────────────────────────────────

def _sign_test_p(wins, losses):
    """Two-sided exact binomial sign test p-value (ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    from math import comb
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def analyze(rows):
    by = {}
    for r in rows:
        if "error" in r:
            continue
        by.setdefault(r["arm"], {})[r["task"]] = r
    base = by.get("baseline", {})
    out = {"n_tasks": len(base), "arms": {}}
    lines = [f"\n{'=' * 70}", "OBJECTIVE OUTCOMES — fraction of hidden tests passed (paired vs "
             f"baseline, n={len(base)})", "=" * 70]
    for arm in [a for a in by if a != "baseline"]:
        cells = by[arm]
        shared = sorted(set(cells) & set(base))
        deltas = [cells[t]["frac"] - base[t]["frac"] for t in shared]
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        stats = {
            "n": len(shared),
            "mean_frac": round(statistics.mean(cells[t]["frac"] for t in shared), 3) if shared else None,
            "baseline_mean_frac": round(statistics.mean(base[t]["frac"] for t in shared), 3) if shared else None,
            "mean_delta": round(statistics.mean(deltas), 3) if deltas else None,
            "wins": wins, "losses": losses, "ties": len(deltas) - wins - losses,
            "sign_p": round(_sign_test_p(wins, losses), 4),
            "unanswerable_rate": round(sum(cells[t]["unanswerable"] for t in shared)
                                       / max(1, sum(len(cells[t]["qa"]) for t in shared)), 3),
        }
        out["arms"][arm] = stats
        lines.append(f"  {arm:12} pass {stats['mean_frac']} vs baseline "
                     f"{stats['baseline_mean_frac']}  Δ {stats['mean_delta']:+.3f}  "
                     f"wins {wins}/{len(deltas)} (losses {losses}, p={stats['sign_p']})  "
                     f"unanswerable {stats['unanswerable_rate']:.0%}")
    # P6 anchor: does the skill's own value score predict objective benefit?
    nbq = by.get("nbq", {})
    shared = sorted(set(nbq) & set(base))
    qv = [statistics.mean(nbq[t]["meta"].get("q_values") or [0.0]) for t in shared]
    dp = [nbq[t]["frac"] - base[t]["frac"] for t in shared]
    if len(shared) >= 4:
        rho = spearman(qv, dp)
        out["p6_qvalue_vs_delta_rho"] = round(rho, 3) if rho is not None else None
        lines.append(f"  P6 anchor: Spearman(mean asked q_value, Δpass) = "
                     f"{out['p6_qvalue_vs_delta_rho']} (n={len(shared)})")
    print("\n".join(lines))
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task-ids", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=["baseline", "nbq", "zeroshot", "prompt-evsi"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--skill-model", default="deepseek",
                    help="model for the skill run / question generation arms")
    ap.add_argument("--solver-model", default="deepseek")
    ap.add_argument("--sim-model", default="deepseek")
    ap.add_argument("--max-rounds", type=int, default=None, help="cap the skill's rounds")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    tasks = [t for t in outcome_bank.TASKS if not args.task_ids or t["id"] in args.task_ids]
    if args.dry_run:
        t = tasks[0]
        print("DRY RUN — prompts only.\n\n=== solve (baseline) ===\n" + solve_prompt(t, []))
        print("\n=== simulator ===\n" + simulator_prompt(t["hidden_spec"], "<question>"))
        print("\n=== zeroshot ===\n" + questions_zeroshot.__doc__ if False else "")
        return 0

    models = {"skill": pipeline.resolve_alias(args.skill_model),
              "solver": pipeline.resolve_alias(args.solver_model),
              "sim": pipeline.resolve_alias(args.sim_model)}
    for role, m in models.items():
        preflight_model(m, role)

    rows, t0 = [], time.time()
    for task in tasks:
        for arm in args.arms:
            print(f"… {task['id']} × {arm}", file=sys.stderr, flush=True)
            try:
                rows.append(run_cell(task, arm, args.k, models, args.max_rounds))
            except Exception as e:
                rows.append({"task": task["id"], "arm": arm, "error": str(e)})
            if args.out:
                with open(args.out, "w") as f:
                    json.dump({"rows": rows, "partial": True, "k": args.k,
                               "models": models}, f, indent=1)
    stats = analyze(rows)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "stats": stats, "partial": False, "k": args.k,
                       "models": models, "elapsed_s": round(time.time() - t0, 1)},
                      f, indent=1)
        print(f"saved {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
