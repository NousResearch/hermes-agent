#!/usr/bin/env python3
"""iterate.py — minimal iterate-context wrapper around the info-gain ranker (KISS v1).

The skill ranks clarifying questions; this wrapper builds context to convergence and then
responds. The loop (one continuously-growing context — append-only):

    tombstones = []                                  # answered facts + known gaps
    for round in range(max_rounds):
        evidence = [t.evidence for t in tombstones]  # the shared growing context
        ranked   = infogain.run(problem, evidence)   # rank given everything known so far
        above    = [q in ranked if value >= floor and not already answered][:K]   # top-K BY RANK
        if not above: stop "converged"
        for q in above: tombstones += grounded_answer(q)   # `ask` skill, all tools, distilled
    final = respond(problem, evidence)

Design notes:
- Selection is **top-K by rank** (sidesteps the absolute-threshold problem; floor only ends the loop).
- The grounded ANSWERER is the `ask` skill (`dispatch_single`: full Hermes agent, all tools, isolated
  context) — only the distilled fact returns, so the loop context stays lean. Runs INSIDE the hermes
  container (`hermes` binary). NOT_FOUND is recorded as a plain gap (no revival machinery yet — YAGNI).
- `answerer`/`responder` are injectable so the loop logic is testable on the host with a mock
  (no hermes / no model calls): `--dry-run`.

Usage (inside the container):
    python3 scripts/iterate.py --problem "Add authentication to my web app"
    python3 scripts/iterate.py --problem "..." --validate top --k 2      # end-to-end test (#21)
On the host (loop logic only): python3 scripts/iterate.py --problem "..." --dry-run
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HOME, "skills", "productivity", "ask", "scripts"))

import infogain  # noqa: E402

try:
    from model_utils import dispatch_single, resolve_alias  # noqa: E402
    _HAVE_ASK = True
except Exception as _e:  # the `ask` skill (model_utils) is required for live (non-mock) runs
    _HAVE_ASK = False
    _ASK_ERR = str(_e)

DEFAULTS = {
    "k": 6, "max_rounds": 3, "floor": 0.12,
    "answer_model": "glm", "answer_provider": "ollama-glm",
    "answer_toolsets": "file,web,terminal", "answer_timeout": 300, "answer_max_turns": None,
    # Where the grounded answerer researches. None = inherit the caller's cwd (the user's project
    # in a real session). Pin it when the wrapper runs detached from the project — a live test showed
    # the answerer otherwise researches the install dir and misses the actual project.
    "answer_cwd": None,
    "responder_model": "glm", "responder_provider": "ollama-glm", "responder_timeout": 300,
    "responder_toolsets": "", "responder_cwd": None,  # responder usually synthesizes from facts (no tools)
}


def _rank_cfg():
    cfg = dict(infogain.DEFAULTS)
    cfg.update(infogain.MODES.get("focus", {}))
    cfg["mode"], cfg["max_rounds"] = "focus", 1
    return cfg


def rank(problem, evidence, rank_cfg):
    """Ranked candidate questions (all_scored, value-desc) given the growing context."""
    res = infogain.run(problem, rank_cfg, evidence=evidence)
    ranked = sorted(res.get("all_scored", []), key=lambda r: r.get("value", 0.0), reverse=True)
    return ranked


# ── grounded answerer + responder (live: the `ask` skill) ─────────────────────

def _strip_suggestion(text):
    """Drop a trailing `SUGGESTION:{...}` block (hermes's interactive next-step artifact)."""
    i = text.rfind("SUGGESTION:{")
    return text[:i].rstrip() if i != -1 else text


def _extract(r):
    """Best text from a dispatch_single result, as (text, error).

    Works around a false-positive in model_utils.is_api_error(): a legitimate response that is
    ABOUT errors / rate-limits / status codes can match ≥2 error keywords, so dispatch_single
    files the whole answer under `error` ("API error: <long text>") and blanks `content`. A long
    payload is a real response misclassified — recover it; a short one is a genuine API error.
    """
    text = (r.get("content") or "").strip()
    if text:
        return _strip_suggestion(text), None
    err = (r.get("error") or "").strip()
    if err.startswith("API error: ") and len(err) > 200:  # long => real response, not an error
        return _strip_suggestion(err[len("API error: "):].strip()), None
    return "", (err or "empty response")


def grounded_answer(question, problem, evidence, cfg):
    """(found, text) — research one question with a full Hermes agent (all tools), distilled."""
    facts = "\n".join(f"- {e}" for e in evidence) or "(none yet)"
    prompt = (f"TASK: {problem}\n\nEstablished so far:\n{facts}\n\n"
              f"Research and answer THIS question CONCISELY (1-3 sentences), using any tools you need:\n"
              f"  {question}\n\n"
              f"If you genuinely cannot determine it, reply EXACTLY: NOT_FOUND: <brief reason>.")
    r = dispatch_single(resolve_alias(cfg["answer_model"]), prompt, "", cfg["answer_toolsets"],
                        cfg["answer_max_turns"], cfg["answer_timeout"], cfg["answer_provider"],
                        cwd=cfg.get("answer_cwd"))
    text, err = _extract(r)
    if err:
        return False, f"research error: {err}"
    if "NOT_FOUND" in text.upper()[:48]:
        return False, text.split(":", 1)[-1].strip() if ":" in text else text
    return True, text


def respond(problem, evidence, cfg):
    """Final response over the enriched context (no tools — synthesize from established facts)."""
    facts = "\n".join(f"- {e}" for e in evidence) or "(none)"
    prompt = (f"TASK: {problem}\n\nEstablished facts and known gaps:\n{facts}\n\n"
              f"Produce the best possible response to the task using what's established. "
              f"State any assumptions you make for unresolved gaps. Be direct and useful.")
    r = dispatch_single(resolve_alias(cfg["responder_model"]), prompt, "", cfg.get("responder_toolsets", ""),
                        None, cfg["responder_timeout"], cfg["responder_provider"],
                        cwd=cfg.get("responder_cwd"))
    text, err = _extract(r)
    return text or f"(no response: {err})"


def _tombstone(q, found, text):
    qt = q.get("question", "")
    if found:
        return {"question": qt, "status": "ANSWERED", "fact": text, "evidence": f"{qt} -> {text}"}
    return {"question": qt, "status": "NOT_FOUND", "fact": text,
            "evidence": f"{qt} -> (known gap: {text})"}


# ── the loop ──────────────────────────────────────────────────────────────────

def iterate(problem, cfg=None, answerer=None, responder=None, progress=None):
    cfg = {**DEFAULTS, **(cfg or {})}
    answerer = answerer or grounded_answer
    responder = responder or respond
    progress = progress or (lambda m: None)
    rank_cfg = _rank_cfg()

    tombstones, answered = [], set()
    rounds, stop_reason, k_capped = 0, None, False
    for rnd in range(cfg["max_rounds"]):
        rounds = rnd + 1
        evidence = [t["evidence"] for t in tombstones]
        ranked = rank(problem, evidence, rank_cfg)
        above = [r for r in ranked
                 if r.get("value", 0.0) >= cfg["floor"] and r.get("question") not in answered]
        if not above:
            stop_reason = "converged (no question above floor)"
            break
        if len(above) > cfg["k"]:
            k_capped = True  # more worthwhile questions than K — the cap rate-limited this round
        top = above[: cfg["k"]]
        progress(f"round {rounds}: {len(above)} above floor, researching top {len(top)} "
                 f"(best value={top[0].get('value', 0):.2f})")
        for q in top:
            found, text = answerer(q, problem, evidence, cfg)
            tombstones.append(_tombstone(q, found, text))
            answered.add(q.get("question"))
            evidence = [t["evidence"] for t in tombstones]  # context grows immediately
            progress(f"  {'✓' if found else '∅'} {q.get('question', '')[:60]}")
    else:
        stop_reason = "max_rounds reached"

    final = responder(problem, [t["evidence"] for t in tombstones], cfg)
    artificial_cap = k_capped or stop_reason == "max_rounds reached"
    return {
        "problem": problem, "final": final, "tombstones": tombstones,
        "rounds": rounds, "stop_reason": stop_reason,
        "k_capped": k_capped, "artificial_cap_bound": artificial_cap,
        "n_answered": sum(1 for t in tombstones if t["status"] == "ANSWERED"),
        "n_gaps": sum(1 for t in tombstones if t["status"] == "NOT_FOUND"),
    }


def validate_selection(problem, which, k, cfg=None, answerer=None, responder=None, progress=None):
    """End-to-end test (#21): respond after answering a selection of round-1 ranked questions.

    which: "baseline" (answer none — respond directly) | "top" (top-k) | "bottom" (bottom-k).
    Single round, so the three are directly comparable for a blind A/B judge.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    answerer = answerer or grounded_answer
    responder = responder or respond
    progress = progress or (lambda m: None)
    if which == "baseline":
        sel = []
    else:
        ranked = rank(problem, [], _rank_cfg())
        sel = ranked[:k] if which == "top" else list(reversed(ranked[-k:]))
    progress(f"{which}-{k}: " + (" | ".join(q.get("question", "")[:40] for q in sel) or "(no clarification)"))
    tombstones, evidence = [], []
    for q in sel:
        found, text = answerer(q, problem, evidence, cfg)
        tombstones.append(_tombstone(q, found, text))
        evidence = [t["evidence"] for t in tombstones]
    return {"which": which, "k": k, "selected": [q.get("question") for q in sel],
            "values": [round(q.get("value", 0), 3) for q in sel],
            "final": responder(problem, evidence, cfg), "tombstones": tombstones}


# ── mock answerer/responder for host-side loop testing (no hermes) ────────────

def _mock_answerer(q, problem, evidence, cfg):
    return True, f"[mock fact for: {q.get('question', '')[:40]}]"


def _mock_responder(problem, evidence, cfg):
    return f"[mock response over {len(evidence)} established fact(s)]"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--problem", required=True)
    p.add_argument("--k", type=int, default=DEFAULTS["k"])
    p.add_argument("--max-rounds", type=int, default=DEFAULTS["max_rounds"])
    p.add_argument("--floor", type=float, default=DEFAULTS["floor"])
    p.add_argument("--validate", choices=["baseline", "top", "bottom"],
                   help="end-to-end test: respond after answering baseline/top-k/bottom-k.")
    p.add_argument("--dry-run", action="store_true", help="mock answerer/responder (host, no hermes).")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = {"k": args.k, "max_rounds": args.max_rounds, "floor": args.floor}
    answerer = _mock_answerer if args.dry_run else None
    responder = _mock_responder if args.dry_run else None
    if not args.dry_run and not _HAVE_ASK:
        print(f"live runs need the `ask` skill (model_utils): {_ASK_ERR}", file=sys.stderr)
        return 2
    prog = lambda m: print(f"… {m}", file=sys.stderr, flush=True)

    t0 = time.time()
    if args.validate:
        out = validate_selection(args.problem, args.validate, args.k, cfg, answerer, responder, prog)
    else:
        out = iterate(args.problem, cfg, answerer, responder, prog)
    out["elapsed_s"] = round(time.time() - t0, 1)

    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"\n=== stop: {out.get('stop_reason', out.get('which'))} · "
              f"rounds={out.get('rounds', 1)} · answered={out.get('n_answered', '-')} "
              f"gaps={out.get('n_gaps', '-')} · {out['elapsed_s']}s ===\n")
        for t in out["tombstones"]:
            print(f"  [{t['status']}] {t['question'][:70]}\n      {t['fact'][:120]}")
        print(f"\n--- FINAL RESPONSE ---\n{out['final']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
