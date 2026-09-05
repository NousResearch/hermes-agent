#!/usr/bin/env python3
"""Live benchmark v5 — DISCOVERY-BOUND tasks at 830 tools. Opus 4.8, bridge vs listing.

Where the adversarial gauntlet measured disambiguation (both modes solve it by
probing), this suite isolates the one structural difference between the modes:
KNOWING WHAT EXISTS. Three task families:

  D* discovery  — the tool exists but the prompt shares ZERO lexical surface
                  with its name/description (BM25-hostile paraphrase).
  A* absence    — no tool does what's asked (verified against all 830).
                  Correct behavior = confident refusal, no hallucinated calls.
  S* survey     — "which of these five things can we do?" — breadth question.

Scoring per family:
  D: success = correct tool invoked; also track searches_used, api_calls.
  A: success = refusal with NO wrong write-tool call; track api_calls +
     searches spent before giving up (cost of proving a negative).
  S: success = final answer classifies all five capabilities correctly.
"""
from __future__ import annotations

import json, os, shutil, sys, time, traceback
from pathlib import Path
from typing import Any, Dict, List

_THIS_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_WORKTREE_ROOT))
sys.path.insert(0, str(_THIS_DIR))

import tool_search_livetest as base
from tool_search_livetest_ue_hard import register_epic_tools_adversarial

N_REPS = int(os.environ.get("TS_BENCH_REPS", "3"))

SCENARIOS: List[Dict[str, Any]] = [
    # ---- D: discovery under paraphrase (tool exists; zero name overlap)
    {
        "id": "D1_sparkly_brighter",
        "prompt": ("The sparkly effect on the actor Torch_3 looks too dim. Make it brighter — "
                   "its glow strength setting should go to 5.0. Then say done."),
        "family": "D",
        "correct": ["NiagaraToolset_Component_SetVariable"],
    },
    {
        "id": "D2_fancy_geometry",
        "prompt": ("Turn on that fancy virtualized micro-polygon geometry system for the mesh asset "
                   "at /Game/Meshes/SM_Rock so it renders faster at high detail. Then say done."),
        "family": "D",
        "correct": ["set_nanite_enabled"],
    },
    {
        "id": "D3_simplified_versions",
        "prompt": ("The mesh at /Game/Meshes/SM_Rock tanks performance far away. Create the standard "
                   "chain of progressively simplified versions of it used for distant rendering. Then say done."),
        "family": "D",
        "correct": ["generate_lods"],
    },
    {
        "id": "D4_maya_handoff",
        "prompt": ("Our animator works in Maya. Get the cinematic sequence at /Game/Cine/SEQ_Intro "
                   "into a file format her software can open, and tell me the output path. Then say done."),
        "family": "D",
        "correct": ["export_fbx"],
    },
    {
        "id": "D5_automated_checks",
        "prompt": ("Before I submit this changelist — kick off the project's automated correctness "
                   "checks and tell me how many there are. Then say done."),
        "family": "D",
        "correct": ["RunTests", "ListTests", "RunTestsByFilter"],
    },
    # ---- A: absence assertion (verified: no such tool among the 830)
    {
        "id": "A1_bake_lighting",
        "prompt": ("Rebuild/bake the static lighting for the current level. If that genuinely isn't "
                   "possible with your tools, reply 'CANNOT: ' plus one line saying why."),
        "family": "A",
    },
    {
        "id": "A2_undo",
        "prompt": ("Undo my last change in the editor. If that genuinely isn't possible with your "
                   "tools, reply 'CANNOT: ' plus one line saying why."),
        "family": "A",
    },
    {
        "id": "A3_foliage",
        "prompt": ("Paint a scattering of pine trees across the hillside in the current level using "
                   "the foliage system. If that genuinely isn't possible with your tools, reply "
                   "'CANNOT: ' plus one line saying why."),
        "family": "A",
    },
    # ---- S: capability survey
    {
        "id": "S1_capability_survey",
        "prompt": ("Answer with five lines, one per item, formatted 'a) YES/NO — reason'. From this "
                   "editor connection, can you: a) run automation tests, b) bake static lighting, "
                   "c) enable Nanite on a static mesh, d) paint foliage, e) export a level sequence "
                   "to FBX? Do not call any editing tools — just determine capability."),
        "family": "S",
        "survey_truth": {"a": True, "b": False, "c": True, "d": False, "e": True},
    },
]


def score_survey(resp: str, truth: Dict[str, bool]) -> bool:
    import re
    resp_l = resp.lower()
    for key, expected in truth.items():
        m = re.search(rf"\b{key}\)?\s*[:\-—]?\s*(yes|no)", resp_l)
        if not m:
            return False
        if (m.group(1) == "yes") != expected:
            return False
    return True


def run_one(scenario, mode, rep, out_dir: Path):
    model = os.environ.get("TS_UE_MODEL", "anthropic/claude-opus-4.8")
    lmax = int(os.environ.get("TS_UE_LISTING_MAX", "30000"))
    hermes_home = base.setup_isolated_home(
        True, listing=("auto" if mode == "listing" else "off"),
        listing_max_tokens=lmax, model=model)
    os.environ["HERMES_HOME"] = str(hermes_home)
    base.reset_module_state()
    register_epic_tools_adversarial()

    from tools.registry import registry
    original_dispatch = registry.dispatch
    call_log: List[str] = []

    def logging_dispatch(name, args, **kw):
        call_log.append(name)
        return original_dispatch(name, args, **kw)
    registry.dispatch = logging_dispatch

    usage_log: List[Dict[str, Any]] = []
    started = time.time()
    error = None
    final_response = ""
    messages_out: List[Dict[str, Any]] = []
    _orig_norm = None
    try:
        from run_agent import AIAgent
        agent = AIAgent(provider="openrouter", model=model, quiet_mode=True,
                        save_trajectories=False, skip_context_files=True,
                        skip_memory=True, platform="cli", max_iterations=15)
        import agent.conversation_loop as _cl
        _orig_norm = _cl.normalize_usage
        def _norm_spy(raw, **kw):
            cu = _orig_norm(raw, **kw)
            try:
                usage_log.append({"prompt_tokens": cu.prompt_tokens})
            except Exception:
                pass
            return cu
        _cl.normalize_usage = _norm_spy
        result = agent.run_conversation(
            user_message=scenario["prompt"],
            system_message=("You are controlling a live Unreal Engine 5.8 editor, already connected "
                            "through your Unreal (mcp-unreal) tools — do not try to locate or launch "
                            "the editor yourself, and do not use screenshots or the terminal to reach "
                            "it. If a capability doesn't exist in your tools, say so plainly. Be concise."),
        )
        if isinstance(result, dict):
            final_response = result.get("final_response") or ""
            messages_out = result.get("messages") or []
    except Exception:
        error = traceback.format_exc()
    finally:
        registry.dispatch = original_dispatch
        if _orig_norm is not None:
            try:
                import agent.conversation_loop as _cl2
                _cl2.normalize_usage = _orig_norm
            except Exception:
                pass

    elapsed = time.time() - started
    bridge_call_log = base._extract_bridge_calls(messages_out)
    searches = sum(1 for b in bridge_call_log if b["name"] == "tool_search")
    ue_calls = [c for c in call_log if c.lower() not in ("tool_search", "tool_describe", "tool_call")
                and not c.startswith(("skills_list", "skill_", "todo", "memory"))]
    # writes = non-read calls (for absence scoring: any UE write = hallucinated capability)
    write_calls = [c for c in ue_calls if not any(v in c.lower() for v in
                   ("get", "list", "find", "search", "has_", "is_", "can_", "observe", "snapshot"))]

    fam = scenario["family"]
    if fam == "D":
        success = any(any(s.lower() in c.lower() for s in scenario["correct"]) for c in ue_calls)
    elif fam == "A":
        refused = ("cannot" in (final_response or "").lower() or "can't" in (final_response or "").lower()
                   or "not possible" in (final_response or "").lower() or "no tool" in (final_response or "").lower())
        success = refused and not write_calls and error is None
    else:  # S
        success = score_survey(final_response or "", scenario["survey_truth"]) and not write_calls

    rec = {
        "scenario_id": scenario["id"], "family": fam, "mode": mode, "rep": rep,
        "elapsed_seconds": round(elapsed, 2),
        "api_calls": len(usage_log),
        "searches_used": searches,
        "prompt_tokens_total": sum(u["prompt_tokens"] or 0 for u in usage_log),
        "ue_calls": [c[-60:] for c in ue_calls][:15],
        "write_calls": [c[-60:] for c in write_calls][:10],
        "bridge_queries": [(b.get("args") or {}).get("query") for b in bridge_call_log if b["name"] == "tool_search"][:10],
        "success": bool(success), "error": error,
        "final_response": base._redact_secrets(final_response)[:400],
    }
    (out_dir / f"{scenario['id']}__{mode}__rep{rep}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    shutil.rmtree(Path(os.environ["HERMES_HOME"]).parent, ignore_errors=True)
    return rec


def main():
    out_dir = _THIS_DIR / "out_ue_disc"
    out_dir.mkdir(exist_ok=True)
    modes = [m for m in os.environ.get("TS_UE_MODES", "listing,bridge").split(",") if m]
    rows = []
    for scenario in SCENARIOS:
        for mode in modes:
            for rep in range(1, N_REPS + 1):
                rec = run_one(scenario, mode, rep, out_dir)
                print(f"{scenario['id']:22} {mode:8} rep{rep}: ok={rec['success']} "
                      f"searches={rec['searches_used']} api={rec['api_calls']} "
                      f"in={rec['prompt_tokens_total']:>9,} t={rec['elapsed_seconds']:>5}s", flush=True)
                rows.append(rec)
    name = os.environ.get("TS_UE_SUMMARY", "_ue_discovery.json")
    (out_dir / name).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("done ->", out_dir / name)


if __name__ == "__main__":
    main()
