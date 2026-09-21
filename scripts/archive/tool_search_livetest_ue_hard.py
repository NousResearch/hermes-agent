#!/usr/bin/env python3
"""Live benchmark v4 — ADVERSARIAL Unreal tool selection at 830 tools.

Differences from tool_search_livetest_ue.py (which had a ceiling effect):

1. Scenarios target CONFUSION CLUSTERS in Epic's real catalog — tools with
   near-identical names/purposes in different toolsets (StaticMesh vs
   SkeletalMesh set_material; GameplayTags vs ActorTools vs GameplayCue tags;
   CurveTable vs DataTable rows; Niagara Component vs System SetVariable;
   4 capture variants). Prompts avoid quoting exact tool names.
2. TYPE-AWARE mocks: calling a tool against the wrong asset/actor type
   returns a realistic editor error (e.g. "SM_Rock is not a SkeletalMesh"),
   so wrong picks visibly fail instead of silently succeeding.
3. STRICT scoring per run:
     - first_correct: the FIRST non-bridge tool call is in the correct set
     - final_correct: a correct tool was called with the right asset arg
     - wrong_calls:   # of calls to distractor tools
     - success = final_correct AND wrong_calls == 0 (clean solve)

Env: TS_UE_MODEL, TS_BENCH_REPS, TS_UE_MODES (eager,bridge,listing),
     TS_UE_SUMMARY. Scale is always "full" (830 tools).
"""
from __future__ import annotations

import json, os, re, shutil, sys, time, traceback
from pathlib import Path
from typing import Any, Dict, List

_THIS_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_WORKTREE_ROOT))
sys.path.insert(0, str(_THIS_DIR))

import tool_search_livetest as base
from tool_search_livetest_ue import load_epic_tools, _SANITIZE  # reuse loader

N_REPS = int(os.environ.get("TS_BENCH_REPS", "2"))

# ---------------------------------------------------------------------------
# Type-aware mock world
# ---------------------------------------------------------------------------

WORLD = {
    "/Game/Meshes/SM_Rock": "StaticMesh",
    "/Game/Chars/SK_Guard": "SkeletalMesh",
    "/Game/Data/CT_Damage": "CurveTable",
    "/Game/Data/DT_Loot": "DataTable",
    "/Game/FX/NS_Sparks": "NiagaraSystem",
    "Torch_3": "Actor",         # has a NiagaraComponent
    "Crate_2": "Actor",
}

def _mentioned_path(kwargs: Dict[str, Any]) -> str:
    blob = json.dumps(kwargs)
    for p in WORLD:
        if p in blob:
            return p
    return ""

def make_mock(sanitized_name: str):
    n = sanitized_name.lower()

    def _h(*a, **kw):
        path = _mentioned_path(kw)
        t = WORLD.get(path, "")
        # Wrong-type guards mirror the real editor's failures.
        if "skeletalmeshtools" in n and t and t != "SkeletalMesh":
            return json.dumps({"error": f"{path} is a {t}, not a SkeletalMesh. Use the StaticMesh tools."})
        if "staticmeshtools" in n and t and t != "StaticMesh":
            return json.dumps({"error": f"{path} is a {t}, not a StaticMesh."})
        if "curvetabletools" in n and t and t != "CurveTable":
            return json.dumps({"error": f"{path} is a {t}, not a CurveTable."})
        if "datatabletools" in n and t and t != "DataTable":
            return json.dumps({"error": f"{path} is a {t}, not a DataTable."})
        if "niagaratoolset_system" in n and t == "Actor":
            return json.dumps({"error": f"{path} is a level actor, not a NiagaraSystem asset. Use the Niagara component tools for actors."})
        if "niagaratoolset_component" in n and t == "NiagaraSystem":
            return json.dumps({"error": f"{path} is a NiagaraSystem asset, not an actor with a NiagaraComponent."})
        # Coherent world reads so the model can chain calls.
        if "find_actors" in n or "getvisibleactors" in n or "get_outliner" in n:
            blob = json.dumps(kw)
            actors = [{"label": "Torch_3", "path": "/Game/Map:PersistentLevel.Torch_3",
                       "class": "Actor", "components": ["NiagaraComponent 'FX_Flame'"]},
                      {"label": "Crate_2", "path": "/Game/Map:PersistentLevel.Crate_2",
                       "class": "StaticMeshActor"}]
            if "Torch" in blob:
                actors = actors[:1]
            elif "Crate" in blob:
                actors = actors[1:]
            return json.dumps({"result": actors})
        if "get_components" in n:
            blob = json.dumps(kw)
            if "Torch" in blob:
                return json.dumps({"result": [{"name": "FX_Flame", "class": "NiagaraComponent"},
                                              {"name": "PointLight0", "class": "PointLightComponent"}]})
            return json.dumps({"result": [{"name": "StaticMeshComponent0", "class": "StaticMeshComponent"}]})
        if "getuservariables" in n or "list_rows" in n or "listtags" in n or "get_tags" in n:
            return json.dumps({"result": [{"name": "Brightness", "type": "float", "value": 1.0}]})
        if any(v in n for v in ("get", "list", "find", "search", "has_", "is_", "can_")):
            return json.dumps({"result": [{"name": "Entry_0", "value": 1.0}]})
        if "capture" in n or "screenshot" in n:
            return json.dumps({"result": {"image_path": "/tmp/ue_capture_0001.png"}})
        return json.dumps({"result": {"ok": True}})
    return _h


def register_epic_tools_adversarial() -> int:
    from tools.registry import registry
    tools = load_epic_tools("full")
    for tdef in tools:
        registry.register(
            name=tdef["name"], toolset="mcp-unreal",
            schema={"name": tdef["name"], "description": tdef["description"],
                    "parameters": tdef["parameters"]},
            handler=make_mock(tdef["name"]),
        )
    return len(tools)


# ---------------------------------------------------------------------------
# Adversarial scenarios: (prompt, correct substrings, distractor substrings)
# Substrings match against sanitized full tool names, case-insensitive.
# ---------------------------------------------------------------------------

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "V1_static_material",
        "prompt": "Assign the material /Game/Mats/M_Stone to slot 0 of the mesh asset at /Game/Meshes/SM_Rock. Then say done.",
        "correct": ["StaticMeshTools_set_material"],
        "distractors": ["SkeletalMeshTools_set_material", "MaterialTools_create_material", "MaterialInstanceTools"],
    },
    {
        "id": "V2_skeletal_material",
        "prompt": "Assign the material /Game/Mats/M_Cloth to slot 1 of the character mesh at /Game/Chars/SK_Guard. Then say done.",
        "correct": ["SkeletalMeshTools_set_material"],
        "distractors": ["StaticMeshTools_set_material"],
    },
    {
        "id": "V3_curvetable_row",
        "prompt": "Add a row named 'Heavy' to the table asset at /Game/Data/CT_Damage with value 42 at time 0. Then say done.",
        "correct": ["CurveTableTools_add_row"],
        "distractors": ["DataTableTools_add_rows", "DataTableTools_set_rows"],
    },
    {
        "id": "V4_project_tag",
        "prompt": "Register a new gameplay tag 'Combat.Stun' in the project's tag registry so designers can use it. Then say done.",
        "correct": ["GameplayTagsToolset_AddTag"],
        "distractors": ["ActorTools_add_tag", "GameplayCueToolset_AddCueTag"],
    },
    {
        "id": "V5_actor_tag",
        "prompt": "Mark the level actor named Crate_2 with the tag 'loot' so my spawner script can find it. Then say done.",
        "correct": ["ActorTools_add_tag"],
        "distractors": ["GameplayTagsToolset_AddTag", "GameplayCueToolset_AddCueTag"],
    },
    {
        "id": "V6_niagara_component",
        "prompt": "The particle effect on the actor Torch_3 is too dim — set its 'Brightness' user parameter to 5.0 on that actor's effect component. Then say done.",
        "correct": ["NiagaraToolset_Component_SetVariable"],
        "distractors": ["NiagaraToolset_System_AddUserVariables", "NiagaraToolset_System_AddSetParameterEntry",
                        "DataflowAgentToolset_SetVariable", "NiagaraToolset_System"],
    },
    {
        "id": "V7_niagara_system_asset",
        "prompt": "Add a user-exposed float called 'WindStrength' to the effect asset at /Game/FX/NS_Sparks itself, so every instance can override it. Then say done.",
        "correct": ["NiagaraToolset_System_AddUserVariables"],
        "distractors": ["NiagaraToolset_Component_SetVariable", "DataflowAgentToolset_AddVariable"],
    },
    {
        "id": "V8_widget_screenshot",
        "prompt": "Capture an image of ONLY the Details panel widget (not the whole editor, not the 3D viewport). Tell me the file path. Then say done.",
        "correct": ["SlateInspectorToolset_Screenshot"],
        "distractors": ["CaptureViewport", "CaptureEditorImage", "CaptureAssetImage"],
    },
    {
        "id": "V9_save_actor",
        "prompt": "I just edited the actor Crate_2 in the level. Persist exactly that actor's changes to disk (not a full save-all). Then say done.",
        "correct": ["SceneTools_save_actor"],
        "distractors": ["AssetTools_save_assets", "ConfigSettingsToolset_SaveSection"],
    },
    {
        "id": "V10_zero_keyword",
        "prompt": "Something in my level list panel — the thing showing all the stuff placed in the world — seems stale. Get me whatever that panel's current contents are. Then say done.",
        "correct": ["SceneTools_find_actors", "GetVisibleActors", "get_outliner"],
        "distractors": ["GetContentBrowserPath", "SetContentBrowserPath"],
    },
]


def run_one(scenario, mode, rep, out_dir: Path):
    enabled = mode in ("bridge", "listing")
    model = os.environ.get("TS_UE_MODEL", "anthropic/claude-opus-4.8")
    lmax = int(os.environ.get("TS_UE_LISTING_MAX", "30000"))
    hermes_home = base.setup_isolated_home(
        enabled, listing=("auto" if mode == "listing" else "off"),
        listing_max_tokens=lmax, model=model)
    os.environ["HERMES_HOME"] = str(hermes_home)
    base.reset_module_state()
    n_registered = register_epic_tools_adversarial()

    from tools.registry import registry
    original_dispatch = registry.dispatch
    call_log: List[Dict[str, Any]] = []

    def logging_dispatch(name, args, **kw):
        call_log.append({"name": name, "args": args})
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
            system_message=("You are controlling a live Unreal Engine 5.8 editor. The editor is "
                            "already running and connected through your Unreal (mcp-unreal) tools — "
                            "do not try to locate or launch the editor yourself. Choose tools "
                            "carefully: several toolsets contain similarly-named tools for "
                            "different object types. Be concise."),
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
    # underlying calls: registry log + tool_call unwraps (registry sees both; dedupe consecutive)
    ue_calls = [c for c in call_log if c["name"].lower() not in ("tool_search", "tool_describe", "tool_call")
                and not c["name"].startswith(("skills_list", "skill_", "todo", "memory"))]

    def matches(name, subs):
        return any(s.lower() in name.lower() for s in subs)

    correct, distract = scenario["correct"], scenario["distractors"]
    first_ue = next((c["name"] for c in ue_calls), "")
    first_correct = matches(first_ue, correct) if first_ue else False
    final_correct = any(matches(c["name"], correct) for c in ue_calls)
    wrong_calls = sum(1 for c in ue_calls if matches(c["name"], distract))
    success = final_correct and wrong_calls == 0

    rec = {
        "scenario_id": scenario["id"], "mode": mode, "rep": rep,
        "n_tools_registered": n_registered,
        "elapsed_seconds": round(elapsed, 2),
        "api_calls": len(usage_log),
        "prompt_tokens_total": sum(u["prompt_tokens"] or 0 for u in usage_log),
        "first_tool": first_ue.split("_")[-2:] if first_ue else None,
        "first_correct": first_correct, "final_correct": final_correct,
        "wrong_calls": wrong_calls, "success": bool(success),
        "ue_calls": [c["name"][-70:] for c in ue_calls][:20],
        "bridge_calls": [(b["name"], (b.get("args") or {}).get("query") or (b.get("args") or {}).get("name")) for b in bridge_call_log][:20],
        "error": error,
        "final_response": base._redact_secrets(final_response)[:300],
    }
    (out_dir / f"{scenario['id']}__{mode}__rep{rep}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    shutil.rmtree(Path(os.environ["HERMES_HOME"]).parent, ignore_errors=True)
    return rec


def main():
    out_dir = _THIS_DIR / "out_ue_hard"
    out_dir.mkdir(exist_ok=True)
    modes = [m for m in os.environ.get("TS_UE_MODES", "listing,bridge").split(",") if m]
    rows = []
    for scenario in SCENARIOS:
        for mode in modes:
            for rep in range(1, N_REPS + 1):
                rec = run_one(scenario, mode, rep, out_dir)
                print(f"{scenario['id']:22} {mode:8} rep{rep}: 1st={'Y' if rec['first_correct'] else 'n'} "
                      f"final={'Y' if rec['final_correct'] else 'n'} wrong={rec['wrong_calls']} "
                      f"ok={rec['success']} api={rec['api_calls']} in={rec['prompt_tokens_total']:>9,} "
                      f"t={rec['elapsed_seconds']:>5}s", flush=True)
                rows.append(rec)
    name = os.environ.get("TS_UE_SUMMARY", "_ue_hard.json")
    (out_dir / name).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("done ->", out_dir / name)


if __name__ == "__main__":
    main()
