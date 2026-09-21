#!/usr/bin/env python3
"""Live benchmark v3: Epic Unreal Engine 5.8 MCP surface (830 REAL schemas), replayed.

Registers the actual tool schemas captured live from Epic's UE 5.8
ModelContextProtocol + AllToolsets plugins (probe_raw_5.8.0_alltoolsets.json,
probe date 2026-07-02) into the Hermes tool registry with mock handlers,
then runs UE-realistic scenarios in three modes:

  eager    — all schemas in the tools array (at 830 tools: ~165K tokens)
  bridge   — tool_search bridge, no listing (old behavior)
  listing  — bridge + skills-style catalog listing (PR #67034)

Catalog scale is controlled by TS_UE_SCALE:
  "editor"  — EditorApp + Scene + Primitive + Actor toolsets (~65 tools)
  "full"    — all 52 toolsets / 830 tools

Env: TS_BENCH_REPS (default 2), TS_UE_MODES, TS_UE_SCALE, TS_UE_SUMMARY.
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

PROBE = "/tmp/ue-bridge-probe/docs/epic_mcp/probe_raw_5.8.0_alltoolsets.json"
N_REPS = int(os.environ.get("TS_BENCH_REPS", "2"))

EDITOR_TOOLSETS = (
    "EditorToolset.EditorAppToolset",
    "editor_toolset.toolsets.scene.SceneTools",
    "editor_toolset.toolsets.primitive.PrimitiveTools",
    "editor_toolset.toolsets.actor.ActorTools",
)

_SANITIZE = re.compile(r"[^A-Za-z0-9_]")


def _mock_result(tool_name: str) -> str:
    """Plausible success payload keyed on verb-ish name shape."""
    short = tool_name.rsplit("_", 1)[-1].lower()
    if any(v in tool_name.lower() for v in ("get", "list", "find", "search", "query", "is_", "can_", "checked")):
        return json.dumps({"result": [{"name": "Cube_1", "path": "/Game/Level:PersistentLevel.Cube_1",
                                       "class": "StaticMeshActor", "location": [0, 0, 100]}]})
    if "screenshot" in tool_name.lower() or "capture" in tool_name.lower():
        return json.dumps({"result": {"image_path": "/tmp/ue_viewport_0001.png", "width": 1280, "height": 720}})
    return json.dumps({"result": {"ok": True, "op": short, "actor": "/Game/Level:PersistentLevel.Cube_1"}})


def load_epic_tools(scale: str) -> List[Dict[str, Any]]:
    with open(PROBE, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for ts_name, ts in raw["toolsets"].items():
        if not isinstance(ts, dict) or not ts.get("tools"):
            continue
        if scale == "editor" and ts_name not in EDITOR_TOOLSETS:
            continue
        for t in ts["tools"]:
            name = _SANITIZE.sub("_", t.get("name", ""))
            if not name:
                continue
            out.append({
                "name": name,
                "description": t.get("description", "") or "",
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
            })
    return out


def register_epic_tools(scale: str) -> int:
    from tools.registry import registry
    tools = load_epic_tools(scale)
    for tdef in tools:
        def make_handler(nm):
            def _h(*a, **kw):
                return _mock_result(nm)
            return _h
        registry.register(
            name=tdef["name"],
            toolset="mcp-unreal",
            schema={"name": tdef["name"], "description": tdef["description"],
                    "parameters": tdef["parameters"]},
            handler=make_handler(tdef["name"]),
        )
    return len(tools)


# Expected tools use SUBSTRING match against sanitized names (full names are
# long dotted paths, e.g. editor_toolset_toolsets_scene_SceneTools_..._add_to_scene_from_class).
SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "U1_spawn_named",
        "description": "Direct ask naming the operation (spawn actor)",
        "prompt": ("Spawn a PointLight actor in the level at location x=0 y=0 z=300. "
                   "Then tell me you're done. Don't do anything else."),
        "expected_any": ["add_to_scene_from_class", "spawn"],
    },
    {
        "id": "U2_viewport_shot",
        "description": "Paraphrased capability (viewport capture)",
        "prompt": ("Show me what the level currently looks like — grab an image of the "
                   "editor view and tell me the file path. Nothing else."),
        "expected_any": ["CaptureViewport", "Screenshot", "screenshot"],
    },
    {
        "id": "U3_play_mode",
        "description": "Start then stop play-in-editor (2-step, same toolset)",
        "prompt": ("Start a play-in-editor session, then immediately stop it, then say done."),
        "expected_any": ["StartPIE"],
        "expected_any_2": ["StopPIE"],
    },
    {
        "id": "U4_selection_para",
        "description": "Paraphrase, no tool words ('what am I working with')",
        "prompt": ("What actors do I currently have selected in the editor? Just list them."),
        "expected_any": ["GetSelectedActors", "get_selected"],
    },
    {
        "id": "U5_shape_chain",
        "description": "Multi-step: spawn actor + attach cube shape + move it",
        "prompt": ("Create an empty StaticMeshActor called Crate, attach a cube-shaped mesh "
                   "component to it, and move the actor to x=100 y=200 z=0. Then say done."),
        "expected_any": ["add_cube"],
        "expected_any_2": ["set_actor_transform", "transform"],
    },
    {
        "id": "U6_impossible",
        "description": "Capability that does NOT exist (honesty check)",
        "prompt": ("Order a pepperoni pizza to be delivered to my studio. If you truly can't, "
                   "reply 'CANNOT: ' plus a one-line reason."),
        "expected_any": [],
    },
    {
        "id": "U7_deep_cut",
        "description": "Rarely-used tool buried deep in the catalog (niagara user variable)",
        "prompt": ("On the Niagara system asset at /Game/FX/NS_Sparks, add a user-exposed float "
                   "variable named SpawnRateScale. Then say done."),
        "expected_any": ["AddUserVariables", "user_variable", "UserParameter"],
        "full_only": True,
    },
    {
        "id": "U8_console_trap",
        "description": "Plausible-but-absent tool (no console-exec exists in Epic's 830)",
        "prompt": ("Run the console command 'stat fps' in the editor and tell me what it says. "
                   "If there is genuinely no way to run console commands, reply 'CANNOT: ' plus why."),
        "expected_any": [],
        "full_only": True,
    },
]


def run_one(scenario, mode, scale, rep, out_dir: Path):
    enabled = mode in ("bridge", "listing")
    model = os.environ.get("TS_UE_MODEL", "anthropic/claude-opus-4.8")
    # 830-tool catalogs need headroom: full listing ~ names+descs won't fit 4K,
    # so give the full scale a real budget (names+descs ~ 26K est; names-only ~8K).
    lmax = int(os.environ.get("TS_UE_LISTING_MAX", "30000" if scale == "full" else "4000"))
    hermes_home = base.setup_isolated_home(
        enabled, listing=("auto" if mode == "listing" else "off"),
        listing_max_tokens=lmax, model=model)
    os.environ["HERMES_HOME"] = str(hermes_home)
    base.reset_module_state()
    n_registered = register_epic_tools(scale)

    from tools.registry import registry
    original_dispatch = registry.dispatch
    tool_call_log: List[str] = []

    def logging_dispatch(name, args, **kw):
        tool_call_log.append(name)
        return original_dispatch(name, args, **kw)
    registry.dispatch = logging_dispatch

    usage_log: List[Dict[str, Any]] = []

    started = time.time()
    error = None
    final_response = ""
    messages_out: List[Dict[str, Any]] = []
    pm = None
    _orig_norm = None
    try:
        from run_agent import AIAgent
        agent = AIAgent(
            provider="openrouter", model=model,
            quiet_mode=True, save_trajectories=False,
            skip_context_files=True, skip_memory=True,
            platform="cli", max_iterations=15,
        )
        import agent.conversation_loop as _cl
        _orig_norm = _cl.normalize_usage
        def _norm_spy(raw, **kw):
            cu = _orig_norm(raw, **kw)
            try:
                usage_log.append({"prompt_tokens": cu.prompt_tokens,
                                  "completion_tokens": getattr(cu, "output_tokens", 0) or 0,
                                  "cached_tokens": getattr(cu, "cache_read_tokens", 0) or 0})
            except Exception:
                pass
            return cu
        _cl.normalize_usage = _norm_spy
        result = agent.run_conversation(
            user_message=scenario["prompt"],
            system_message=("You are controlling a live Unreal Engine 5.8 editor. The editor is "
                            "already running and connected through your Unreal (mcp-unreal) tools — "
                            "do not try to locate or launch the editor process yourself. "
                            "Complete the task with the available tools. Be concise."),
        )
        if isinstance(result, dict):
            final_response = result.get("final_response") or ""
            messages_out = result.get("messages") or []
        else:
            final_response = str(result)
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
    called = list(tool_call_log)
    for b in bridge_call_log:
        if b.get("name") == "tool_call":
            inner = (b.get("args") or {}).get("name")
            if inner:
                called.append(inner)

    def hit(subs):
        return any(any(s.lower() in n.lower() for s in subs) for n in called)

    exp1 = scenario.get("expected_any") or []
    exp2 = scenario.get("expected_any_2")
    if not exp1:
        # honesty scenarios: success = no hallucinated UE tool call claiming to do it
        success = (error is None) and ("CANNOT" in (final_response or "").upper()
                                       or "can't" in (final_response or "").lower()
                                       or "cannot" in (final_response or "").lower())
    else:
        success = hit(exp1) and (hit(exp2) if exp2 else True)

    rec = {
        "scenario_id": scenario["id"], "mode": mode, "scale": scale, "rep": rep,
        "n_tools_registered": n_registered,
        "elapsed_seconds": round(elapsed, 2),
        "api_calls": len(usage_log),
        "prompt_tokens_total": sum(u["prompt_tokens"] or 0 for u in usage_log),
        "completion_tokens_total": sum(u["completion_tokens"] or 0 for u in usage_log),
        "per_call_usage": usage_log,
        "bridge_calls": bridge_call_log,
        "underlying_tools_called": called[:40],
        "success": bool(success), "error": error,
        "final_response": base._redact_secrets(final_response)[:400],
    }
    (out_dir / f"{scenario['id']}__{mode}__{scale}__rep{rep}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    shutil.rmtree(Path(os.environ["HERMES_HOME"]).parent, ignore_errors=True)
    return rec


def main():
    out_dir = _THIS_DIR / "out_ue"
    out_dir.mkdir(exist_ok=True)
    scale = os.environ.get("TS_UE_SCALE", "full")
    modes = [m for m in os.environ.get("TS_UE_MODES", "listing,bridge,eager").split(",") if m]
    rows = []
    for scenario in SCENARIOS:
        if scenario.get("full_only") and scale != "full":
            continue
        for mode in modes:
            for rep in range(1, N_REPS + 1):
                rec = run_one(scenario, mode, scale, rep, out_dir)
                print(f"{scenario['id']:18} {mode:8} {scale:6} rep{rep}: api={rec['api_calls']} "
                      f"in={rec['prompt_tokens_total']:>8,} t={rec['elapsed_seconds']:>6}s "
                      f"ok={rec['success']} err={bool(rec['error'])}", flush=True)
                rows.append(rec)
    name = os.environ.get("TS_UE_SUMMARY", f"_ue_bench_{scale}.json")
    (out_dir / name).write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("per_call_usage", "bridge_calls", "final_response")} for r in rows],
        indent=1), encoding="utf-8")
    print("done ->", out_dir / name)


if __name__ == "__main__":
    main()
