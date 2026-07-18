#!/usr/bin/env python3
"""Tool Search live benchmark v2 — real token accounting + more scenarios + reps.

Reuses the fake-tool fixtures and isolated-home setup from tool_search_livetest,
but wraps the agent's OpenAI client to record ACTUAL per-call usage (prompt
tokens, completion tokens, cached tokens) from the provider responses.

Runs each scenario N_REPS times in each mode (on/off). Output:
  scripts/out2/<scenario>__<mode>__rep<k>.json
  scripts/out2/_bench_summary.json
"""
from __future__ import annotations

import json, os, shutil, sys, tempfile, time, traceback
from pathlib import Path
from typing import Any, Dict, List

_THIS_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_WORKTREE_ROOT))
sys.path.insert(0, str(_THIS_DIR))

import tool_search_livetest as base  # fixtures + helpers

N_REPS = int(os.environ.get("TS_BENCH_REPS", "3"))

SCENARIOS: List[Dict[str, Any]] = base.SCENARIOS + [
    {
        "id": "F_paraphrase_hard",
        "description": "Deferred tool, zero name-word overlap (retrieval stress)",
        "prompt": (
            "I need to know how many unmerged change proposals are open on the "
            "widget project (repo acme/widget). Just the count. Then you're done."
        ),
        "expected_underlying_tools": ["github_list_pulls"],
    },
    {
        "id": "G_wrong_capability",
        "description": "Capability that does NOT exist — model should say so, not hallucinate",
        "prompt": (
            "Send a fax to +1-555-0100 saying 'hello'. If you truly can't, say "
            "'CANNOT: ' plus a one-line reason."
        ),
        "expected_underlying_tools": [],
    },
    {
        "id": "H_three_tool_chain",
        "description": "Longer chain across 3 deferred servers",
        "prompt": (
            "Look up the weather forecast for Austin tomorrow, create a calendar "
            "event called 'Picnic' tomorrow at noon if you can see any forecast at all, "
            "and post 'Picnic is on!' to the #random Slack channel. Then say done."
        ),
        "expected_underlying_tools": ["weather_get", "evt_create", "slack_send_message"],
    },
]


def run_one(scenario: Dict[str, Any], mode: str, rep: int, out_dir: Path) -> Dict[str, Any]:
    """mode: 'enabled' (bare bridge) | 'listing' (bridge + catalog listing) | 'disabled' (eager)."""
    enabled = mode in ("enabled", "listing")
    hermes_home = base.setup_isolated_home(enabled, listing=("auto" if mode == "listing" else "off"))
    os.environ["HERMES_HOME"] = str(hermes_home)
    base.reset_module_state()
    n_registered = base.register_fake_tools()

    Path("/tmp/livetest").mkdir(exist_ok=True)
    (Path("/tmp/livetest/notes.txt")).write_text("Hello from the test fixture.\n")

    from tools.registry import registry
    original_dispatch = registry.dispatch

    tool_call_log: List[Dict[str, Any]] = []
    def logging_dispatch(name, args, **kw):
        tool_call_log.append({"name": name})
        return original_dispatch(name, args, **kw)
    registry.dispatch = logging_dispatch

    # Capture REAL per-call usage via the post_api_request plugin hook —
    # it fires on both streaming and non-streaming paths with normalized
    # usage. NOTE: registered AFTER AIAgent construction because plugin
    # discovery during init calls _hooks.clear().
    usage_log: List[Dict[str, Any]] = []
    def usage_hook(**kw):
        u = kw.get("usage") or {}
        if u:
            usage_log.append({
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "cached_tokens": u.get("cached_tokens") or u.get("cache_read_input_tokens") or 0,
            })

    started = time.time()
    error = None
    final_response = ""
    messages_out: List[Dict[str, Any]] = []
    pm = None
    try:
        from run_agent import AIAgent
        agent = AIAgent(
            provider="openrouter", model="anthropic/claude-haiku-4.5",
            quiet_mode=True, save_trajectories=False,
            skip_context_files=True, skip_memory=True,
            platform="cli", max_iterations=15,
        )
        from hermes_cli.plugins import get_plugin_manager, discover_plugins
        discover_plugins()  # idempotent; ensures no later clear wipes our hook
        pm = get_plugin_manager()
        pm._hooks.setdefault("post_api_request", []).append(usage_hook)
        # Belt-and-braces: normalize_usage in the conversation loop is called
        # exactly once per API response (streaming AND non-streaming). Wrap it
        # to capture canonical usage the hook path may miss.
        import agent.conversation_loop as _cl
        _orig_norm = _cl.normalize_usage
        def _norm_spy(raw, **kw):
            cu = _orig_norm(raw, **kw)
            try:
                usage_log.append({
                    "prompt_tokens": cu.prompt_tokens,
                    "completion_tokens": getattr(cu, "output_tokens", 0) or 0,
                    "cached_tokens": getattr(cu, "cache_read_tokens", 0) or 0,
                    "src": "norm",
                })
            except Exception:
                pass
            return cu
        _cl.normalize_usage = _norm_spy
        result = agent.run_conversation(
            user_message=scenario["prompt"],
            system_message=("You are a test agent. Complete the user's task using available "
                            "tools. Be concise; don't add commentary beyond what's needed."),
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
        try:
            import agent.conversation_loop as _cl2
            if "_orig_norm" in dir() or True:
                try:
                    _cl2.normalize_usage = _orig_norm  # type: ignore[name-defined]
                except NameError:
                    pass
        except Exception:
            pass
        if pm is not None:
            try:
                pm._hooks.get("post_api_request", []).remove(usage_hook)
            except ValueError:
                pass

    # Prefer the normalize_usage spy entries (one per API response, streaming
    # included); fall back to hook entries when the spy saw nothing.
    norm_entries = [u for u in usage_log if u.get("src") == "norm"]
    if norm_entries:
        usage_log = norm_entries

    elapsed = time.time() - started
    bridge_call_log = base._extract_bridge_calls(messages_out)

    expected = scenario.get("expected_underlying_tools", [])
    called_names = [c.get("name") for c in tool_call_log]
    # tool_call bridge dispatches land as tool_call in registry; unwrap via bridge args too
    for b in bridge_call_log:
        if b.get("name") == "tool_call":
            inner = (b.get("args") or {}).get("name")
            if inner:
                called_names.append(inner)
    success = all(e in called_names for e in expected) if expected else (error is None)

    rec = {
        "scenario_id": scenario["id"], "mode": mode,
        "rep": rep, "elapsed_seconds": round(elapsed, 2),
        "api_calls": len(usage_log),
        "prompt_tokens_total": sum(u.get("prompt_tokens") or 0 for u in usage_log),
        "completion_tokens_total": sum(u.get("completion_tokens") or 0 for u in usage_log),
        "cached_tokens_total": sum(u.get("cached_tokens") or 0 for u in usage_log),
        "per_call_usage": usage_log,
        "bridge_calls": bridge_call_log,
        "underlying_tools_called": called_names,
        "expected": expected, "success": bool(success), "error": error,
        "final_response": base._redact_secrets(final_response)[:500],
    }
    out_path = out_dir / f"{scenario['id']}__{'enabled' if enabled else 'disabled'}__rep{rep}.json"
    out_path.write_text(json.dumps(rec, indent=1))
    shutil.rmtree(Path(os.environ["HERMES_HOME"]).parent, ignore_errors=True)
    return rec


def main():
    out_dir = _THIS_DIR / "out2"
    out_dir.mkdir(exist_ok=True)
    modes = [m for m in os.environ.get("TS_BENCH_MODES", "enabled,listing,disabled").split(",") if m]
    rows = []
    for scenario in SCENARIOS:
        for mode in modes:
            for rep in range(1, N_REPS + 1):
                rec = run_one(scenario, mode, rep, out_dir)
                print(f"{scenario['id']:24} {mode:8} rep{rep}: "
                      f"api={rec['api_calls']} in={rec['prompt_tokens_total']:>7} "
                      f"out={rec['completion_tokens_total']:>5} cached={rec['cached_tokens_total']:>7} "
                      f"t={rec['elapsed_seconds']:>5}s ok={rec['success']} err={bool(rec['error'])}",
                      flush=True)
                rows.append(rec)
    summary_name = os.environ.get("TS_BENCH_SUMMARY", "_bench_summary.json")
    (out_dir / summary_name).write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("per_call_usage", "bridge_calls", "final_response")} for r in rows],
        indent=1))
    print("done ->", out_dir / summary_name)


if __name__ == "__main__":
    main()
