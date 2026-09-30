#!/usr/bin/env python3
"""Static tool-schema AND full-request prefix footprint of Hermes vs SamAgent postures (Phase 0 / T0.3).

No model calls, no network. Measures:
1. Tool-schema tokens across postures (core-59, coding-37, lean-8, lean-5, plus groups).
2. Full request prefix tokens (system prompt stable + context + tool schemas) constructed by
   the real ``run_agent.AIAgent`` and ``agent.system_prompt.build_system_prompt_parts`` in both
   a clean project directory and the repository root.

Run from the repo root with an isolated home so nothing touches a real ~/.hermes:

    HERMES_HOME=$(mktemp -d) python evals/samagent_bench/tool_footprint.py \
        --out docs/samagent/measurements/tool_footprint.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

import model_tools  # noqa: E402  (imports and registers every tool module)
import toolsets  # noqa: E402
from agent.system_prompt import build_system_prompt, build_system_prompt_parts  # noqa: E402
from run_agent import AIAgent  # noqa: E402
from tools.registry import registry  # noqa: E402


def _counter():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda s: len(enc.encode(s))), "tiktoken cl100k_base"
    except Exception:
        return (lambda s: len(s) // 4), "chars/4 proxy"


def _measure_full_prefix(tok) -> dict:
    """Measure full AIAgent system prompt + resolved tool schemas in a clean project vs repo root."""
    postures = {
        "hermes_cli_default": dict(enabled_toolsets=["hermes-cli"], skip_memory=False),
        "hermes_coding": dict(enabled_toolsets=["coding"], skip_memory=False),
        "samagent_lean_worker": dict(enabled_toolsets=["terminal", "file"], skip_memory=True),
    }
    out: dict = {}
    orig_cwd = os.getcwd()
    orig_env_cwd = os.environ.get("TERMINAL_CWD")
    try:
        with tempfile.TemporaryDirectory(prefix="samagent-clean-proj-") as clean_dir:
            for env_name, target_cwd in (("clean_project", clean_dir), ("hermes_repo_root", str(ROOT))):
                os.environ["TERMINAL_CWD"] = target_cwd
                os.chdir(target_cwd)
                env_res = {}
                for label, kw in postures.items():
                    ag = AIAgent(
                        api_key="offline-not-a-credential",
                        base_url="http://127.0.0.1:9/v1",
                        provider="openai-compat",
                        model="offline-probe",
                        quiet_mode=True,
                        skip_background_review=True,
                        save_trajectories=False,
                        platform="cli",
                        session_id=f"20260929_{env_name}_{label}",
                        **kw,
                    )
                    full_sys = build_system_prompt(ag)
                    parts = build_system_prompt_parts(ag)
                    tools_json = json.dumps(ag.tools, separators=(",", ":"))
                    sys_tok = tok(full_sys)
                    tool_tok = tok(tools_json)
                    env_res[label] = {
                        "resolved_tools": len(ag.tools),
                        "tool_names": [t["function"]["name"] for t in ag.tools],
                        "tool_schema_tokens": tool_tok,
                        "system_prompt_tokens": sys_tok,
                        "system_stable_tokens": tok(parts.get("stable", "")),
                        "system_context_tokens": tok(parts.get("context", "")),
                        "total_prefix_tokens": sys_tok + tool_tok,
                    }
                out[env_name] = env_res
    finally:
        os.chdir(orig_cwd)
        if orig_env_cwd is None:
            os.environ.pop("TERMINAL_CWD", None)
        else:
            os.environ["TERMINAL_CWD"] = orig_env_cwd
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write JSON results here")
    args = ap.parse_args()
    tok, method = _counter()

    def cost(name: str) -> int | None:
        schema = registry.get_schema(name)
        if not schema:
            return None
        return tok(json.dumps({"type": "function", "function": schema}, separators=(",", ":")))

    def tally(names):
        have = [n for n in names if cost(n) is not None]
        return {
            "tools": len(names),
            "registered": len(have),
            "schema_tokens": sum(cost(n) for n in have),
        }

    core = list(toolsets._HERMES_CORE_TOOLS)
    coding = list(toolsets._CODING_TOOLS)
    lean5 = ["read_file", "search_files", "patch", "write_file", "terminal"]
    lean8 = lean5 + ["todo_list", "delegate_task", "clarify"]

    result = {
        "method": method,
        "scope": "tool schemas + full AIAgent system-prompt prefix (no model API calls)",
        "postures": {
            "core_all": tally(core),
            "coding_posture": tally(coding),
            "lean_5": tally(lean5),
            "lean_8": tally(lean8),
        },
        "groups_in_core": {
            "browser_*": tally([n for n in core if n.startswith("browser_")]),
            "kanban_*": tally([n for n in core if n.startswith("kanban_")]),
        },
        "per_tool_top12": sorted(
            ((cost(n), n) for n in core if cost(n) is not None), reverse=True
        )[:12],
    }

    for label, kw in (("assembly_on", {}), ("assembly_off", {"skip_tool_search_assembly": True})):
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["coding"], quiet_mode=True, **kw
        )
        result.setdefault("resolved_here_coding_toolset", {})[label] = {
            "tools": len(defs),
            "schema_tokens": tok(json.dumps(defs, separators=(",", ":"))),
            "names": [d["function"]["name"] for d in defs],
        }

    result["full_request_prefix"] = _measure_full_prefix(tok)

    result["context_share"] = {
        f"{w // 1024}k_window": {
            k: round(v["schema_tokens"] / w, 3) for k, v in result["postures"].items()
        }
        for w in (32768, 65536, 131072)
    }

    text = json.dumps(result, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
