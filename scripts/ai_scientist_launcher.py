#!/usr/bin/env python3
"""Hermes-aware entrypoint for Sakana launch_scientist.py.

Patches ``ai_scientist.llm.create_client`` so OpenAI-compatible Hermes routes
(Codex OAuth, Nous free tier, NVIDIA, Groq, xAI OAuth, etc.) work without
bare ``OPENAI_API_KEY`` / Docker. Child env is prepared by ``tools.ai_scientist_env``.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AI_SCIENTIST_DIR = REPO_ROOT / "vendor" / "openclaw-mirror" / "AI-Scientist"
LAUNCH_SCRIPT = AI_SCIENTIST_DIR / "launch_scientist.py"


def _patch_create_client() -> None:
    import openai

    import ai_scientist.llm as llm_mod

    original = llm_mod.create_client

    def _bridged_create_client(model: str):
        bridge = os.environ.get("AI_SCIENTIST_HERMES_BRIDGE", "").strip() == "1"
        base = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or ""
        ).strip().rstrip("/")
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        api_model = (os.environ.get("AI_SCIENTIST_API_MODEL") or model).strip()
        force_shim = os.environ.get("AI_SCIENTIST_FORCE_OPENAI_SHIM", "").strip() == "1"
        gpt_like = "gpt" in model or "o1" in model or "o3" in model

        if bridge and base and api_key and (gpt_like or force_shim):
            print(
                f"Using Hermes-bridged OpenAI-compatible API "
                f"(sakana_model={model}, api_model={api_model})."
            )
            return openai.OpenAI(api_key=api_key, base_url=base), api_model
        return original(model)

    llm_mod.create_client = _bridged_create_client


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not LAUNCH_SCRIPT.is_file():
        print(f"AI-Scientist entrypoint missing: {LAUNCH_SCRIPT}", file=sys.stderr)
        return 2

    if AI_SCIENTIST_DIR.is_dir() and str(AI_SCIENTIST_DIR) not in sys.path:
        sys.path.insert(0, str(AI_SCIENTIST_DIR))

    from tools.ai_scientist_env import apply_ai_scientist_run_config
    from tools.ai_scientist_deps import ensure_ai_scientist_deps

    model = None
    for idx, token in enumerate(args):
        if token == "--model" and idx + 1 < len(args):
            model = args[idx + 1]
            break

    ensure_ai_scientist_deps(prompt=False)
    apply_ai_scientist_run_config(model=model)
    _patch_create_client()

    sys.argv = [str(LAUNCH_SCRIPT), *args]
    runpy.run_path(str(LAUNCH_SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
