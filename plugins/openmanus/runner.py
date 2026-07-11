from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path


def _literal_path(path: Path) -> str:
    return repr(str(path))


def _copy_source(source_root: Path, checkout: Path, workspace_root: Path) -> None:
    shutil.copytree(
        source_root,
        checkout,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "tests", ".github", "assets", "workspace"),
    )
    config_path = checkout / "app" / "config.py"
    text = config_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"def get_project_root\(\) -> Path:\n.*?PROJECT_ROOT = get_project_root\(\)\nWORKSPACE_ROOT = PROJECT_ROOT / \"workspace\"",
        re.DOTALL,
    )
    replacement = (
        "def get_project_root() -> Path:\n"
        f"    return Path({_literal_path(checkout)})\n\n\n"
        "PROJECT_ROOT = get_project_root()\n"
        f"WORKSPACE_ROOT = Path({_literal_path(workspace_root)})"
    )
    patched, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("unsupported OpenManus config.py layout; refusing to run")
    config_path.write_text(patched, encoding="utf-8")


def _write_config(checkout: Path, args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env, "")
    if not args.model or not args.base_url or not api_key:
        raise RuntimeError("OpenManus LLM configuration is incomplete")
    config_dir = checkout / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": api_key,
        "max_tokens": 8192,
        "temperature": 0.0,
        "api_type": args.api_type,
        "api_version": "",
    }
    lines = ["[llm]"]
    for key, value in values.items():
        if isinstance(value, int):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f"{key} = {json.dumps(str(value))}")
    lines.extend(
        [
            "",
            "[browser]",
            "headless = true",
            "disable_security = false",
            "max_content_length = 4000",
            "",
            "[mcp]",
            'server_reference = "app.mcp.server"',
            "",
            "[runflow]",
            "use_data_analysis_agent = false",
        ]
    )
    (config_dir / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _run(args: argparse.Namespace) -> str:
    source_root = Path(args.source_root).resolve()
    run_root = Path(args.run_root).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    checkout = run_root / "openmanus-checkout"
    run_root.mkdir(parents=True, exist_ok=True)
    _copy_source(source_root, checkout, workspace_root)
    config_path = checkout / "config" / "config.toml"
    agent = None
    try:
        _write_config(checkout, args)
        sys.path.insert(0, str(checkout))
        os.chdir(checkout)
        if args.agent_mode == "data_analysis":
            from app.agent.data_analysis import DataAnalysis

            agent = DataAnalysis()
        else:
            from app.agent.manus import Manus

            agent = await Manus.create()
        agent.max_steps = max(1, int(args.max_steps))
        if not args.allow_network:
            from app.tool import ToolCollection

            blocked = ("browser", "mcp", "web")
            agent.available_tools = ToolCollection(
                *(tool for tool in agent.available_tools.tools if not any(word in tool.name.lower() for word in blocked))
            )
        result = await agent.run(args.prompt)
        return str(result or "")
    finally:
        if agent is not None:
            cleanup = getattr(agent, "cleanup", None)
            if cleanup is not None:
                await cleanup()
        config_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-type", default="openai")
    parser.add_argument("--api-key-env", default="OPENMANUS_API_KEY")
    parser.add_argument("--agent-mode", choices=["manus", "data_analysis"], default="manus")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--no-network", dest="allow_network", action="store_false")
    parser.set_defaults(allow_network=False)
    args = parser.parse_args()
    args.prompt = sys.stdin.read()
    try:
        result = asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"OpenManus runner failed: {exc}", file=sys.stderr)
        return 1
    print("__HERMES_OPENMANUS_RESULT__:" + json.dumps({"result": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
