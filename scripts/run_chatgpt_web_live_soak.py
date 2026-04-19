#!/usr/bin/env python3
"""Run live ChatGPT Web end-to-end soak cases against the local Hermes CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class SoakCase:
    name: str
    prompt: str
    output_pattern: str
    required_tools: tuple[str, ...] = ()
    forbid_tools: tuple[str, ...] = ()


def _load_session(session_path: Path) -> dict[str, Any]:
    return json.loads(session_path.read_text(encoding="utf-8"))


def _tool_names(messages: Iterable[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            name = str(function.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def _latest_session_file(sessions_dir: Path, before: set[Path]) -> Path:
    deadline = time.time() + 15.0
    while time.time() < deadline:
        current = {path for path in sessions_dir.glob("session_*.json") if path.is_file()}
        created = sorted(current - before, key=lambda path: path.stat().st_mtime, reverse=True)
        if created:
            return created[0]
        if current:
            newest = max(current, key=lambda path: path.stat().st_mtime)
            if newest not in before:
                return newest
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for a new session file in {sessions_dir}")


def _run_case(
    *,
    repo_root: Path,
    env: dict[str, str],
    sessions_dir: Path,
    case: SoakCase,
    model: str,
) -> dict[str, Any]:
    before = {path for path in sessions_dir.glob("session_*.json") if path.is_file()}
    command = [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "chat",
        "--provider",
        "chatgpt-web",
        "-m",
        model,
        "--quiet",
        "-q",
        case.prompt,
    ]
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    session_path = _latest_session_file(sessions_dir, before)
    session = _load_session(session_path)
    messages = session.get("messages") if isinstance(session.get("messages"), list) else []
    tools = _tool_names(messages)

    if completed.returncode != 0:
        raise RuntimeError(f"{case.name}: hermes chat failed with code {completed.returncode}: {stderr or stdout}")
    if not re.search(case.output_pattern, stdout, re.IGNORECASE | re.DOTALL):
        raise RuntimeError(f"{case.name}: output did not match /{case.output_pattern}/: {stdout!r}")
    missing = [tool for tool in case.required_tools if tool not in tools]
    if missing:
        raise RuntimeError(f"{case.name}: missing required tool calls {missing}; saw {tools}")
    forbidden = [tool for tool in case.forbid_tools if tool in tools]
    if forbidden:
        raise RuntimeError(f"{case.name}: saw forbidden tool calls {forbidden}; saw {tools}")

    return {
        "name": case.name,
        "stdout": stdout,
        "stderr": stderr,
        "session": str(session_path),
        "tools": tools,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", default="gpt-5-4-thinking")
    parser.add_argument("--debug-base", default=os.getenv("CHATGPT_WEB_DEBUG_BASE", "http://127.0.0.1:9225"))
    parser.add_argument("--hermes-home", type=Path, default=None)
    parser.add_argument("--pythonpath", default="")
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    hermes_home = args.hermes_home or (Path(tempfile.gettempdir()) / f"hermes-live-soak-{int(time.time())}")
    hermes_home.mkdir(parents=True, exist_ok=True)
    sessions_dir = hermes_home / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["CHATGPT_WEB_DEBUG_BASE"] = args.debug_base
    pythonpath_parts = [str(args.repo_root)]
    if args.pythonpath:
        pythonpath_parts.append(args.pythonpath)
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)

    cases = [
        SoakCase(
            name="hello",
            prompt="Hello. Reply only with READY.",
            output_pattern=r"\bready\b",
            forbid_tools=("terminal", "search_files", "read_file"),
        ),
        SoakCase(
            name="whoami-natural",
            prompt="Try terminal tool and check whoami on it. Answer only the result.",
            output_pattern=r"\S+",
            required_tools=("terminal",),
        ),
        SoakCase(
            name="multi-terminal",
            prompt="Use the terminal tool to run whoami, then use the terminal tool to run pwd. Answer with two lines: username first, path second.",
            output_pattern=r".+\n.+",
            required_tools=("terminal",),
        ),
        SoakCase(
            name="file-route-natural",
            prompt="Can you check where stream_chatgpt_web_completion is defined in hermes_cli/chatgpt_web.py and answer only with the exact def line?",
            output_pattern=r"^def\s+stream_chatgpt_web_completion\(",
            required_tools=("search_files",),
        ),
    ]
    if args.image:
        cases.append(
            SoakCase(
                name="image-natural",
                prompt=f"Look at this local image: {args.image}. Answer only the dominant color and shape.",
                output_pattern=r"red\s+square",
                forbid_tools=("vision_analyze",),
            )
        )

    results: list[dict[str, Any]] = []
    for case in cases:
        results.append(
            _run_case(
                repo_root=args.repo_root,
                env=env,
                sessions_dir=sessions_dir,
                case=case,
                model=args.model,
            )
        )

    print(json.dumps({"ok": True, "hermes_home": str(hermes_home), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
