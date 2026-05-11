"""
Borge Agent CLI — standalone interactive shell.

Usage:
    borge                          # interactive REPL
    borge "fix the auth bug"       # single turn
    borge --model claude-opus-4-7  # choose model
    borge --soul ./SOUL.md         # custom soul file
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="borge",
        description="Borge Agent — cognitively-grounded AI assistant",
    )
    parser.add_argument("prompt", nargs="?", help="Single prompt (non-interactive)")
    parser.add_argument("--model", "-m", default=os.environ.get("BORGE_MODEL", "claude-opus-4-7"))
    parser.add_argument("--soul", "-s", default=None, help="Path to SOUL.md")
    parser.add_argument("--db", default=None, help="SQLite DB path for memory")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    try:
        from borge.runner import BorgeRunner
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    runner = BorgeRunner(
        model=args.model,
        soul_path=args.soul,
        db_path=args.db,
    )

    # Single-turn mode
    if args.prompt:
        if args.no_stream:
            print(runner.run(args.prompt))
        else:
            for chunk in runner.stream(args.prompt):
                print(chunk, end="", flush=True)
            print()
        return

    # Interactive REPL
    print(f"Borge Agent  (model: {args.model})  — type /exit or Ctrl+D to quit\n")
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            runner.reset()
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            runner.reset()
            break
        if user_input == "/reset":
            runner.reset()
            print("[Session reset]\n")
            continue

        print("Borge> ", end="", flush=True)
        if args.no_stream:
            print(runner.run(user_input))
        else:
            for chunk in runner.stream(user_input):
                print(chunk, end="", flush=True)
            print()
