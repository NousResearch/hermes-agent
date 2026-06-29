#!/usr/bin/env python3
# vendored from bmad-workflow-builder/scripts; canonical source there
# /// script
# requires-python = ">=3.9"
# dependencies = ["tiktoken"]
# ///
"""count_tokens — the single length metric for skill authoring.

Token counts replace line counts everywhere in the builder and eval-runner.
This script reports the token length of a file or of text piped on stdin, using
the tiktoken cl100k_base encoding. When tiktoken is not installed it falls back
to a character-based estimate (len(text) // 4) and says so, so the script always
runs under a bare python3 even with no third-party packages present.

Usage:
  count_tokens.py <file>     count the tokens in a file
  count_tokens.py --stdin    count the tokens read from stdin

Output (one line of JSON on stdout):
  {"tokens": <int>, "method": "tiktoken"}   when tiktoken loaded
  {"tokens": <int>, "method": "fallback"}   when it fell back to chars // 4

Budgets this feeds: SKILL.md ~1500-2500, multi-branch reference ~4500,
single-purpose reference ~9000.
"""
import argparse
import json
import sys

ENCODING = "cl100k_base"


def count_tokens(text: str) -> tuple[int, str]:
    """Return (token_count, method).

    Tries tiktoken's cl100k_base encoding first. If tiktoken cannot be imported
    or initialized, estimates with len(text) // 4 and reports method "fallback".
    """
    try:
        import tiktoken
    except Exception:
        return len(text) // 4, "fallback"
    try:
        enc = tiktoken.get_encoding(ENCODING)
    except Exception:
        return len(text) // 4, "fallback"
    return len(enc.encode(text)), "tiktoken"


def read_input(args) -> str:
    if args.stdin:
        return sys.stdin.read()
    with open(args.file, encoding="utf-8") as f:
        return f.read()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("file", nargs="?", help="path to the file to count")
    p.add_argument("--stdin", action="store_true", help="read text from stdin instead of a file")
    args = p.parse_args(argv)

    if not args.stdin and not args.file:
        p.error("provide a file path or --stdin")
    if args.stdin and args.file:
        p.error("provide either a file path or --stdin, not both")

    text = read_input(args)
    tokens, method = count_tokens(text)
    print(json.dumps({"tokens": tokens, "method": method}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
