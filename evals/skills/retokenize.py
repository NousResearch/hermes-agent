#!/usr/bin/env python3
"""Recount the blocks prompt_growth.py rendered with a real tokenizer (o200k_base).

Needs only tiktoken, so it runs outside the Hermes environment without changing it:

  uv run --no-project --with "tiktoken>=0.9,<1" python evals/skills/retokenize.py <dir written by prompt_growth.py>

Prints one row per rendered block and writes retokenized.json next to them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(__doc__)
        return 2
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    out_dir = Path(sys.argv[1])
    rows = {}
    for path in sorted(out_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        rows[path.name] = {"chars": len(text), "tokens_o200k_base": len(enc.encode(text, disallowed_special=()))}
        print(f"{path.name}: {rows[path.name]}")
    (out_dir / "retokenized.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
