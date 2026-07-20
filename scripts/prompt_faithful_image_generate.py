#!/usr/bin/env python3
"""CLI wrapper for prompt-faithful image generation."""

from __future__ import annotations

import json
import sys

from tools.prompt_faithful_image_tool import generate_prompt_faithful_image


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        print("Usage: python scripts/prompt_faithful_image_generate.py <prompt>", file=sys.stderr)
        return 1

    try:
        result = generate_prompt_faithful_image(prompt)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(result["media_tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
