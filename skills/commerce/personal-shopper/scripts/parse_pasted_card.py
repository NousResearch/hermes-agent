#!/usr/bin/env python3
"""Parse a card the user pasted from their banking app's disposable
card feature.

Usage:
  python3 parse_pasted_card.py 'PAN MM/YY CVV' [--json]
  python3 parse_pasted_card.py - [--json]    # read text from stdin

Outputs (JSON mode):
  {
    "pan": "1234567812345678",
    "exp_month": 12,
    "exp_year": 27,
    "cvv": "123",
    "last4": "5678"
  }
  or
  {"error": "could not parse"}

Format accepted:
  - PAN with spaces or dashes (e.g. '4111 1111 1111 1111' or '4111-1111-1111-1111')
  - MM/YY or MM/YYYY expiry (slash-anchored, no dashes/dots between month and year)
  - CVV: 3 or 4 digits anywhere after the expiry

We accept slashes only (not dashes/dots) for the expiry separator
because PAN groups frequently use dashes -- otherwise the heuristic
would misfire on '1234-5678' inputs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys


def parse(text: str) -> dict[str, object] | None:
    exp_matches = list(re.finditer(r"(\b\d{1,2})\s*/\s*(\d{2,4}\b)", text))
    if not exp_matches:
        return None
    exp = exp_matches[-1]
    mm = int(exp.group(1))
    yy_raw = exp.group(2)
    yy = int(yy_raw) if len(yy_raw) == 2 else int(yy_raw[-2:])
    if not (1 <= mm <= 12):
        return None

    before = text[: exp.start()]
    after = text[exp.end():]

    pan_digits = re.sub(r"\D", "", before)
    if not (12 <= len(pan_digits) <= 19):
        return None

    cvv_match = re.search(r"(\d{3,4})", after)
    if not cvv_match:
        return None
    cvv = cvv_match.group(1)
    return {
        "pan": pan_digits,
        "exp_month": mm,
        "exp_year": yy,
        "cvv": cvv,
        "last4": pan_digits[-4:],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("text", help="The pasted text, or '-' to read stdin")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    text = sys.stdin.read() if args.text == "-" else args.text
    out = parse(text)
    if out is None:
        payload = {"error": "could not parse PAN+exp+CVV from input"}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return 2

    if args.json:
        sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(
            f"PAN  {out['pan']}\nExp  {out['exp_month']:02d}/{out['exp_year']:02d}\n"
            f"CVV  {out['cvv']}\nLast4 {out['last4']}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
