#!/usr/bin/env python3
"""Validate MVP candidate JSON for the digital product advisor skill."""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

REQUIRED_CANDIDATE_FIELDS = ["model", "brand", "price", "currency", "scores"]


def validate(data):
    errors = []
    if not isinstance(data.get("weights"), dict) or not data["weights"]:
        errors.append("weights must be a non-empty object")
    if not isinstance(data.get("candidates"), list) or not data["candidates"]:
        errors.append("candidates must be a non-empty list")
        return errors
    for index, candidate in enumerate(data["candidates"]):
        for field in REQUIRED_CANDIDATE_FIELDS:
            if field not in candidate:
                errors.append(f"candidates[{index}] missing required field: {field}")
        scores = candidate.get("scores", {})
        if not isinstance(scores, dict):
            errors.append(f"candidates[{index}].scores must be an object")
            continue
        for key, value in scores.items():
            if value is not None and not (0 <= float(value) <= 10):
                errors.append(f"candidates[{index}].scores.{key} must be between 0 and 10 or null")
        links = candidate.get("links", {})
        if links is not None and not isinstance(links, dict):
            errors.append(f"candidates[{index}].links must be an object when present")
        elif isinstance(links, dict):
            for key, value in links.items():
                if value is None:
                    continue
                if not isinstance(value, str):
                    errors.append(f"candidates[{index}].links.{key} must be a URL string or null")
                    continue
                parsed = urlparse(value)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    errors.append(f"candidates[{index}].links.{key} must be an http(s) URL")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to candidate JSON")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    errors = validate(data)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, ensure_ascii=False, indent=2))
        sys.exit(1)
    print(json.dumps({"valid": True, "errors": []}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
