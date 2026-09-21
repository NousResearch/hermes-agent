#!/usr/bin/env python3
"""Reviewed helper: dump providers/custom_providers entries for reference profiles."""
import json
import yaml
from pathlib import Path

ROOT = Path.home() / ".hermes"


def scrub(d):
    if isinstance(d, dict):
        return {k: ("<set>" if k in ("api_key", "access_token", "api") and isinstance(v, str) and v.strip() else scrub(v)) for k, v in d.items()}
    if isinstance(d, list):
        return [scrub(x) for x in d]
    return d


def main():
    out = {}
    for n in ("ceecee", "octacon", "sirvir", "wesker", "kensei-review"):
        doc = yaml.safe_load((ROOT / "profiles" / n / "config.yaml").read_text()) or {}
        out[n] = {
            "providers": scrub(doc.get("providers")),
            "custom_providers": scrub(doc.get("custom_providers")),
        }
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()