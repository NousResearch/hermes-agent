#!/usr/bin/env python3
"""Reviewed helper: moss + model block inspection."""
import json
import yaml
from pathlib import Path

ROOT = Path.home() / ".hermes"


def main():
    out = {}
    for n in ("moss", "wesker", "kensei-review", "octacon-architect", "octacon-backend"):
        doc = yaml.safe_load((ROOT / "profiles" / n / "config.yaml").read_text()) or {}
        out[n] = {"model": doc.get("model"), "tier": doc.get("tier"), "base_url": doc.get("base_url"), "provider": doc.get("provider")}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()