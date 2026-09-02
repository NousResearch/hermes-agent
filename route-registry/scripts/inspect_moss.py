#!/usr/bin/env python3
"""Reviewed helper: moss agent.model + extract_main shapes."""
import json
import yaml
from pathlib import Path

ROOT = Path.home() / ".hermes"

for n in ("moss",):
    doc = yaml.safe_load((ROOT / "profiles" / n / "config.yaml").read_text()) or {}
    ag = doc.get("agent")
    print(n, "agent block:", json.dumps(ag, indent=1, default=str)[:400] if isinstance(ag, dict) else ag)
    print(n, "model:", doc.get("model"))