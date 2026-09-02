#!/usr/bin/env python3
"""Reviewed helper: fleet config inspection utilities (read-only).

Paths are constructed programmatically; never printed with secrets.
"""
import json
import yaml
from pathlib import Path

ROOT = Path.home() / ".hermes"
PROFILES = ROOT / "profiles"


def load_profile(name):
    cfg = PROFILES / name / "config.yaml"
    return yaml.safe_load(cfg.read_text()) or {}


def load_root():
    return yaml.safe_load((ROOT / "config.yaml").read_text()) or {}


def profile_names():
    return sorted(p.name for p in PROFILES.iterdir() if p.is_dir())


def summarize(doc):
    model = doc.get("model") or {}
    fb = doc.get("fallback_providers") or []
    return {
        "model_default": model.get("default") if isinstance(model, dict) else None,
        "model_provider": model.get("provider") if isinstance(model, dict) else None,
        "fallbacks": [
            {k: e.get(k) for k in ("provider", "model", "base_url", "route_slot", "credential_pool")}
            for e in fb if isinstance(e, dict)
        ],
        "keys": sorted(doc.keys()),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            print(f"== {name} ==")
            print(json.dumps(summarize(load_profile(name)), indent=1))
    else:
        out = {"root": summarize(load_root())}
        for n in profile_names():
            out[n] = summarize(load_profile(n))
        print(json.dumps(out, indent=1))