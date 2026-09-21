#!/usr/bin/env python3
"""Reviewed helper: dump custom-provider wiring across all profile configs."""
import json
import yaml
from pathlib import Path

ROOT = Path.home() / ".hermes"


def entry_summary(entry):
    if not isinstance(entry, dict):
        return entry
    out = {}
    for k in ("name", "provider_key", "base_url", "default_model", "model",
              "request_timeout_seconds", "extra_headers"):
        if k in entry:
            out[k] = entry[k]
    if "api_key" in entry:
        v = entry.get("api_key")
        out["api_key"] = ("<set:%d chars>" % len(v)) if isinstance(v, str) and v.strip() else v
    models = entry.get("models")
    if isinstance(models, dict):
        out["models_count"] = len(models)
    elif isinstance(models, list):
        out["models_count"] = len(models)
    return out


def main():
    names = sorted(p.name for p in (ROOT / "profiles").iterdir() if p.is_dir())
    result = {}
    for n in names:
        cfg = ROOT / "profiles" / n / "config.yaml"
        if not cfg.exists():
            continue
        doc = yaml.safe_load(cfg.read_text()) or {}
        cp = doc.get("custom_providers")
        pv = doc.get("providers")
        cps = doc.get("credential_pool_strategies")
        result[n] = {
            "custom_providers": [entry_summary(e) for e in cp] if isinstance(cp, list) else cp,
            "providers_keys": sorted(pv.keys()) if isinstance(pv, dict) else None,
            "providers_summary": {k: entry_summary(v) for k, v in pv.items()} if isinstance(pv, dict) else None,
            "credential_pool_strategies": cps,
        }
    # root
    doc = yaml.safe_load((ROOT / "config.yaml").read_text()) or {}
    result["__root__"] = {
        "custom_providers": [entry_summary(e) for e in doc.get("custom_providers")] if isinstance(doc.get("custom_providers"), list) else doc.get("custom_providers"),
        "providers_keys": sorted((doc.get("providers") or {}).keys()),
        "credential_pool_strategies": doc.get("credential_pool_strategies"),
    }
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()