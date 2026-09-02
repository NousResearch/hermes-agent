#!/usr/bin/env python3
"""Reviewed helper: dump per-profile auth.json pool KEYS + providers dict shape."""
import json
from pathlib import Path
import yaml

ROOT = Path.home() / ".hermes"


def main():
    out = {}
    for pd in sorted((ROOT / "profiles").iterdir()):
        if not pd.is_dir():
            continue
        auth = pd / "auth.json"
        pool_keys = None
        cp_pool = None
        if auth.exists():
            try:
                d = json.loads(auth.read_text())
                pool = d.get("credential_pool") or {}
                pool_keys = sorted(pool.keys())
                cp_pool = {k: len(v) if isinstance(v, list) else v for k, v in pool.items() if k.startswith("custom:")}
            except Exception as e:
                pool_keys = f"error: {e}"
        cfg = pd / "config.yaml"
        providers_keys = None
        if cfg.exists():
            doc = yaml.safe_load(cfg.read_text()) or {}
            pv = doc.get("providers")
            providers_keys = sorted(pv.keys()) if isinstance(pv, dict) else None
            cpv = doc.get("custom_providers")
            cp_names = [e.get("name") for e in cpv if isinstance(e, dict)] if isinstance(cpv, list) else None
        out[pd.name] = {
            "auth_pool_keys": pool_keys,
            "auth_custom_pool_sizes": cp_pool,
            "config_providers_keys": providers_keys,
            "config_custom_provider_names": cp_names,
        }
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()