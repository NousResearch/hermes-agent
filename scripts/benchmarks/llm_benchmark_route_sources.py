#!/usr/bin/env python3
"""Benchmark provider sources for llm_benchmark_measurements.py.

Builds the non-secret measurement config from two source families:

- Legacy provider route config (existing benchmark config JSON): passed
  through unchanged so pre-cutover benchmarks keep working until the flip
  explicitly retires them.

- Turbofit gateway (default http://127.0.0.1:8091): recognised via its
  OpenAI-compatible /v1/models catalog.  The stable route IDs ``auto``,
  ``active:main`` and ``active:aux`` are recognised by name (never by
  resolving a concrete model tag), and the ``context_length`` published in
  the catalog is carried through so reports record the served window, not a
  hardcoded guess.  Turbofit entries are only included when explicitly
  enabled (--enable-turbofit / TURBOFIT_BENCH_ENABLE=1): enabling them is a
  cutover decision, not a side effect of running this tool.

Credential handling: entries carry only a ``key_env`` name.  The Turbofit
gateway ignores Authorization headers, but llm_benchmark_measurements skips
providers whose key_env is empty, so set TURBOFIT_BENCH_KEY=local (any
non-empty value) in the measurement job environment.  No credential material
is ever read or serialised here.

Output is a measurement config plus provenance, suitable for
llm_benchmark_measurements.py --config and for audit trails.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TURBOFIT_URL = "http://127.0.0.1:8091"
TURBOFIT_KEY_ENV = "TURBOFIT_BENCH_KEY"

# Stable route IDs published by turbofit-gateway/2.0 provider_models().
TURBOFIT_STABLE_IDS = ("auto", "active:main", "active:aux")

SOURCE_SCHEMA = "kensei.llm-benchmark-sources/v1"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_model_catalog(base_url, timeout=5.0):
    """GET {base}/v1/models and return the parsed catalog, or raise."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/models",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("unrecognised model catalog payload")
    return payload


def turbofit_sources(catalog, base_url):
    """Project the stable route IDs out of a /v1/models catalog.

    Unknown concrete model IDs are ignored: benchmarks address the stable
    routes so a route flip never changes the benchmark's view.
    """
    entries = []
    for model in catalog.get("data", []):
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "")
        if model_id not in TURBOFIT_STABLE_IDS:
            continue
        context_length = model.get("context_length")
        entry = {
            "name": f"turbofit-{model_id.replace(':', '-')}",
            "model": model_id,
            "endpoint": base_url.rstrip("/") + "/v1",
            "key_env": TURBOFIT_KEY_ENV,
            "enabled": True,
            "backend": "turbofit-gateway",
        }
        if isinstance(context_length, int) and context_length > 0:
            entry["context_length"] = context_length
        entries.append(entry)
    return entries


def legacy_sources(config):
    """Return legacy provider entries projected to the measurement contract.

    Allow-list projection (the fields llm_benchmark_measurements consumes):
    unknown keys — including any credential material accidentally present in
    a hand-edited config — cannot pass through into the output.
    """
    providers = config.get("providers") if isinstance(config, dict) else None
    if not isinstance(providers, list):
        return []
    allowed = ("name", "model", "endpoint", "key_env", "context_length")
    projected = []
    for item in providers:
        if not isinstance(item, dict):
            continue
        entry = {key: item[key] for key in allowed if key in item}
        if entry:
            projected.append(entry)
    return projected


def build_sources(legacy_config, *, turbofit_url=None, enable_turbofit=False,
                  fetch_catalog=None, now=None):
    """Merge legacy + (optional) Turbofit entries into one measurement config."""
    now = now or _utc_iso()
    fetch_catalog = fetch_catalog or (
        lambda url: fetch_model_catalog(url)
    )
    turbofit_status = "disabled"
    turbofit_entries: list[dict] = []
    if enable_turbofit:
        turbofit_url = turbofit_url or DEFAULT_TURBOFIT_URL
        try:
            catalog = fetch_catalog(turbofit_url)
            turbofit_entries = turbofit_sources(catalog, turbofit_url)
            turbofit_status = "ok" if turbofit_entries else "no_stable_routes"
        except (OSError, ValueError, urllib.error.URLError) as error:
            turbofit_status = f"unavailable: {error}"

    # Legacy entries keep their position; turbofit entries are appended so
    # pre-cutover automation that reads providers[0] still sees the legacy route.
    providers = legacy_sources(legacy_config) + turbofit_entries
    return {
        "schema": SOURCE_SCHEMA,
        "generated_at": now,
        "turbofit_url": turbofit_url or DEFAULT_TURBOFIT_URL,
        "turbofit_enabled": bool(enable_turbofit),
        "turbofit_status": turbofit_status,
        "key_envs": sorted({str(item.get("key_env")) for item in providers if item.get("key_env")}),
        "providers": providers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build non-secret benchmark provider sources")
    parser.add_argument("--legacy-config", type=Path, default=None,
                        help="Existing benchmark provider config JSON (compatibility until flip)")
    parser.add_argument("--turbofit-url", default=os.environ.get("TURBOFIT_BENCH_URL", DEFAULT_TURBOFIT_URL))
    parser.add_argument("--enable-turbofit", action="store_true",
                        help="Include Turbofit gateway entries (cutover decision; default off)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    enable = args.enable_turbofit or os.environ.get("TURBOFIT_BENCH_ENABLE", "").lower() in ("1", "true", "yes")
    legacy = {}
    if args.legacy_config and args.legacy_config.is_file():
        try:
            legacy = json.loads(args.legacy_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(json.dumps({"error": f"legacy config unreadable: {error}"}))
            return 2

    result = build_sources(legacy, turbofit_url=args.turbofit_url, enable_turbofit=enable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(args.output, 0o600)
    print(json.dumps({
        "output": str(args.output),
        "providers": len(result["providers"]),
        "turbofit_status": result["turbofit_status"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())