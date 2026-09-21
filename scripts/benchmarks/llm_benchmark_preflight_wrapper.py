#!/usr/bin/env python3
"""Executable preflight wrapper for llm-benchmark-weekly.

Runs the bounded credentialed measurements collector and the deterministic
preflight builder in one shot, writing both state files atomically and
emitting the preflight JSON to stdout for LLM context.

Design contract:
  * PYTHONPATH-independent: bootstraps sys.path from its own location so the
    scripts.* imports resolve regardless of how it was invoked.
  * Bounded: measurements capped at --max-providers / --timeout-seconds.
  * Deterministic: all paths derive from HERMES_HOME and the live
    config/catalogue/source snapshot — no hardcoded paths.
  * Atomic state writes: each state file is written to a temp sibling then
    os.replace'd so a partial write can never leave a corrupt JSON behind.
  * Fail-closed: any error exits non-zero with a short diagnostic on stderr;
    the preflight JSON on stdout is only emitted on full success.  Credentials
    are never read from files, never serialised, and never echoed — the
    measurements collector reads keys solely from os.environ (which the cron
    scheduler sanitises), so absent keys produce credential_unavailable rows
    rather than leaking secrets.

Intended to be attached as a no_agent cron script for llm-benchmark-weekly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bootstrap sys.path so `scripts.*` imports resolve even when invoked outside
# the repo (cron runs us with sys.executable and an overlay PYTHONPATH, but a
# standalone or manual invocation may not have it set).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402  (after sys.path bootstrap)

try:  # repo layout: scripts/benchmarks/...; deployed layout: flat scripts/...
    from scripts.benchmarks.llm_benchmark_measurements import collect  # noqa: E402
    from scripts.benchmarks.llm_benchmark_weekly import build_preflight  # noqa: E402
except ModuleNotFoundError:
    from scripts.llm_benchmark_measurements import collect  # noqa: E402
    from scripts.llm_benchmark_weekly import build_preflight  # noqa: E402

STATE_DIR_NAME = "llm-benchmark-weekly"
MEASUREMENTS_FILE = "measurements.json"
PREFLIGHT_FILE = "preflight.json"
SOURCES_FILE = "sources.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_hermes_home(arg: Path | None) -> Path:
    home = arg or Path(os.environ.get("HERMES_HOME", "") or Path.home() / ".hermes")
    home.mkdir(parents=True, exist_ok=True)
    return home.resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        val = yaml.safe_load(fh)
    return val if isinstance(val, dict) else {}


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def build_provider_routes(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive the non-secret provider route config from config.yaml.

    Each entry has: name, model, endpoint, key_env.  Credentials are named by
    key_env (an env-var name, never the value) so the measurements collector
    can read them from os.environ without the wrapper ever touching secrets.
    """
    routes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for prov in config.get("custom_providers", []) or []:
        if not isinstance(prov, dict):
            continue
        name = str(prov.get("name") or "")
        if not name or name in seen:
            continue
        models = prov.get("models") or []
        model = prov.get("default_model") or prov.get("model") or (models[0] if models else "")
        endpoint = prov.get("base_url") or ""
        if not endpoint:
            continue
        key_env = str(prov.get("key_env") or "")
        routes.append({"name": name, "model": str(model), "endpoint": str(endpoint), "key_env": key_env})
        seen.add(name)
    # Built-in providers with a base_url (e.g. local ollama).
    for name, cfg in (config.get("providers") or {}).items():
        if not isinstance(cfg, dict) or name in seen:
            continue
        endpoint = cfg.get("base_url") or ""
        if not endpoint:
            continue
        key_env = str(cfg.get("key_env") or "")
        if not key_env and cfg.get("api_key"):
            # Built-in providers may inline a non-secret placeholder like
            # 'not-needed'; only treat it as an env-var name if it looks like
            # one (uppercase + underscore), otherwise pass empty so the
            # collector treats it as credential_unavailable.
            candidate = str(cfg.get("api_key") or "")
            key_env = candidate if candidate.isupper() and "_" in candidate else ""
        model = str(cfg.get("model") or config.get("model", {}).get("default", "") or "")
        routes.append({"name": str(name), "model": model, "endpoint": str(endpoint), "key_env": key_env})
        seen.add(name)
    return routes


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to a temp sibling then atomically replace the target.

    Guarantees a reader never sees a partially-written file.  The temp file is
    created in the same directory so the rename is atomic on POSIX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.chmod(tmp, 0o644 if path.name == PREFLIGHT_FILE else 0o600)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def run(
    hermes_home: Path,
    *,
    timeout_seconds: float = 30,
    max_providers: int = 10,
    pythonpath_root: Path | None = None,
) -> dict[str, Any]:
    """Run measurements + preflight, write state atomically, return preflight."""
    state_dir = hermes_home / "cron_states" / STATE_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)

    # --- Provider route config from config.yaml (non-secret) ---
    config = _read_yaml(hermes_home / "config.yaml")
    provider_routes = build_provider_routes(config)

    # --- Bounded credentialed measurements ---
    # Credentials come only from os.environ; the wrapper never reads .env or
    # any secret file, so in a sanitised cron env these yield credential_unavailable.
    measurements = collect(
        {"providers": provider_routes},
        timeout_seconds=min(max(timeout_seconds, 1), 60),
        max_providers=min(max(max_providers, 1), 20),
    )
    _atomic_write_json(state_dir / MEASUREMENTS_FILE, measurements)

    # --- Deterministic preflight from live snapshot paths ---
    catalogue_path = hermes_home / "cache" / "model_catalog.json"
    sources_path = state_dir / SOURCES_FILE
    previous_path = state_dir / PREFLIGHT_FILE
    # Read previous BEFORE we overwrite it.
    previous = _read_json(previous_path, None)

    preflight = build_preflight(
        hermes_home,
        catalogue_path,
        sources_path,
        previous_path,  # build_preflight reads it internally; pass the path
    )
    # Ensure previous_snapshot reflects what we read (build_preflight reads the
    # file itself, which is the same path — consistent).
    _atomic_write_json(state_dir / PREFLIGHT_FILE, preflight)
    return preflight


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the llm-benchmark-weekly preflight and emit JSON to stdout.",
    )
    parser.add_argument("--hermes-home", type=Path, default=None, help="HERMES_HOME (default: $HERMES_HOME or ~/.hermes)")
    parser.add_argument("--timeout-seconds", type=float, default=30, help="Per-request measurement timeout (1-60)")
    parser.add_argument("--max-providers", type=int, default=10, help="Max providers to measure (1-20)")
    args = parser.parse_args()

    hermes_home = _resolve_hermes_home(args.hermes_home)

    try:
        preflight = run(
            hermes_home,
            timeout_seconds=args.timeout_seconds,
            max_providers=args.max_providers,
        )
    except Exception as exc:
        # Fail closed: short diagnostic on stderr, nothing on stdout, non-zero exit.
        print(f"ERROR: preflight failed: {exc}", file=sys.stderr)
        return 1

    # Emit preflight JSON to stdout for LLM context.
    json.dump(preflight, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
