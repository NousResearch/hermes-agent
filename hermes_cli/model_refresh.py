"""Refresh Hermes model catalogs from one shared entrypoint.

This module is intentionally non-agentic.  It refreshes the model caches that
feed the CLI, gateway, TUI, dashboard, and cron model picker surfaces.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _status_path() -> Path:
    return _hermes_home() / "cache" / "model_catalog_refresh_status.json"


def _model_catalog_cache_path() -> Path:
    return _hermes_home() / "cache" / "model_catalog.json"


def _repo_model_catalog_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "website"
        / "static"
        / "api"
        / "model-catalog.json"
    )


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_runtime_env() -> dict[str, Any]:
    """Load Hermes env files into this process without overriding live env."""

    from hermes_cli.config import load_env

    dotenv_vars = load_env()
    dotenv_applied = 0
    for key, value in dotenv_vars.items():
        key = str(key).strip()
        value = str(value).strip()
        if not key or not value or key in os.environ:
            continue
        os.environ[key] = value
        dotenv_applied += 1

    zshenv_vars: dict[str, str] = {}
    try:
        from hermes_cli.config import _load_zshenv_exports

        zshenv_vars = _load_zshenv_exports()
    except Exception:
        zshenv_vars = {}

    return {
        "dotenv_loaded": len(dotenv_vars),
        "dotenv_applied": dotenv_applied,
        "zshenv_loaded": len(zshenv_vars),
        # Do not inject ~/.zshenv wholesale into os.environ here. The config
        # layer reads simple exports on demand via get_env_value(); exporting
        # every parsed line would turn unexpanded shell expressions such as
        # "$ZEROBREW_PREFIX/..." into live SSL/PATH variables and break HTTPS.
        "zshenv_applied": 0,
    }


def _count_models_dev(data: dict[str, Any]) -> dict[str, int]:
    providers = 0
    models = 0
    for provider in data.values():
        if not isinstance(provider, dict):
            continue
        providers += 1
        provider_models = provider.get("models")
        if isinstance(provider_models, dict):
            models += len(provider_models)
        elif isinstance(provider_models, list):
            models += len(provider_models)
    return {"providers": providers, "models": models}


def _count_catalog(data: dict[str, Any]) -> dict[str, Any]:
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return {"providers": 0, "models": 0, "provider_counts": {}}

    provider_counts: dict[str, int] = {}
    for name, block in providers.items():
        if not isinstance(block, dict):
            continue
        models = block.get("models")
        provider_counts[str(name)] = len(models) if isinstance(models, list) else 0
    return {
        "providers": len(provider_counts),
        "models": sum(provider_counts.values()),
        "provider_counts": provider_counts,
    }


def _catalog_timestamp(data: dict[str, Any]) -> float:
    raw = str(data.get("updated_at") or "").strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _prefer_repo_manifest(repo_data: dict[str, Any], fetched_data: dict[str, Any]) -> bool:
    """Return True when the checked-out manifest is newer or clearly richer."""

    repo_counts = _count_catalog(repo_data)
    fetched_counts = _count_catalog(fetched_data)
    repo_ts = _catalog_timestamp(repo_data)
    fetched_ts = _catalog_timestamp(fetched_data)
    if repo_ts and fetched_ts and repo_ts != fetched_ts:
        return repo_ts > fetched_ts
    return int(repo_counts["models"]) > int(fetched_counts["models"])


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp_path, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _run_step(name: str, func: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = func()
        ok = bool(result.pop("ok", True))
        return {
            "name": name,
            "ok": ok,
            "duration_s": round(time.monotonic() - started, 3),
            **result,
        }
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "duration_s": round(time.monotonic() - started, 3),
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def _refresh_models_dev() -> dict[str, Any]:
    from agent.models_dev import fetch_models_dev

    data = fetch_models_dev(force_refresh=True)
    counts = _count_models_dev(data if isinstance(data, dict) else {})
    return {
        "ok": counts["providers"] > 0 and counts["models"] > 0,
        "cache": str(_hermes_home() / "models_dev_cache.json"),
        **counts,
    }


def _refresh_remote_manifest() -> dict[str, Any]:
    from hermes_cli.model_catalog import get_catalog, reset_cache

    reset_cache()
    data = get_catalog(force_refresh=True)
    data = data if isinstance(data, dict) else {}
    source = "remote"
    fetched_counts = _count_catalog(data)

    repo_path = _repo_model_catalog_path()
    repo_data = _read_json(repo_path)
    if repo_data and _prefer_repo_manifest(repo_data, data):
        data = repo_data
        source = "repo_static"
        _write_json(_model_catalog_cache_path(), repo_data)
        reset_cache()
        get_catalog(force_refresh=False)

    counts = _count_catalog(data)
    return {
        "ok": counts["providers"] > 0 and counts["models"] > 0,
        "cache": str(_model_catalog_cache_path()),
        "repo_manifest": str(repo_path),
        "source": source,
        "updated_at": str(data.get("updated_at") or ""),
        "fetched_models": fetched_counts["models"],
        **counts,
    }


def _refresh_ollama_cloud() -> dict[str, Any]:
    from hermes_cli.models import fetch_ollama_cloud_models

    models = fetch_ollama_cloud_models(force_refresh=True)
    return {
        "ok": bool(models),
        "cache": str(_hermes_home() / "ollama_cloud_models_cache.json"),
        "models": len(models),
    }


def _snapshot_authenticated_picker() -> dict[str, Any]:
    from hermes_cli.config import get_compatible_custom_providers, load_config
    from hermes_cli.model_switch import list_authenticated_providers

    cfg = load_config() or {}
    user_providers = cfg.get("providers")
    if not isinstance(user_providers, dict):
        user_providers = {}
    rows = list_authenticated_providers(
        current_provider=str(cfg.get("provider") or ""),
        current_base_url=str(cfg.get("base_url") or ""),
        user_providers=user_providers,
        custom_providers=get_compatible_custom_providers(cfg),
        max_models=10000,
        current_model=str(cfg.get("model") or ""),
        live_refresh=True,
        cache_result=True,
    )
    providers = [
        {
            "slug": str(row.get("slug") or ""),
            "name": str(row.get("name") or ""),
            "source": str(row.get("source") or ""),
            "models": int(row.get("total_models") or 0),
        }
        for row in rows
    ]
    return {
        "ok": bool(providers),
        "providers": providers,
        "provider_rows": len(providers),
        "total_models": sum(item["models"] for item in providers),
    }


def write_refresh_status(payload: dict[str, Any]) -> None:
    _write_json(_status_path(), payload)


def refresh_model_catalogs(*, write_status: bool = True) -> dict[str, Any]:
    """Refresh all model catalogs needed by picker surfaces."""

    started = time.monotonic()
    env_result = _run_step("load_env", _load_runtime_env)
    steps = [
        _run_step("models_dev", _refresh_models_dev),
        _run_step("remote_model_manifest", _refresh_remote_manifest),
        _run_step("ollama_cloud", _refresh_ollama_cloud),
        _run_step("authenticated_picker", _snapshot_authenticated_picker),
    ]
    failures = [step["name"] for step in steps if not step.get("ok")]
    payload = {
        "ok": not failures,
        "updated_at": _now_iso(),
        "duration_s": round(time.monotonic() - started, 3),
        "hermes_home": str(_hermes_home()),
        "env": env_result,
        "failures": failures,
        "steps": steps,
        "status_path": str(_status_path()),
    }

    if write_status:
        try:
            write_refresh_status(payload)
        except Exception as exc:
            payload.setdefault("failures", []).append("write_status")
            payload["ok"] = False
            payload["status_write_error"] = f"{exc.__class__.__name__}: {exc}"
    return payload


def format_refresh_summary(payload: dict[str, Any]) -> str:
    """Return a compact human-readable refresh summary."""

    ok = bool(payload.get("ok"))
    status = "ok" if ok else "failed"
    duration = payload.get("duration_s", 0)
    lines = [f"Model catalog refresh: {status} ({duration}s)"]
    for step in payload.get("steps", []):
        if not isinstance(step, dict):
            continue
        marker = "ok" if step.get("ok") else "failed"
        name = str(step.get("name") or "step")
        bits = [f"  {name}: {marker}"]
        if "models" in step:
            bits.append(f"{step.get('models')} models")
        if "total_models" in step:
            bits.append(f"{step.get('total_models')} picker models")
        if "provider_rows" in step:
            bits.append(f"{step.get('provider_rows')} providers")
        if step.get("source"):
            bits.append(f"source={step.get('source')}")
        if step.get("error"):
            bits.append(str(step.get("error")))
        lines.append(" | ".join(bits))
    if payload.get("failures"):
        lines.append(f"Failures: {', '.join(str(x) for x in payload['failures'])}")
    lines.append(f"Status: {payload.get('status_path') or _status_path()}")
    return "\n".join(lines)


def print_refresh_result(
    payload: dict[str, Any],
    *,
    json_output: bool = False,
    wake_gate: bool = False,
) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(format_refresh_summary(payload))
    if wake_gate:
        print(json.dumps({"wakeAgent": False}))


def cli_main(
    *,
    json_output: bool = False,
    wake_gate: bool = False,
    exit_failure: bool = True,
) -> int:
    try:
        payload = refresh_model_catalogs(write_status=True)
    except Exception as exc:
        payload = {
            "ok": False,
            "updated_at": _now_iso(),
            "hermes_home": str(_hermes_home()),
            "failures": ["unhandled"],
            "error": f"{exc.__class__.__name__}: {exc}",
            "status_path": str(_status_path()),
        }
        try:
            write_refresh_status(payload)
        except Exception as write_exc:
            payload.setdefault("failures", []).append("write_status")
            payload["status_write_error"] = (
                f"{write_exc.__class__.__name__}: {write_exc}"
            )

    print_refresh_result(payload, json_output=json_output, wake_gate=wake_gate)
    if exit_failure and not payload.get("ok"):
        return 1
    return 0
