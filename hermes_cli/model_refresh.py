"""Refresh model catalogs used by the Hermes model picker."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _status_path() -> Path:
    return _hermes_home() / "cache" / "model_catalog_refresh_status.json"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _count_models_dev(data: dict[str, Any]) -> int:
    total = 0
    for provider in data.values():
        if not isinstance(provider, dict):
            continue
        models = provider.get("models")
        if isinstance(models, dict):
            total += len(models)
        elif isinstance(models, list):
            total += len(models)
    return total


def _run_step(name: str, func) -> dict[str, Any]:
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


def _load_dotenv() -> dict[str, Any]:
    from hermes_cli.config import load_env

    dotenv = load_env()
    applied = 0
    for key, value in dotenv.items():
        if key not in os.environ and value:
            os.environ[key] = value
            applied += 1
    return {
        "dotenv_loaded": len(dotenv),
        "dotenv_applied": applied,
    }


def _refresh_models_dev() -> dict[str, Any]:
    from agent.models_dev import fetch_models_dev

    data = fetch_models_dev(force_refresh=True)
    count = _count_models_dev(data if isinstance(data, dict) else {})
    return {
        "ok": count > 0,
        "cache": str(_hermes_home() / "models_dev_cache.json"),
        "models": count,
    }


def _refresh_ollama_cloud() -> dict[str, Any]:
    from hermes_cli.models import fetch_ollama_cloud_models

    models = fetch_ollama_cloud_models(force_refresh=True)
    return {
        "ok": bool(models),
        "cache": str(_hermes_home() / "ollama_cloud_models_cache.json"),
        "models": len(models),
    }


def _refresh_authenticated_picker() -> dict[str, Any]:
    from hermes_cli.config import get_compatible_custom_providers, load_config
    from hermes_cli.model_switch import list_authenticated_providers

    cfg = load_config() or {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    user_providers = cfg.get("providers")
    if not isinstance(user_providers, dict):
        user_providers = {}

    rows = list_authenticated_providers(
        current_provider=str(
            model_cfg.get("provider") or cfg.get("provider") or ""
        ),
        current_base_url=str(model_cfg.get("base_url") or cfg.get("base_url") or ""),
        user_providers=user_providers,
        custom_providers=get_compatible_custom_providers(cfg),
        max_models=10000,
        current_model=str(
            model_cfg.get("default")
            or model_cfg.get("model")
            or cfg.get("model")
            or ""
        ),
        live_refresh=True,
        cache_result=True,
    )
    return {
        "ok": bool(rows),
        "cache": str(_hermes_home() / "cache" / "authenticated_provider_models.json"),
        "provider_rows": len(rows),
        "total_models": sum(int(row.get("total_models") or 0) for row in rows),
    }


def refresh_model_catalogs(*, write_status: bool = True) -> dict[str, Any]:
    started = time.monotonic()
    env = _run_step("load_env", _load_dotenv)
    steps = [
        _run_step("models_dev", _refresh_models_dev),
        _run_step("ollama_cloud", _refresh_ollama_cloud),
        _run_step("authenticated_picker", _refresh_authenticated_picker),
    ]
    failures = [step["name"] for step in [env, *steps] if not step.get("ok")]
    payload = {
        "ok": not failures,
        "updated_at": _now_iso(),
        "duration_s": round(time.monotonic() - started, 3),
        "hermes_home": str(_hermes_home()),
        "failures": failures,
        "env": env,
        "steps": steps,
        "status_path": str(_status_path()),
    }
    if write_status:
        _write_json(_status_path(), payload)
    return payload


def format_refresh_summary(payload: dict[str, Any]) -> str:
    status = "ok" if payload.get("ok") else "failed"
    lines = [f"Model catalog refresh: {status} ({payload.get('duration_s', 0)}s)"]
    for step in payload.get("steps", []):
        if not isinstance(step, dict):
            continue
        marker = "ok" if step.get("ok") else "failed"
        bits = [f"  {step.get('name')}: {marker}"]
        if "models" in step:
            bits.append(f"{step.get('models')} models")
        if "total_models" in step:
            bits.append(f"{step.get('total_models')} picker models")
        if "provider_rows" in step:
            bits.append(f"{step.get('provider_rows')} providers")
        if step.get("error"):
            bits.append(str(step.get("error")))
        lines.append(" | ".join(bits))
    if payload.get("failures"):
        lines.append(f"Failures: {', '.join(str(x) for x in payload['failures'])}")
    lines.append(f"Status: {payload.get('status_path') or _status_path()}")
    return "\n".join(lines)


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
            _write_json(_status_path(), payload)
        except Exception as write_exc:
            payload["status_write_error"] = (
                f"{write_exc.__class__.__name__}: {write_exc}"
            )

    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(format_refresh_summary(payload))
    if wake_gate:
        print(json.dumps({"wakeAgent": False}))
    if exit_failure and not payload.get("ok"):
        return 1
    return 0
