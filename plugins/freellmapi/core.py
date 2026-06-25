"""FreeLLMAPI Hermes plugin — setup, doctor, and status probes."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hermes_constants import display_hermes_home, get_hermes_home

DEFAULT_BASE_URL = "http://127.0.0.1:3001/v1"
REPO_INSTALL_URL = "https://freellmapi.co/install.sh"
GITHUB_URL = "https://github.com/tashfeenahmed/freellmapi"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _read_env_key(name: str) -> str:
    home = get_hermes_home()
    env_path = home / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return (os.environ.get(name) or "").strip()


def _resolve_base_url(config: dict[str, Any] | None = None) -> str:
    cfg = config or {}
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    for candidate in (
        _read_env_key("FREELLMAPI_BASE_URL"),
        str(model.get("base_url") or "").strip() if model.get("provider") == "freellmapi" else "",
        DEFAULT_BASE_URL,
    ):
        if candidate:
            return candidate.rstrip("/")
    return DEFAULT_BASE_URL


def _provider_registered() -> tuple[bool, str]:
    try:
        from providers import get_provider_profile

        profile = get_provider_profile("freellmapi")
    except Exception as exc:
        return False, f"provider import failed: {exc}"
    if profile is None:
        return False, "freellmapi model-provider profile not registered"
    return True, profile.display_name or "FreeLLMAPI"


def _probe_models(base_url: str, api_key: str, timeout: float = 6.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "hermes-freellmapi-plugin/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        return {
            "ok": False,
            "status_code": exc.code,
            "error": f"HTTP {exc.code}: {detail or exc.reason}",
            "models": [],
        }
    except Exception as exc:
        return {"ok": False, "status_code": None, "error": str(exc), "models": []}

    models: list[str] = []
    if isinstance(payload, dict):
        for item in payload.get("data") or []:
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
    return {
        "ok": True,
        "status_code": 200,
        "error": "",
        "models": models,
        "model_count": len(models),
    }


def _plugin_enabled(config: dict[str, Any]) -> bool:
    plugins = config.get("plugins") if isinstance(config.get("plugins"), dict) else {}
    enabled = plugins.get("enabled") if isinstance(plugins.get("enabled"), list) else []
    return "freellmapi" in enabled


def _ensure_plugin_enabled(config: dict[str, Any]) -> bool:
    plugins = config.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        plugins = {}
        config["plugins"] = plugins
    enabled = plugins.setdefault("enabled", [])
    if not isinstance(enabled, list):
        enabled = []
        plugins["enabled"] = enabled
    if "freellmapi" in enabled:
        return False
    enabled.append("freellmapi")
    return True


def _ensure_fallback_entry(config: dict[str, Any]) -> bool:
    entries = config.setdefault("fallback_providers", [])
    if not isinstance(entries, list):
        entries = []
        config["fallback_providers"] = entries
    for entry in entries:
        if isinstance(entry, dict) and entry.get("provider") == "freellmapi":
            return False
    entries.insert(0, {"provider": "freellmapi", "model": "auto"})
    return True


def _ensure_model_defaults(config: dict[str, Any]) -> bool:
    model = config.setdefault("model", {})
    if not isinstance(model, dict):
        model = {}
        config["model"] = model
    changed = False
    if model.get("provider") != "freellmapi":
        model["provider"] = "freellmapi"
        changed = True
    if not str(model.get("default") or "").strip():
        model["default"] = "auto"
        changed = True
    if not str(model.get("base_url") or "").strip():
        model["base_url"] = DEFAULT_BASE_URL
        changed = True
    return changed


def status() -> dict[str, Any]:
    config_path = get_hermes_home() / "config.yaml"
    config = _load_yaml(config_path)
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    registered, provider_label = _provider_registered()
    api_key = _read_env_key("FREELLMAPI_API_KEY")
    base_url = _resolve_base_url(config)
    return {
        "ok": registered and bool(api_key),
        "provider_registered": registered,
        "provider_label": provider_label,
        "plugin_enabled": _plugin_enabled(config),
        "model_provider": model.get("provider"),
        "model_default": model.get("default"),
        "base_url": base_url,
        "api_key_set": bool(api_key),
        "api_key_prefix": (api_key[:16] + "…") if len(api_key) > 16 else ("set" if api_key else ""),
        "config_path": str(config_path),
        "hermes_home": display_hermes_home(),
    }


def doctor() -> dict[str, Any]:
    base = status()
    checks: list[dict[str, Any]] = []
    ok = True

    def add(name: str, passed: bool, detail: str, *, required: bool = True) -> None:
        nonlocal ok
        if required and not passed:
            ok = False
        checks.append({"name": name, "ok": passed, "detail": detail, "required": required})

    add("model_provider_profile", base["provider_registered"], base["provider_label"])
    add("plugin_enabled", base["plugin_enabled"], "freellmapi in plugins.enabled")
    add("api_key", base["api_key_set"], "FREELLMAPI_API_KEY in ~/.hermes/.env or env")

    probe: dict[str, Any] | None = None
    if base["api_key_set"]:
        probe = _probe_models(base["base_url"], _read_env_key("FREELLMAPI_API_KEY"))
        if probe.get("ok"):
            detail = f"{probe.get('model_count', 0)} model(s) at {base['base_url']}/models"
        else:
            detail = probe.get("error") or "models probe failed"
        add("models_probe", bool(probe.get("ok")), detail)
    else:
        add(
            "models_probe",
            False,
            "skipped — set FREELLMAPI_API_KEY first (dashboard → Keys page)",
            required=False,
        )

    add(
        "model_config",
        base.get("model_provider") == "freellmapi",
        f"model.provider={base.get('model_provider')!r} (freellmapi recommended)",
        required=False,
    )

    return {
        "ok": ok,
        "checks": checks,
        "status": base,
        "models_probe": probe,
        "next_steps": _next_steps(base, probe),
    }


def _next_steps(base: dict[str, Any], probe: dict[str, Any] | None) -> list[str]:
    steps: list[str] = []
    if not base["plugin_enabled"]:
        steps.append("hermes plugins enable freellmapi")
    if not base["api_key_set"]:
        steps.append(
            f"Start FreeLLMAPI ({GITHUB_URL}), open the dashboard, copy the unified "
            "freellmapi-… key into ~/.hermes/.env as FREELLMAPI_API_KEY"
        )
    if probe and not probe.get("ok"):
        steps.append(
            f"Ensure FreeLLMAPI is listening at {base['base_url']} "
            f"(Docker: curl -fsSL {REPO_INSTALL_URL} | bash)"
        )
    if base.get("model_provider") != "freellmapi":
        steps.append("hermes freellmapi setup --apply-model")
    if not steps:
        steps.append("Ready — run a short Hermes chat with model provider freellmapi / model auto")
    return steps


def setup(*, apply_model: bool = False, enable_plugin: bool = True) -> dict[str, Any]:
    config_path = get_hermes_home() / "config.yaml"
    config = _load_yaml(config_path)
    changes: list[str] = []

    if enable_plugin and _ensure_plugin_enabled(config):
        changes.append("added freellmapi to plugins.enabled")
    if _ensure_fallback_entry(config):
        changes.append("inserted freellmapi/auto at fallback_providers[0]")
    if apply_model and _ensure_model_defaults(config):
        changes.append("set model.provider=freellmapi, model.default=auto")

    if changes:
        _save_yaml(config_path, config)

    result = doctor()
    result["changes"] = changes
    result["config_updated"] = bool(changes)
    return result


def to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
