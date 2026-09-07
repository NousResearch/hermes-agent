"""Profile-scoped Groupchat settings under the dashboard's existing auth gate."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
from pathlib import Path
import yaml

from plugins.groupchat.config import (
    GroupchatSettings,
    available_platforms,
    matrix_require_mention,
    validate_settings,
)

router = APIRouter()


def groupchat_participation():
    """Return non-secret, saved Matrix participation state for named profiles."""
    from hermes_cli.profiles import list_profiles
    result = []
    for info in list_profiles():
        if info.is_default or not info.gateway_running:
            continue
        home = Path(info.path)
        try:
            config = yaml.safe_load((home / "config.yaml").read_text()) or {}
        except (OSError, yaml.YAMLError):
            config = {}
        groupchat = config.get("groupchat") or {}
        plugins = config.get("plugins") or {}
        matrix = ((config.get("platforms") or {}).get("matrix") or {})
        plugin_enabled = "groupchat" in (plugins.get("enabled") or []) and "groupchat" not in (plugins.get("disabled") or [])
        relevance_enabled = bool((groupchat.get("relevance") or {}).get("enabled"))
        groupchat_enabled = bool(groupchat.get("enabled") and plugin_enabled and matrix.get("enabled") and "matrix" in (groupchat.get("platforms") or []))
        require_mention = matrix_require_mention(home, config)
        result.append({"profile": info.name, "require_mention": require_mention,
                       "groupchat_enabled": groupchat_enabled,
                       "participates": bool(groupchat_enabled and relevance_enabled and not require_mention)})
    return result


class SettingsUpdate(BaseModel):
    settings: dict[str, Any]


@router.get("/settings")
def get_settings(profile: str | None = None):
    from hermes_cli.web_server import _config_profile_scope
    from hermes_cli.config import read_raw_config
    with _config_profile_scope(profile):
        from hermes_constants import get_hermes_home
        from plugins.groupchat.audit import log_paths
        config = read_raw_config()
        try:
            settings = validate_settings(config.get("groupchat"))
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, "Invalid Groupchat configuration") from exc
        plugins = config.get("plugins") or {}
        return {"settings": settings, "profile": profile or "current",
                "defaults": validate_settings({}),
                "available_platforms": available_platforms(),
                "decision_log_directory": str(Path(get_hermes_home()) / "logs"),
                "persistent_context_directory": str(Path(get_hermes_home()) / "groupchat"),
                "decision_logs": {platform: log_paths(get_hermes_home(), platform)
                                  for platform in available_platforms()},
                "groupchat_participation": groupchat_participation(),
                "plugin_enabled": "groupchat" in (plugins.get("enabled") or [])
                    and "groupchat" not in (plugins.get("disabled") or []),
                "restart_required": True}


@router.put("/settings")
def put_settings(body: SettingsUpdate, profile: str | None = None):
    from hermes_cli.web_server import _config_profile_scope, _CONFIG_MUTATION_LOCK
    from hermes_cli.config import read_raw_config, save_config
    try:
        settings = validate_settings(body.settings)
    except (ValueError, TypeError) as exc:
        # Validation contains no credentials; expose the field/line diagnostic.
        detail = str(exc) if str(exc).startswith(('Invalid system pattern', 'Invalid multiline pattern', 'Invalid silence pattern', 'Invalid literal pattern', 'Too many ')) else "Invalid Groupchat settings: check provider, model and numeric limits"
        raise HTTPException(422, detail) from exc
    with _config_profile_scope(profile), _CONFIG_MUTATION_LOCK:
        config = read_raw_config()
        plugins = dict(config.get("plugins") or {})
        if settings["enabled"]:
            plugins["enabled"] = list(dict.fromkeys([*(plugins.get("enabled") or []), "groupchat"]))
            plugins["disabled"] = [name for name in plugins.get("disabled", []) if name != "groupchat"]
        config["groupchat"] = settings
        config["plugins"] = plugins
        save_config(config)
    return {"ok": True, "restart_required": True}
