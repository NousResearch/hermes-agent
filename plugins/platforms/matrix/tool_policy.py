"""Shared Matrix tool policy; YAML wins over profile-scoped legacy env gates."""
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home


def tools_config() -> dict:
    # Gate checks need explicit keys (merged defaults hide legacy env fallback).
    # Unlike the general raw-config reader, malformed policy must not become an
    # empty mapping that re-enables permissive legacy environment values.
    import yaml

    try:
        with (get_hermes_home() / "config.yaml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError("Cannot read Matrix tool policy; check config.yaml") from exc
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("Matrix tool policy requires a config mapping")
    matrix = config.get("matrix", {})
    if not isinstance(matrix, dict):
        raise ValueError("matrix must be a config mapping")
    tools = matrix.get("tools", {})
    if not isinstance(tools, dict):
        raise ValueError("matrix.tools must be a config mapping")
    return tools


def parse_bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1", "yes", "on"}:
            return True
        if value.strip().lower() in {"false", "0", "no", "off", ""}:
            return False
    return default


def gate(key: str, env_name: str, default=False) -> bool:
    tools = tools_config()
    value = tools[key] if key in tools else get_secret(env_name)
    return parse_bool(value, default)
