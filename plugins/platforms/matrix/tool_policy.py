"""Shared Matrix tool policy; YAML wins over profile-scoped legacy env gates."""
from agent.secret_scope import get_secret
from hermes_cli.config import load_config_readonly


def tools_config() -> dict:
    # The canonical owner preserves managed policy and environment expansion.
    # Strict validation prevents malformed input from enabling legacy env gates.
    # Matrix tool action defaults remain in gate(), not merged DEFAULT_CONFIG.
    try:
        config = load_config_readonly(strict=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError("Cannot read Matrix tool policy; check config.yaml") from exc
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
