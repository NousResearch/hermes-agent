"""Dotted-path access into a raw config.yaml mapping, for the post-update config restore."""
from pathlib import Path
from typing import Any, Optional


def _read_raw_yaml_dict(path: Path) -> Optional[dict[str, Any]]:
    """Parse ``path`` as a YAML mapping. ``None`` = missing/unreadable/non-dict."""
    if not path.is_file():
        return None
    try:
        import hermes_yaml as yaml

        with open(path, "r", encoding="utf-8-sig") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _get_config_path_value(data: dict[str, Any], dotted: tuple[str, ...]) -> Any:
    node: Any = data
    for key in dotted:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _set_config_path_value(data: dict[str, Any], dotted: tuple[str, ...], value: Any) -> None:
    node = data
    for key in dotted[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[dotted[-1]] = value
