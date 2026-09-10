from __future__ import annotations

import json
import os
from pathlib import Path

from robin.files import atomic_write_text

LEGACY_SPLIT_LAYOUT_KEY = "vault" + "_path"


def state_dir(explicit_state_dir: str | None = None) -> Path:
    value = explicit_state_dir or os.environ.get("ROBIN_STATE_DIR")
    if not value:
        raise SystemExit(
            "Robin state directory is not configured. Pass --state-dir or set ROBIN_STATE_DIR."
        )
    return Path(value).expanduser().resolve()


def config_path(explicit_state_dir: str | None = None) -> Path:
    return state_dir(explicit_state_dir) / "robin-config.json"


def index_path(explicit_state_dir: str | None = None) -> Path:
    return state_dir(explicit_state_dir) / "robin-review-index.json"


def _confined_subpath(root: Path, value: str, field: str) -> Path:
    """Resolve a configured subdirectory, refusing to leave the state directory.

    Absolute values (``Path(root) / "/etc"`` discards the base) and ``..`` escapes
    would otherwise let topic/media operations touch paths outside the state root.
    """
    root = root.resolve()
    candidate = (root / value).resolve()
    if candidate != root and root not in candidate.parents:
        raise SystemExit(
            f"Config {field}={value!r} must be a relative path inside the state directory."
        )
    return candidate


def topics_path(config: dict, explicit_state_dir: str | None = None) -> Path:
    return _confined_subpath(state_dir(explicit_state_dir), config.get("topics_dir", "topics"), "topics_dir")


def media_path(config: dict, explicit_state_dir: str | None = None) -> Path:
    return _confined_subpath(state_dir(explicit_state_dir), config.get("media_dir", "media"), "media_dir")


def load_config(explicit_state_dir: str | None = None) -> dict:
    path = config_path(explicit_state_dir)
    try:
        with path.open(encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Config not found at {path}. Create robin-config.json in the state directory first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Config at {path} is invalid JSON: {exc}") from exc

    if LEGACY_SPLIT_LAYOUT_KEY in config:
        raise SystemExit(
            f"Config at {path} uses the old split-layout field for a separate content root. "
            "Robin now stores topics and media inside the state directory. "
            "Remove the legacy content-root field and keep only topics_dir/media_dir/review settings."
        )

    return config


def empty_index() -> dict:
    return {"items": {}}


def load_index(explicit_state_dir: str | None = None) -> dict:
    path = index_path(explicit_state_dir)
    if path.exists():
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Review index at {path} is invalid JSON: {exc}") from exc
    else:
        data = empty_index()

    data.setdefault("items", {})
    return data


def save_index(index: dict, explicit_state_dir: str | None = None) -> None:
    path = index_path(explicit_state_dir)
    atomic_write_text(path, json.dumps(index, indent=2, sort_keys=True) + "\n")
