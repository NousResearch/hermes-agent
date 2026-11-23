#!/usr/bin/env python3
"""Apply Unitalk distribution defaults to homes that already exist, on every container boot.

The image seeds config.yaml from cli-config.yaml.example only on a volume's first boot, so a
default added to the template never reaches an instance created earlier. This fills a default
in only where the home has not made its own choice; a section already chosen is never touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

from hermes_cli.config import atomic_config_write, require_readable_config_before_write
from hermes_cli.env_loader import selected_config_backend_name
from hermes_constants import get_hermes_home

# Keep identical to the image_gen block in cli-config.yaml.example (a test enforces it).
IMAGE_GEN_DEFAULT = {
    "provider": "openai",
    "max_parallel_requests": 4,
    "openai": {
        "base_url": "https://llmgateway.unitalk.ai/v1",
        "key_env": "UNITALK_API_KEY",
        "model": "fal_ai/fal-ai/nano-banana-pro",
    },
}


def with_image_gen_default(config: dict) -> dict | None:
    """``config`` with the Unitalk image default filled in, or None when the home already chose."""
    current = config.get("image_gen")
    if isinstance(current, dict) and (current.get("provider") or current.get("use_gateway")):
        return None
    section = dict(current) if isinstance(current, dict) else {}
    for key, value in IMAGE_GEN_DEFAULT.items():
        if isinstance(value, dict):
            existing = section.get(key)
            section[key] = {**value, **existing} if isinstance(existing, dict) else dict(value)
        else:
            section.setdefault(key, value)
    return {**config, "image_gen": section}


def _config_files() -> list[Path]:
    from hermes_cli.profiles import profiles_to_serve

    homes = [home for _, home in profiles_to_serve(True, include_standalone=True, include_parked=True)]
    return [home / "config.yaml" for home in dict.fromkeys(homes) if (home / "config.yaml").is_file()]


def main() -> int:
    if selected_config_backend_name(get_hermes_home()) == "remote":
        print("[unitalk-defaults] remote config backend: nothing to apply")
        return 0
    for path in _config_files():
        updated = with_image_gen_default(require_readable_config_before_write(path))
        if updated is None:
            continue
        atomic_config_write(path, updated)
        print(f"[unitalk-defaults] added the Unitalk image_gen default to {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # a boot must never fail because a default could not be applied
        print(f"[unitalk-defaults] WARNING: {exc}", file=sys.stderr)
        raise SystemExit(0)
