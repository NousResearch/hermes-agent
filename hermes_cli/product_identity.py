from __future__ import annotations

from pathlib import Path

from hermes_cli.default_soul import DEFAULT_SOUL_MD
from hermes_cli.product_config import load_product_config


def default_product_soul() -> str:
    return DEFAULT_SOUL_MD.strip() + "\n"


def resolve_product_soul_template_path(config: dict | None = None) -> Path | None:
    product_config = config or load_product_config()
    raw_path = (
        str(product_config.get("product", {}).get("agent", {}).get("soul_template_path", "")).strip()
    )
    if not raw_path:
        return None
    return Path(raw_path).expanduser().resolve()


def render_product_soul(config: dict | None = None) -> str:
    template_path = resolve_product_soul_template_path(config)
    if template_path is None:
        return default_product_soul()
    if not template_path.exists():
        raise FileNotFoundError(f"Configured SOUL.md template was not found: {template_path}")
    content = template_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Configured SOUL.md template is empty: {template_path}")
    return content + "\n"
