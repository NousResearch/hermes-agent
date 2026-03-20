from __future__ import annotations

from pathlib import Path

from hermes_cli.default_soul import DEFAULT_SOUL_MD
from hermes_cli.product_config import load_product_config


def default_product_soul() -> str:
    return DEFAULT_SOUL_MD.strip() + "\n"


def _runtime_capability_overlay(config: dict | None = None) -> str:
    product_config = config or load_product_config()
    toolsets = product_config.get("tools", {}).get("hermes_toolsets", [])
    normalized = [str(item).strip() for item in toolsets if str(item).strip()]
    if not normalized:
        normalized = ["memory", "session_search"]
    rendered_toolsets = ", ".join(normalized)
    return (
        "\n## Product Runtime Contract\n\n"
        "You are running inside a Hermes Core product runtime.\n\n"
        f"Your currently enabled Hermes toolsets are: {rendered_toolsets}.\n\n"
        "If someone asks what tools or capabilities you have, answer only from the toolsets enabled in this runtime.\n"
        "Do not describe the full Hermes tool universe unless those toolsets are actually enabled here.\n"
        "Admin permissions in the web app do not grant extra runtime tools.\n"
    )


def resolve_product_soul_template_path(config: dict | None = None) -> Path | None:
    product_config = config or load_product_config()
    raw_path = (
        str(product_config.get("product", {}).get("agent", {}).get("soul_template_path", "")).strip()
    )
    if not raw_path:
        return None
    return Path(raw_path).expanduser().resolve()


def render_product_soul(config: dict | None = None) -> str:
    product_config = config or load_product_config()
    template_path = resolve_product_soul_template_path(config)
    if template_path is None:
        return default_product_soul().rstrip() + _runtime_capability_overlay(product_config).rstrip() + "\n"
    if not template_path.exists():
        raise FileNotFoundError(f"Configured SOUL.md template was not found: {template_path}")
    content = template_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Configured SOUL.md template is empty: {template_path}")
    return content + _runtime_capability_overlay(product_config).rstrip() + "\n"
