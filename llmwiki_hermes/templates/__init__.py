"""Template loading and rendering."""

from __future__ import annotations

from importlib import resources
from string import Template


def render_template(name: str, values: dict[str, str]) -> str:
    """Render a bundled template with string.Template semantics."""

    raw = resources.files("llmwiki_hermes.templates").joinpath(name).read_text(encoding="utf-8")
    return Template(raw).safe_substitute(values).rstrip() + "\n"
