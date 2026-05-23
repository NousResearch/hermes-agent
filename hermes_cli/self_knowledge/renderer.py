"""Render Hermes self-knowledge markdown by refreshing AUTO blocks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from hermes_cli.self_knowledge import generators
from hermes_cli.self_knowledge.parser import parse_auto_blocks, replace_auto_blocks


DOC_PATH = Path("context/self/hermes-agent.md")
Generator = Callable[[], str]
GENERATORS: dict[str, Generator] = {
    "capabilities": generators.generate_capabilities,
    "toolsets": generators.generate_toolsets,
    "slash_commands": generators.generate_slash_commands,
    "gateway_platforms": generators.generate_gateway_platforms,
    "voice_loop": generators.generate_voice_loop,
    "skills_profiles": generators.generate_skills_profiles,
    "plugins_integrations": generators.generate_plugins_integrations,
    "recent_activity": generators.generate_recent_activity,
}


def _render_block(name: str) -> str:
    generator = GENERATORS.get(name)
    if generator is None:
        return f"_unavailable: no generator registered for `{name}`_"
    try:
        rendered = generator()
    except Exception as exc:  # pragma: no cover - defensive boundary
        return f"_unavailable: generator `{name}` failed ({type(exc).__name__})_"
    rendered = rendered.strip()
    return rendered or f"_unavailable: generator `{name}` returned no content_"


def render_self_knowledge(doc_path: Path = DOC_PATH) -> str:
    """Return *doc_path* with every AUTO block body freshly rendered."""
    text = Path(doc_path).read_text(encoding="utf-8")
    blocks = parse_auto_blocks(text)
    replacements = {name: _render_block(name) for name in blocks}
    return replace_auto_blocks(text, replacements)


def refresh_self_knowledge(doc_path: Path = DOC_PATH) -> bool:
    """Write rendered self-knowledge to disk. Return True when the file changed."""
    path = Path(doc_path)
    original = path.read_text(encoding="utf-8")
    rendered = render_self_knowledge(path)
    if rendered == original:
        return False
    path.write_text(rendered, encoding="utf-8")
    return True
