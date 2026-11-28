"""Configured, additive context files; discovery and content safety stay in prompt_builder."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def configured_context(home_override: Path | None = None) -> dict:
    """Read context options once, bound to the owning profile when explicitly supplied."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_cli.config import load_config_readonly
    from hermes_constants import get_hermes_home

    if home_override is None or home_override == get_hermes_home():
        context = load_config_readonly().get("context")
    else:
        with bind_profile_scope(home_override, hydrate_secrets=False):
            context = load_config_readonly().get("context")
    if not isinstance(context, dict):
        logger.warning("Ignoring context configuration: expected a mapping")
        return {}
    return context


def _external_file_specs(context: dict) -> list[tuple[Path, str]]:
    """Validate optional entries and deduplicate resolved paths, preserving declaration order."""
    raw_files = context.get("external_files", [])
    if not isinstance(raw_files, list):
        logger.warning("Ignoring context.external_files: expected a list")
        return []
    specs = []
    seen: set[Path] = set()
    for item in raw_files:
        path_value = item.get("path") if isinstance(item, dict) else item
        label = item.get("label") if isinstance(item, dict) else None
        if not isinstance(path_value, str) or not path_value.strip():
            logger.warning("Ignoring context.external_files entry: expected a non-empty path")
            continue
        try:
            path = Path(os.path.expandvars(path_value.strip())).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Could not resolve configured context path: %s", exc)
            continue
        if path in seen:
            continue
        seen.add(path)
        specs.append((path, label.strip() if isinstance(label, str) and label.strip() else str(path)))
    return specs


def external_context_candidates(context: dict) -> tuple[list[tuple[str, Path, str | None]], set[Path]]:
    """Read configured candidates once; content includes the provenance scanned by the builder.

    Empty/unreadable candidates remain visible to the manifest. Only non-empty candidates contribute
    resolved paths to automatic-discovery exclusions, matching prompt assembly.
    """
    from agent.prompt_builder import _read_context_file, _strip_yaml_frontmatter

    candidates = []
    loaded_paths: set[Path] = set()
    for path, label in _external_file_specs(context):
        content = _read_context_file(path, path.parent)
        if content:
            if path.name in (".hermes.md", "HERMES.md"):
                content = _strip_yaml_frontmatter(content)
            content = f"Source: {path}\n\n{content}"
            loaded_paths.add(path)
        candidates.append((label, path, content))
    return candidates, loaded_paths


def load_external_context_files(context: dict, context_length: int | None = None) -> tuple[str, set[Path]]:
    """Render shared external candidates through the existing scanner and truncation pipeline."""
    from agent.prompt_builder import _context_section

    candidates, loaded_paths = external_context_candidates(context)
    sections = [_context_section(content, label, label, path, context_length)
                for label, path, content in candidates if content]
    return "\n".join(sections), loaded_paths
