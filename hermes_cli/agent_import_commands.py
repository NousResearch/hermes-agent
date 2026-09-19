"""Convert the portable subset of Claude Markdown commands into Hermes skills."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from utils import atomic_write_text


# Hermes does not implement Claude's argument substitution, file expansion or
# inline shell execution. Leaving these in an installed skill changes its meaning.
_CLAUDE_SYNTAX = re.compile(r"\$(?:ARGUMENTS\b|\d+|\{)|!`|(?<!\S)@\S+")


def convert_command(text: str, name: str) -> str:
    """Return SKILL.md content, or explain why manual conversion is required."""
    metadata = {}
    body = text
    if text.startswith("---\n"):
        closing = re.search(r"^---[ \t]*$", text[4:], re.MULTILINE)
        if closing is None:
            raise ValueError("Malformed command frontmatter")
        try:
            metadata = yaml.safe_load(text[4:4 + closing.start()]) or {}
        except yaml.YAMLError:
            raise ValueError("Malformed command frontmatter") from None
        if not isinstance(metadata, dict):
            raise ValueError("Command frontmatter must be a mapping")
        body = text[4 + closing.end():].removeprefix("\n")
    if set(metadata) - {"description"}:
        raise ValueError("Claude command metadata requires manual conversion (only description is portable)")
    if not body.strip():
        raise ValueError("Command has no instructions")
    if _CLAUDE_SYNTAX.search(body):
        raise ValueError("Claude argument, file or shell expansion requires manual conversion")
    description = metadata.get("description", f"Imported Claude command: {name}.")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Command description must be nonempty text")
    header = yaml.safe_dump({"name": f"claude-command-{name}", "description": description}, sort_keys=False)
    return f"---\n{header}---\n{body}"


def import_commands(importer, source_root: Path) -> None:
    """Use the existing preview/report boundary; never execute command content."""
    if not source_root.is_dir():
        return
    if source_root.is_symlink():
        importer.record("slash-command", source_root, None, "skipped", "Command directory must not be a symlink")
        return
    destination_root = importer.target_root / "skills" / "claude-code-commands"
    for source in sorted(source_root.glob("*.md")):
        destination = destination_root / source.stem / "SKILL.md"
        if source.is_symlink() or not source.is_file():
            importer.record("slash-command", source, None, "skipped", "Only regular command files are supported")
            continue
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", source.stem):
            importer.record("slash-command", source, None, "skipped", "Command name requires manual conversion")
            continue
        try:
            content = convert_command(source.read_text(encoding="utf-8"), source.stem)
        except (ValueError, OSError) as exc:
            importer.record("slash-command", source, None, "skipped", str(exc))
            continue
        # Do not follow existing redirects below the selected destination home,
        # including with --overwrite. The home itself may be user-symlinked.
        components = destination.relative_to(importer.target_root).parts
        cursor = importer.target_root
        redirected = False
        for component in components:
            cursor = cursor / component
            if cursor.is_symlink():
                redirected = True
                break
        if redirected:
            importer.record("slash-command", source, destination, "skipped", "Destination contains a symlink")
            continue
        if destination.parent.exists() and not importer.overwrite:
            from hermes_cli.agent_import_sync import skill_tree_digest
            ownership_key = f"claude-code-commands/{source.stem}"
            if ownership_key not in importer.sync_skills:
                importer.record("slash-command", source, destination, "conflict", "Destination command skill already exists")
                continue
            expected = importer.sync_skills[ownership_key]
            if expected is not None and skill_tree_digest(destination.parent) != expected:
                importer.record("slash-command", source, destination, "conflict",
                                "Imported command skill was modified locally — not refreshed")
                continue

        def write(destination=destination, content=content):
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(destination, content)
            except OSError as exc:
                return f"Could not write command skill: {exc}"
            return None

        importer.apply("slash-command", source, destination, "Would convert command to skill", write)
