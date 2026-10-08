"""Opt-in setup for shared instruction files, without changing project discovery."""

from pathlib import Path


_EXTERNAL_CONTEXT_FILE_CANDIDATES = (
    ("Codex AGENTS.md", ".codex/AGENTS.md"),
    ("Claude CLAUDE.md", ".claude/CLAUDE.md"),
)


def _context_file_key(path: Path) -> tuple:
    """Treat hardlinks and symlinks as one choice, including already configured aliases."""
    try:
        info = path.stat()
        if info.st_ino:
            return ("inode", info.st_dev, info.st_ino)
        return ("path", path.resolve())
    except (OSError, ValueError, RuntimeError):
        return ("path", path.absolute())


def _configured_paths(config: dict) -> list[str]:
    context = config.get("context")
    values = context.get("external_files") if isinstance(context, dict) else None
    if not isinstance(values, list):
        return []
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


def _discover_candidates() -> list[tuple[str, str, tuple]]:
    candidates = []
    seen = set()
    for label, relative in _EXTERNAL_CONTEXT_FILE_CANDIDATES:
        path = Path.home() / relative
        try:
            if not path.is_file():
                continue
            key = _context_file_key(path)
        except (OSError, ValueError, RuntimeError):
            continue
        if key not in seen:
            seen.add(key)
            candidates.append((label, "~/" + relative, key))
    return candidates


def setup_external_context_files(config: dict) -> None:
    """Offer detected shared files while preserving custom entries and navigation choices."""
    from agent.external_context import _expanded_path
    from hermes_cli.setup import _DOCS_BASE, _info, _sub_dict, print_header, prompt_yes_no, save_config

    print_header("External Context Files")
    _info("Reuse shared instruction files before working-directory project context.",
          "Only enable files you trust. Changes apply to new sessions or a context-compression rebuild.",
          f"Guide: {_DOCS_BASE}/user-guide/configuration#external-context-files")
    current = _configured_paths(config)
    if current:
        _info("Currently configured:", *(f"  {path}" for path in current))
    candidates = _discover_candidates()
    if not candidates:
        _info("No known shared context files found. Configure paths later with:",
              "  hermes config set context.external_files '~/.codex/AGENTS.md'")
        return
    keys = []
    for value in current:
        try:
            keys.append(_context_file_key(_expanded_path(value)))
        except (OSError, ValueError, RuntimeError):
            keys.append(None)
    choices = {}
    for label, display_path, key in candidates:
        choices[key] = prompt_yes_no(
            f"Use {label} ({display_path}) as external context?", default=key in keys)
    # Commit only after all prompts finish. Escape leaves the section unchanged, and revisiting
    # it with Left still presents the saved choices instead of skipping newly configured files.
    paths = [value for value, key in zip(current, keys) if choices.get(key, True)]
    paths.extend(path for _, path, key in candidates if choices[key] and key not in keys)
    if paths != current:
        _sub_dict(config, "context")["external_files"] = paths
        save_config(config)
