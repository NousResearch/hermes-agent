"""Environment manifest for project verification.

Ported from superagent-ai/grok-cli ``src/verify/environment.ts``. The manifest
at ``<project>/.hermes/environment.json`` is the user-editable source of truth:
when present and valid it wins over fresh static detection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.verify.recipes import Recipe, detect_recipe

MANIFEST_VERSION = 1
_MANIFEST_RELPATH = Path(".hermes") / "environment.json"


def manifest_path(root: Path) -> Path:
    """Path of the verify manifest for the project at ``root``."""
    return Path(root) / _MANIFEST_RELPATH


def load_manifest(root: Path) -> Recipe | None:
    """Load the saved recipe; any read/parse/shape problem returns ``None`` so a
    corrupt manifest degrades to fresh detection. Accepts the wrapped
    ``{version, recipe}`` shape and a bare recipe."""
    try:
        manifest = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError includes JSONDecodeError
        return None
    return Recipe.from_dict(manifest.get("recipe", manifest)) if isinstance(manifest, dict) else None


def save_manifest(root: Path, recipe: Recipe) -> Path:
    """Persist ``recipe`` in the versioned wrapper shape; returns the manifest path."""
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": MANIFEST_VERSION, "recipe": recipe.to_dict(), "updatedAt": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_or_detect(root: Path) -> tuple[Recipe | None, str]:
    """Return ``(recipe, source)``; a saved manifest ('manifest') wins over 'detected'."""
    saved = load_manifest(root)
    return (saved, "manifest") if saved is not None else (detect_recipe(root), "detected")
