"""Environment manifest for project verification.

Ported from superagent-ai/grok-cli ``src/verify/environment.ts``.
The manifest lives at ``<project>/.hermes/environment.json`` and is the
user-editable source of truth: when present and valid it wins over fresh
static detection.
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
    """Load the saved recipe from the manifest, tolerating malformed files.

    Mirrors grok's ``loadVerifyEnvironment``: any read/parse/shape problem
    returns ``None`` rather than raising, so a corrupt manifest degrades to
    fresh detection instead of breaking ``hermes verify``.
    """
    path = manifest_path(root)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        manifest = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    # Accept both the wrapped {version, recipe} shape and a bare recipe.
    recipe_raw = manifest.get("recipe", manifest)
    return Recipe.from_dict(recipe_raw)


def save_manifest(root: Path, recipe: Recipe) -> Path:
    """Persist ``recipe`` as the project's verify manifest.

    Writes the versioned wrapper shape (grok's ``saveVerifyEnvironment``
    equivalent) and returns the manifest path.
    """
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MANIFEST_VERSION,
        "recipe": recipe.to_dict(),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_or_detect(root: Path) -> tuple[Recipe | None, str]:
    """Return (recipe, source) where source is 'manifest' or 'detected'.

    A saved manifest wins over fresh detection, matching grok-cli's
    behavior where ``.grok/environment.json`` is the source of truth.
    """
    saved = load_manifest(root)
    if saved is not None:
        return saved, "manifest"
    return detect_recipe(root), "detected"
