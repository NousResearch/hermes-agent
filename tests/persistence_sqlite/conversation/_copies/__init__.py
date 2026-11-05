"""What the conversation backend's ``_store`` modules copy from the legacy store.

One spec module per backend module (``<target>.py`` beside this file) defines ``COPY``, one ``Copy`` or a tuple of them when the module copies from several legacy modules: the legacy module, the module-level functions and assignments it copies, the legacy class whose members it copies as module-level functions and assignments, and the explicit substitutions. ``load_specs`` reads every spec.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_SPEC_DIR = Path(__file__).parent

# Legacy one-line methods that call the module function of the same name with ``self``: they have no copy.
DELEGATES = {
    "update_conversation_model_config_value": "model_config",
    "session_ids_with_model_config_key": "model_config",
}


@dataclass(frozen=True)
class Copy:
    """One backend module: ``names`` are module-level definitions copied in legacy order, ``members`` the members of
    ``legacy_class`` copied after them as module-level definitions, in legacy order; ``substitutions`` maps a copied
    name to its ``(old, new)`` edits."""

    legacy: str
    doc: str = ""
    names: tuple[str, ...] = ()
    legacy_class: Optional[str] = None
    members: tuple[str, ...] = ()
    substitutions: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)


def load_specs() -> dict[str, tuple[Copy, ...]]:
    """Every spec beside this module, keyed by its backend module name; the first ``Copy`` holds the module's doc."""
    specs = {}
    for path in sorted(_SPEC_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{path.stem}")
        specs[path.stem] = module.COPY if isinstance(module.COPY, tuple) else (module.COPY,)
    return specs
