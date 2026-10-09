"""Checkout seeding for the canonical remote model catalogue cache."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from models.catalog_manifest import validate_manifest
from models.catalog_runtime import reset_cache
from utils import atomic_json_write

logger = logging.getLogger(__name__)


def seed_cache_from_checkout(
    project_root: Path | str,
    cache_path: Path,
) -> bool:
    src = Path(project_root) / "website" / "static" / "api" / "model-catalog.json"
    try:
        data = json.loads(src.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.debug("model catalog seed from checkout skipped (%s): %s", src, exc)
        return False
    if not validate_manifest(data):
        logger.debug(
            "model catalog seed from checkout skipped: invalid manifest at %s",
            src,
        )
        return False
    try:
        atomic_json_write(cache_path, data)
    except OSError as exc:
        logger.debug("model catalog seed cache write failed (%s): %s", cache_path, exc)
        return False
    reset_cache(cache_path)
    return True


__all__ = ["seed_cache_from_checkout"]
