"""Image-authored deployment provenance for immutable Hermes runtimes.

The image bakes ``/etc/hermes/image-provenance.json`` outside ``$HERMES_HOME`` and the checkout, so a
bind-mounted checkout cannot hide the build fact and env/config cannot forge it. Absence preserves every
source/package install path. Presence fails closed: an unreadable, non-regular, or malformed marker still
means image-managed — an integrity defect, never permission to mutate the image in place."""

from __future__ import annotations

import json
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

IMAGE_PROVENANCE_PATH = Path("/etc/hermes/image-provenance.json")
IMAGE_PROVENANCE_SCHEMA = 1


@dataclass(frozen=True)
class ImageProvenance:
    """Validated provenance, or a fail-closed description of an invalid one."""

    schema: int
    deployment_kind: str
    manager: str
    image: Optional[str] = None
    version: Optional[str] = None
    revision: Optional[str] = None
    marker_path: str = ""
    valid: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _invalid(path: Path, reason: str) -> ImageProvenance:
    return ImageProvenance(IMAGE_PROVENANCE_SCHEMA, "image", "unknown", marker_path=str(path), valid=False, error=reason)


def _optional_string(payload: dict, name: str) -> Optional[str]:
    value = payload.get(name)
    if value is not None and not isinstance(value, str):
        raise TypeError(name)
    return (value.strip() or None) if value is not None else None


def read_image_provenance(marker_path: Optional[Path] = None) -> Optional[ImageProvenance]:
    """Read the baked marker without consulting environment or config. Never raises.
    ``marker_path`` is a dependency-injection seam for tests and alternate image builders."""
    path = IMAGE_PROVENANCE_PATH
    try:
        path = Path(marker_path) if marker_path is not None else path
    except BaseException as exc:
        return _invalid(path, f"marker_presence_unreadable:{type(exc).__name__}")
    try:
        marker_stat = path.lstat()
    except FileNotFoundError:
        return None
    except BaseException as exc:  # permission errors and other lookup failures do not prove absence
        return _invalid(path, f"marker_presence_unreadable:{type(exc).__name__}")
    if not stat.S_ISREG(marker_stat.st_mode):
        return _invalid(path, "marker_not_regular_file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # may vanish between lstat/read; it was observed present, so fail closed
        return _invalid(path, f"marker_unreadable:{type(exc).__name__}")
    if not isinstance(payload, dict):
        return _invalid(path, "marker_not_object")
    schema = payload.get("schema")
    # ``bool`` subclasses ``int``: schema ``true`` must not be accepted as schema 1.
    if type(schema) is not int or schema != IMAGE_PROVENANCE_SCHEMA:
        return _invalid(path, "unsupported_marker_schema")
    if payload.get("deployment_kind") != "image":
        return _invalid(path, "invalid_deployment_kind")
    manager = payload.get("manager")
    if not isinstance(manager, str) or not manager.strip():
        return _invalid(path, "missing_manager")
    try:
        optional = {name: _optional_string(payload, name) for name in ("image", "version", "revision")}
    except TypeError as exc:
        return _invalid(path, f"invalid_{exc.args[0]}")
    return ImageProvenance(IMAGE_PROVENANCE_SCHEMA, "image", manager.strip(), marker_path=str(path), **optional)
