"""Image-authored deployment provenance for immutable Hermes runtimes.

The published image bakes ``/etc/hermes/image-provenance.json`` outside both
``$HERMES_HOME`` and the mutable checkout.  A bind-mounted checkout (including
``.git``) therefore cannot hide the build fact, and environment or config
values cannot forge it.

Absence preserves every pre-existing source/package install path.  Presence
fails closed: an unreadable, non-regular, or malformed marker still means the
runtime is image-managed; it is an integrity defect, never permission to
mutate the image in place.
"""

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
    image: Optional[str]
    version: Optional[str]
    revision: Optional[str]
    marker_path: str
    valid: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _invalid(path: Path, reason: str) -> ImageProvenance:
    return ImageProvenance(
        schema=IMAGE_PROVENANCE_SCHEMA,
        deployment_kind="image",
        manager="unknown",
        image=None,
        version=None,
        revision=None,
        marker_path=str(path),
        valid=False,
        error=reason,
    )


def read_image_provenance(
    marker_path: Optional[Path] = None,
) -> Optional[ImageProvenance]:
    """Read the baked marker without consulting environment or config.

    ``None`` has one precise meaning: ``lstat`` proved that no marker exists.
    Every other filesystem or validation failure returns an invalid
    :class:`ImageProvenance`, so callers refuse image mutation closed.  In
    particular, ``lstat`` makes a dangling symlink visibly *present* and the
    regular-file check rejects symlinks, directories, and device nodes.

    ``marker_path`` is a dependency-injection seam for tests and alternate
    image builders.  Normal callers always use the image-owned absolute path.
    This function never raises.
    """

    path = IMAGE_PROVENANCE_PATH
    try:
        path = Path(marker_path) if marker_path is not None else path
    except BaseException as exc:
        return _invalid(path, f"marker_presence_unreadable:{type(exc).__name__}")

    try:
        marker_stat = path.lstat()
    except FileNotFoundError:
        return None
    except BaseException as exc:
        # Permission errors and other lookup failures do not prove absence.
        return _invalid(path, f"marker_presence_unreadable:{type(exc).__name__}")

    if not stat.S_ISREG(marker_stat.st_mode):
        return _invalid(path, "marker_not_regular_file")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        # The file may disappear between lstat/read; it was nevertheless
        # observed present, so the decision remains fail-closed.
        return _invalid(path, f"marker_unreadable:{type(exc).__name__}")

    if not isinstance(payload, dict):
        return _invalid(path, "marker_not_object")

    schema = payload.get("schema")
    # ``bool`` is an ``int`` subclass in Python.  Schema ``true`` must not be
    # accepted as schema 1, hence the exact type check.
    if type(schema) is not int or schema != IMAGE_PROVENANCE_SCHEMA:
        return _invalid(path, "unsupported_marker_schema")
    if payload.get("deployment_kind") != "image":
        return _invalid(path, "invalid_deployment_kind")

    manager = payload.get("manager")
    if not isinstance(manager, str) or not manager.strip():
        return _invalid(path, "missing_manager")

    def _optional_string(name: str) -> Optional[str]:
        value = payload.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(name)
        value = value.strip()
        return value or None

    try:
        image = _optional_string("image")
        version = _optional_string("version")
        revision = _optional_string("revision")
    except TypeError as exc:
        return _invalid(path, f"invalid_{exc.args[0]}")

    return ImageProvenance(
        schema=IMAGE_PROVENANCE_SCHEMA,
        deployment_kind="image",
        manager=manager.strip(),
        image=image,
        version=version,
        revision=revision,
        marker_path=str(path),
    )
