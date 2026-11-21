"""Generic local mutators enter the shared catalog gate through this one adapter.

Callers pass the absolute paths their short filesystem effect touches. Paths in the
profile skills root become skills-root-relative targets for ``Projection.generic``,
which holds the mutation gate, rejects overlapping live provider claims and
reservations, takes the writer-hold chain and revalidates final paths. Endpoints under
a reserved dot root (``.archive``) can never carry a provider claim, so they are only
checked here for redirects. Paths outside the profile root still hold the gate.
"""
import os
import stat
from contextlib import contextmanager
from pathlib import Path

from hermes_cli.skill_catalog.projection import Projection
from hermes_cli.skill_catalog.validation import CatalogError

RESERVED_ROOTS = frozenset((".archive",))


def _split(root, paths):
    live, reserved = [], []
    for path in paths:
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(root):
            continue
        relative = path.relative_to(root).as_posix()
        if relative.split("/", 1)[0] in RESERVED_ROOTS:
            reserved.append(relative)
        else:
            live.append(relative)
    return live, reserved


def _check_reserved(root, relative):
    current = root
    for part in relative.split("/"):
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise CatalogError("ownership_conflict", "Catalog target is not a local skill path.") from None
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CatalogError("ownership_conflict", "Catalog target is not a local skill path.")


@contextmanager
def generic(*paths, deadline=None):
    """Hold profile admission for the caller's whole short effect; raises CatalogError."""
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    root = Path(os.path.abspath(home / "skills"))
    live, reserved = _split(root, paths)
    with Projection(home).generic(live, deadline=deadline):
        for relative in reserved:
            _check_reserved(root, relative)
        yield


def refusal(exc: CatalogError) -> str:
    return f"Skill catalog refused this change ({exc.code}): {exc.message}"
