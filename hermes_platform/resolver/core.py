"""Result and context types shared by every resolver, plus passive command location.

`locate_command` reads only file metadata. It never opens a file, spawns, or connects.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Literal, TypeVar

Kind = Literal["explicit_path", "path_executable", "known_path", "package_runner", "missing"]

T = TypeVar("T")


class _Absent:
    __slots__ = ()

    def __repr__(self) -> str:
        return "ABSENT"


ABSENT = _Absent()


class CheckState(Enum):
    NOT_CHECKED = "not_checked"
    PRESENT = "present"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True)
class Observation(Generic[T]):
    """One checked fact. Consumers branch on `state`; `detail` is never parsed."""

    state: CheckState
    value: T | None = None
    detail: str = ""

    @classmethod
    def not_checked(cls) -> Observation[T]:
        return cls(CheckState.NOT_CHECKED)


@dataclass(frozen=True)
class Candidate:
    value: str
    source: str
    present: bool


@dataclass(frozen=True)
class LookupContext:
    """`path`: ABSENT = caller did not say (ambient), None = ambient, "" = search nothing.

    `pathext`: None = ambient Windows extensions, "" = do not add extensions.
    """

    path: str | None | _Absent = ABSENT
    pathext: str | None = None

    def effective_path(self) -> str | None:
        if isinstance(self.path, str):
            return self.path
        return None


@dataclass(frozen=True)
class Resolution:
    kind: Kind
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)

    @property
    def command(self) -> tuple[str, ...]:
        first = next((c for c in self.candidates if c.present), None)
        return (first.value,) if first else ()

    @property
    def source(self) -> str:
        first = next((c for c in self.candidates if c.present), None)
        return first.source if first else "none"

    @property
    def present(self) -> tuple[Candidate, ...]:
        return tuple(c for c in self.candidates if c.present)

    @property
    def found(self) -> bool:
        return self.kind != "missing"


def _which(name: str, ctx: LookupContext) -> str | None:
    if ctx.path == "":
        return None
    if ctx.pathext is not None and os.name == "nt":
        # shutil.which always reads ambient PATHEXT, even when passed a suffixed name.
        path = ctx.effective_path()
        if path is None:
            path = os.environ.get("PATH", os.defpath)
        if not path:
            return None
        for directory in path.split(os.pathsep):
            base = os.path.join(directory, name)
            for candidate in _command_variants(base, ctx.pathext):
                if _is_executable_file(candidate):
                    return candidate
        return None
    return shutil.which(name, path=ctx.effective_path())


def _is_executable_file(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def locate_command(name: str, ctx: LookupContext | None = None, *, known_dirs: tuple[str, ...] = ()) -> Resolution:
    """Find `name` on PATH, then in `known_dirs`, recording every candidate in probe order."""
    ctx = ctx or LookupContext()
    name = name.strip()
    if os.sep in name or (os.altsep and os.altsep in name):
        expanded = _expand(name)
        # A relative explicit path would resolve against the ambient cwd; only absolute paths are trusted.
        present = os.path.isabs(expanded) and _is_executable_file(expanded)
        cand = Candidate(expanded, "explicit", present)
        return Resolution("explicit_path" if present else "missing", (cand,))

    candidates: list[Candidate] = []
    hit = _which(name, ctx)
    candidates.append(Candidate(hit or name, "PATH", hit is not None))
    for d in known_dirs:
        base = os.path.join(_expand(d), name)
        found = next((v for v in _command_variants(base, ctx.pathext) if _is_executable_file(v)), None)
        candidates.append(Candidate(found or base, f"known_dir:{d}", found is not None))

    kind: Kind = "missing"
    for c in candidates:
        if c.present:
            kind = "path_executable" if c.source == "PATH" else "known_path"
            break
    return Resolution(kind, tuple(candidates))


def _command_variants(base: str, pathext: str | None = None) -> tuple[str, ...]:
    """Try the literal name, then Windows suffixes from the caller or ambient PATHEXT."""
    if os.name != "nt":
        return (base,)
    if pathext is None:
        pathext = os.environ.get("PATHEXT", ".EXE;.CMD;.BAT")
    exts = [e for e in pathext.split(os.pathsep) if e]
    return (base, *[base + e for e in exts])
