"""What a venv sync or currency probe should take as its plugin members.

Each request names exactly one source, so conflicting member inputs cannot be
expressed. Stdlib only: the client, the worker and the engine all import it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Members:
    """Build exactly these members and skip config discovery.

    A mapping keys each member's installed identity to the tree that supplies it."""
    dirs: Sequence[Path] | Mapping[Path, Path]


@dataclass(frozen=True)
class Candidates:
    """Discover the enabled members and add these candidate directories."""
    dirs: Sequence[Path]


@dataclass(frozen=True)
class Selection:
    """Publish a proposed plugin enablement with the members it implies."""
    data: Mapping


@dataclass(frozen=True)
class StagedUpdate:
    """Publish a staged plugin tree over its install."""
    data: Mapping


PluginInput = Members | Candidates | Selection | StagedUpdate


def encode(plugins: PluginInput | None) -> dict | None:
    """JSON form for the worker request; paths become absolute strings."""
    if plugins is None:
        return None
    if isinstance(plugins, Members):
        if isinstance(plugins.dirs, Mapping):
            return {"kind": "members", "sources": [[str(Path(key).absolute()), str(Path(source).absolute())]
                                                    for key, source in plugins.dirs.items()]}
        return {"kind": "members", "paths": [str(Path(path).absolute()) for path in plugins.dirs]}
    if isinstance(plugins, Candidates):
        return {"kind": "candidates", "paths": [str(Path(path).absolute()) for path in plugins.dirs]}
    kind = "selection" if isinstance(plugins, Selection) else "staged"
    return {"kind": kind, "data": dict(plugins.data)}


def decode(value: dict | None) -> PluginInput | None:
    if value is None:
        return None
    kind = value["kind"]
    if kind == "members":
        if "sources" in value:
            return Members({Path(identity): Path(source) for identity, source in value["sources"]})
        return Members([Path(path) for path in value["paths"]])
    if kind == "candidates":
        return Candidates([Path(path) for path in value["paths"]])
    if kind == "selection":
        return Selection(value["data"])
    if kind == "staged":
        return StagedUpdate(value["data"])
    raise ValueError(f"unknown plugin input kind: {kind!r}")
