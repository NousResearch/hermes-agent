"""Deterministic ceiling/request scope resolution for projected workspaces.

This module deliberately returns repository-relative, concrete paths.  A caller's
request can never widen a role ceiling; when a repository is supplied, glob
patterns are expanded from the frozen git tree rather than the host filesystem.
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any, Iterable


class ScopeError(ValueError):
    """Raised when a scope is malformed or cannot be safely materialized."""


@dataclass(frozen=True)
class RoleCeiling:
    read: frozenset[str]
    write: frozenset[str] = frozenset()
    hidden: frozenset[str] = frozenset()
    invariant_exclude: frozenset[str] = frozenset({".git/**", ".ssh/**", ".aws/**", "**/.env", ".hermes/**"})

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RoleCeiling":
        return cls(
            read=frozenset(_patterns(value.get("read"))),
            write=frozenset(_patterns(value.get("write"))),
            hidden=frozenset(_patterns(value.get("hidden"))),
            invariant_exclude=frozenset(_patterns(value.get("invariant_exclude"))) or cls.invariant_exclude,
        )


@dataclass(frozen=True)
class Scope:
    read: frozenset[str]
    write: frozenset[str]

    @property
    def paths(self) -> frozenset[str]:
        return self.read | self.write


def _patterns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ScopeError("scope patterns must be strings")
    return [_safe_path(item) for item in value]


def _safe_path(value: Any) -> str:
    path = str(value).replace("\\", "/").strip()
    if not path or path.startswith("/") or "\x00" in path:
        raise ScopeError(f"invalid scope path: {value!r}")
    parts = PurePosixPath(path).parts
    if ".." in parts:
        raise ScopeError(f"scope path traversal is not allowed: {value!r}")
    return path.removeprefix("./")


def _matches(pattern: str, path: str) -> bool:
    # fnmatch's ** is sufficient for repository-relative matching, but a bare
    # directory name should also mean that directory and its descendants.
    return fnmatchcase(path, pattern) or fnmatchcase(path, pattern.rstrip("/") + "/**")


def _tree_paths(repo: Any, base_sha: str) -> tuple[set[str], set[str]]:
    import subprocess
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", base_sha],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise ScopeError(f"cannot inspect git tree {base_sha!r}: {result.stderr.decode(errors='replace').strip()}")
    files: set[str] = set()
    unsafe: set[str] = set()
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        mode = meta.split(b" ", 1)[0]
        path = raw_path.decode("utf-8", "surrogateescape")
        files.add(path)
        if mode in {b"120000", b"160000"}:
            unsafe.add(path)
    return files, unsafe


def resolve_scope(task: Any, ceiling: RoleCeiling | dict[str, Any], *, repo: Any = None, base_sha: str | None = None) -> Scope:
    """Return the concrete ``ceiling ∩ request`` scope.

    ``task.scope_paths`` is the explicit request.  An unset request means the
    entire ceiling.  With ``repo`` and ``base_sha`` all patterns are expanded
    against the pinned tree; without them, literal requests are supported and
    wildcard requests are retained only when they are provably within a
    ceiling pattern.
    """
    if isinstance(ceiling, dict):
        ceiling = RoleCeiling.from_mapping(ceiling)
    if not isinstance(ceiling, RoleCeiling):
        raise ScopeError("ceiling must be a RoleCeiling or mapping")
    requested = _patterns(getattr(task, "scope_paths", None))
    if not requested:
        requested = list(ceiling.read | ceiling.write)
    blocked = set(ceiling.hidden) | set(ceiling.invariant_exclude)

    if repo is not None:
        if not base_sha:
            raise ScopeError("base_sha is required when resolving against a repository")
        candidates, unsafe = _tree_paths(repo, base_sha)
        if unsafe:
            # Any unsafe entry granted by the request/ceiling is rejected, not
            # silently omitted: callers must amend the manifest explicitly.
            if any(_matches(pat, path) for pat in requested for path in unsafe):
                raise ScopeError("scope includes a symlink or submodule")
        allowed = {p for p in candidates if any(_matches(c, p) for c in ceiling.read)}
        visible = {p for p in allowed if not any(_matches(h, p) for h in blocked)}
        granted = {p for p in visible if any(_matches(r, p) for r in requested)}
        writable = {p for p in granted if any(_matches(w, p) for w in ceiling.write)}
        return Scope(frozenset(granted), frozenset(writable))

    def clipped(patterns: Iterable[str]) -> set[str]:
        out = set()
        for req in requested:
            for cap in patterns:
                if _matches(cap, req):
                    # req is within cap: the request is the narrower pattern.
                    narrower = req
                elif _matches(req, cap):
                    # cap is within req: never widen past the role ceiling.
                    narrower = cap
                else:
                    continue
                if not any(_matches(h, narrower) for h in blocked):
                    out.add(narrower)
        return out
    read = clipped(ceiling.read)
    write = clipped(ceiling.write) & read
    return Scope(frozenset(read), frozenset(write))
