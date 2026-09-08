"""Validated, durable definitions for explicit GitHub pull-request stacks."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_STACK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _require(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _branch(value: object, field: str) -> str:
    value = _require(value, field)
    if (
        not _BRANCH.fullmatch(value)
        or value in {".", ".."}
        or ".." in value
        or value.endswith("/")
        or "//" in value
    ):
        raise ValueError(f"{field} is not a safe branch name")
    return value


@dataclass(frozen=True, slots=True)
class StackEntry:
    branch: str
    base_branch: str
    title: str
    body: str
    pr_number: int | None = None
    head_sha: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch", _branch(self.branch, "branch"))
        object.__setattr__(self, "base_branch", _branch(self.base_branch, "base_branch"))
        object.__setattr__(self, "title", _require(self.title, "title"))
        if not isinstance(self.body, str):
            raise ValueError("body must be a string")
        if self.pr_number is not None and (
            isinstance(self.pr_number, bool) or not isinstance(self.pr_number, int) or self.pr_number < 1
        ):
            raise ValueError("pr_number must be a positive integer or null")
        if self.head_sha is not None and (
            not isinstance(self.head_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", self.head_sha)
        ):
            raise ValueError("head_sha must be a full Git object ID or null")


@dataclass(frozen=True, slots=True)
class StackManifest:
    repository: str
    stack_id: str
    base_branch: str
    entries: tuple[StackEntry, ...]
    updated_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not _REPOSITORY.fullmatch(self.repository):
            raise ValueError("repository must be an owner/repository name")
        if not isinstance(self.stack_id, str) or not _STACK_ID.fullmatch(self.stack_id):
            raise ValueError("stack_id is not safe")
        object.__setattr__(self, "base_branch", _branch(self.base_branch, "base_branch"))
        if not self.entries:
            raise ValueError("entries must not be empty")
        if not isinstance(self.updated_at, datetime) or self.updated_at.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware")
        validate_stack_entries(self.base_branch, self.entries)


def validate_stack_entries(base_branch: str, entries: tuple[StackEntry, ...]) -> None:
    _branch(base_branch, "base_branch")
    branches = {entry.branch for entry in entries}
    if len(branches) != len(entries):
        raise ValueError("stack entries must use unique branches")
    if base_branch in branches:
        raise ValueError("base branch cannot also be a stack entry")
    for entry in entries:
        if entry.base_branch != base_branch and entry.base_branch not in branches:
            raise ValueError(f"entry {entry.branch} names an unknown parent branch")
    for entry in entries:
        seen: set[str] = set()
        parent = entry.base_branch
        while parent != base_branch:
            if parent in seen:
                raise ValueError("stack entries contain a cycle")
            seen.add(parent)
            parent = next(item.base_branch for item in entries if item.branch == parent)


class StackStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, repository: str, stack_id: str) -> Path:
        safe_repo = repository.replace("/", "__")
        if not _REPOSITORY.fullmatch(repository) or not _STACK_ID.fullmatch(stack_id):
            raise ValueError("invalid repository or stack_id")
        return self.root / safe_repo / f"{stack_id}.json"

    def load(self, repository: str, stack_id: str) -> StackManifest:
        path = self.path(repository, stack_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return StackManifest(
                repository=raw["repository"],
                stack_id=raw["stack_id"],
                base_branch=raw["base_branch"],
                entries=tuple(StackEntry(**entry) for entry in raw["entries"]),
                updated_at=datetime.fromisoformat(raw["updated_at"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid stack manifest: {path}") from error

    def save(self, manifest: StackManifest) -> Path:
        path = self.path(manifest.repository, manifest.stack_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "repository": manifest.repository,
            "stack_id": manifest.stack_id,
            "base_branch": manifest.base_branch,
            "entries": [asdict(entry) for entry in manifest.entries],
            "updated_at": manifest.updated_at.astimezone(UTC).isoformat(),
        }
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return path
