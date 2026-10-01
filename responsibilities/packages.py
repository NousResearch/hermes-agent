"""Responsibility packages; employee contracts on native Hermes."""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import hermes_yaml as yaml
from responsibilities.filesystem import (WORKSPACE_ARCHIVE_DIRNAME, _read_regular_file_fd)

from responsibilities.common import get_responsibilities_root

from responsibilities.common import (
    ARCHIVE_DIRNAME,
    ARCHIVE_FILE_BYTE_CEILING,
    CONTEXT_DIRNAME,
    CONTEXT_FILE_BYTE_CEILING,
    MAX_ARCHIVE_FILES,
    MAX_ARCHIVE_FILE_CHARS,
    MAX_CONTEXT_FILE_CHARS,
    MAX_RESPONSIBILITY_DESCRIPTION_LENGTH,
    MAX_RESPONSIBILITY_FRONTMATTER_BYTES,
    MAX_RESPONSIBILITY_MD_CHARS,
    MAX_STATE_FILES,
    MAX_STATE_FILE_CHARS,
    MAX_STATE_ROOT_CHARS,
    REFERENCES_DIRNAME,
    RESPONSIBILITY_MD_BYTE_CEILING,
    RESPONSIBILITY_MD_CHAR_CEILING,
    ResponsibilityFilesystemError,
    SCHEDULES_DIRNAME,
    SCRIPTS_DIRNAME,
    STATE_DIRNAME,
    STATE_FILE_BYTE_CEILING,
    STATE_ROOT_BYTE_CEILING,
    WEBHOOKS_DIRNAME,
    _ALLOWED_FRONTMATTER,
    _FRONTMATTER_END_RE,
    _LIFECYCLE_VALUES,
    _MARKDOWN_TREE_DIRNAMES,
    _ROSTER_DESCRIPTION_LENGTH,
    _SCRIPT_SUFFIXES,
    _STRAY_FILES_LIMIT,
    _STRAY_SCAN_MAX_ENTRIES,
    _has_done_when_heading,
    validate_responsibility_name
)
from responsibilities.declarations import (
    _scan_schedule_declarations,
    _scan_webhook_declarations
)

def _scan_package_scripts(package_fd: int) -> tuple[str, ...]:
    """Top-level scripts/ filenames inside an open package.

    Guard-script existence is checked at reconcile against this listing;
    a listing failure reads as empty so a broken scripts/ directory never
    aborts the package scan — the affected declaration simply reports its
    script missing.
    """

    try:
        info = os.stat("scripts", dir_fd=package_fd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            return ()
        scripts_fd = os.open("scripts", os.O_RDONLY, dir_fd=package_fd)
    except OSError:
        return ()
    try:
        return tuple(
            sorted(
                entry.name
                for entry in os.scandir(scripts_fd)
                if entry.is_file(follow_symlinks=False)
            )
        )
    except OSError:
        return ()
    finally:
        os.close(scripts_fd)

def _scan_stray_files(package_fd: int) -> tuple[str, ...]:
    """Package paths a governed write could never have produced.

    The path and markdown rules are tool-enforced; terminal writes bypass
    them, so the index sweep reports out-of-contract paths for the model
    to clean up. Listing failures read as empty — an unreadable package
    is already a package error — the listing caps at
    ``_STRAY_FILES_LIMIT`` entries, and the walk inspects at most
    ``_STRAY_SCAN_MAX_ENTRIES`` entries (symlinks are never followed, so
    depth is bounded by the real tree; an absurdly large tree
    under-reports rather than false-flagging).
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    strays: list[str] = []
    inspected = [0]

    def full(path: str) -> bool:
        strays.append(path)
        return len(strays) >= _STRAY_FILES_LIMIT

    def exhausted() -> bool:
        inspected[0] += 1
        return inspected[0] > _STRAY_SCAN_MAX_ENTRIES

    def walk_markdown_tree(root_fd: int, root_prefix: str) -> bool:
        # Iterative depth-first walk: terminal writes can nest directories
        # deeper than Python's recursion limit. One open descriptor per
        # depth level; a descriptor that cannot be opened (including fd
        # exhaustion on an absurd tree) skips that subtree instead of
        # failing the scan.
        try:
            first_fd = os.dup(root_fd)
        except OSError:
            return False
        try:
            first_entries = sorted(
                os.scandir(first_fd), key=lambda item: item.name
            )
        except OSError:
            os.close(first_fd)
            return False
        frames: list[list] = [[first_fd, root_prefix, first_entries, 0]]
        try:
            while frames:
                frame = frames[-1]
                directory_fd, prefix, entries, index = frame
                if index >= len(entries):
                    frames.pop()
                    os.close(directory_fd)
                    continue
                frame[3] = index + 1
                entry = entries[index]
                if exhausted():
                    return True
                path = f"{prefix}/{entry.name}"
                try:
                    if entry.is_dir(follow_symlinks=False):
                        try:
                            child_fd = os.open(
                                entry.name, flags, dir_fd=directory_fd
                            )
                        except OSError:
                            continue
                        try:
                            child_entries = sorted(
                                os.scandir(child_fd),
                                key=lambda item: item.name,
                            )
                        except OSError:
                            os.close(child_fd)
                            continue
                        frames.append([child_fd, path, child_entries, 0])
                    elif entry.is_file(follow_symlinks=False):
                        if not entry.name.endswith(".md") and full(path):
                            return True
                    elif full(path):
                        return True
                except OSError:
                    continue
            return False
        finally:
            for frame in frames:
                os.close(frame[0])

    def check_flat_dir(dirname: str, allowed_suffixes: tuple[str, ...]) -> bool:
        try:
            directory_fd = os.open(dirname, flags, dir_fd=package_fd)
        except OSError:
            return False
        try:
            entries = sorted(os.scandir(directory_fd), key=lambda item: item.name)
            for entry in entries:
                if exhausted():
                    return True
                path = f"{dirname}/{entry.name}"
                try:
                    if not entry.is_file(follow_symlinks=False):
                        if full(path):
                            return True
                    elif not entry.name.endswith(allowed_suffixes) and full(path):
                        return True
                except OSError:
                    continue
        except OSError:
            return False
        finally:
            os.close(directory_fd)
        return False

    try:
        root_entries = sorted(os.scandir(package_fd), key=lambda item: item.name)
    except OSError:
        return ()
    for entry in root_entries:
        if exhausted():
            break
        try:
            if entry.is_dir(follow_symlinks=False):
                if entry.name in _MARKDOWN_TREE_DIRNAMES:
                    try:
                        child_fd = os.open(entry.name, flags, dir_fd=package_fd)
                    except OSError:
                        continue
                    try:
                        if walk_markdown_tree(child_fd, entry.name):
                            break
                    finally:
                        os.close(child_fd)
                elif entry.name in {SCHEDULES_DIRNAME, WEBHOOKS_DIRNAME}:
                    if check_flat_dir(entry.name, (".yaml",)):
                        break
                elif entry.name == SCRIPTS_DIRNAME:
                    if check_flat_dir(entry.name, _SCRIPT_SUFFIXES):
                        break
                elif full(entry.name):
                    break
            elif entry.is_file(follow_symlinks=False):
                if entry.name not in {"RESPONSIBILITY.md", "STATE.md"} and full(
                    entry.name
                ):
                    break
            elif full(entry.name):
                break
        except OSError:
            continue
    return tuple(strays)

def normalize_responsibility_reference(path: str) -> str:
    """Normalize one root-relative name or matching absolute workspace path."""

    if not isinstance(path, str) or not path.strip() or "\\" in path:
        raise ResponsibilityFilesystemError("responsibility path is required")
    raw = path.strip()
    candidate = PurePosixPath(raw)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(get_responsibilities_root())
        except ValueError as exc:
            raise ResponsibilityFilesystemError(
                "responsibility path must be beneath {profile_home}/responsibilities"
            ) from exc
    else:
        relative = candidate
    if len(relative.parts) != 1 or relative.parts[0] in {"", ".", ".."}:
        raise ResponsibilityFilesystemError(
            "responsibility path must name one direct child of "
            "{profile_home}/responsibilities"
        )
    name = relative.parts[0]
    error = validate_responsibility_name(name)
    if error:
        raise ResponsibilityFilesystemError(error)
    return name

def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({
            str(key): _freeze(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value

def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value

@dataclass(frozen=True, slots=True)
class ResponsibilityIndexEntry:
    name: str
    description: str
    relative_path: str
    frontmatter_extract: Mapping[str, Any]
    content_hash: str
    schedules: Mapping[str, Any] = field(default_factory=dict)
    schedule_errors: Mapping[str, str] = field(default_factory=dict)
    webhooks: Mapping[str, Any] = field(default_factory=dict)
    webhook_errors: Mapping[str, str] = field(default_factory=dict)
    scripts: tuple[str, ...] = ()
    stray_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        error = validate_responsibility_name(self.name)
        if error:
            raise ValueError(error)
        if self.relative_path != self.name:
            raise ValueError("responsibility relative path must equal its name")
        if (
            not self.description
            or len(self.description) > MAX_RESPONSIBILITY_DESCRIPTION_LENGTH
        ):
            raise ValueError("invalid responsibility description")
        object.__setattr__(
            self, "frontmatter_extract", _freeze(dict(self.frontmatter_extract))
        )
        object.__setattr__(self, "schedules", _freeze(dict(self.schedules)))
        object.__setattr__(self, "schedule_errors", _freeze(dict(self.schedule_errors)))
        object.__setattr__(self, "webhooks", _freeze(dict(self.webhooks)))
        object.__setattr__(self, "webhook_errors", _freeze(dict(self.webhook_errors)))
        object.__setattr__(
            self, "scripts", tuple(str(name) for name in self.scripts)
        )
        object.__setattr__(
            self, "stray_files", tuple(str(path) for path in self.stray_files)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "relative_path": self.relative_path,
            "frontmatter_extract": _thaw(self.frontmatter_extract),
            "content_hash": self.content_hash,
            "schedules": _thaw(self.schedules),
            "schedule_errors": _thaw(self.schedule_errors),
            "webhooks": _thaw(self.webhooks),
            "webhook_errors": _thaw(self.webhook_errors),
            "scripts": list(self.scripts),
            "stray_files": list(self.stray_files),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResponsibilityIndexEntry":
        frontmatter = raw.get("frontmatter_extract")
        if not isinstance(frontmatter, Mapping):
            raise ValueError("frontmatter_extract must be an object")
        schedules = raw.get("schedules") or {}
        schedule_errors = raw.get("schedule_errors") or {}
        webhooks = raw.get("webhooks") or {}
        webhook_errors = raw.get("webhook_errors") or {}
        if (
            not isinstance(schedules, Mapping)
            or not isinstance(schedule_errors, Mapping)
            or not isinstance(webhooks, Mapping)
            or not isinstance(webhook_errors, Mapping)
        ):
            raise ValueError("declarations and declaration errors must be objects")
        return cls(
            name=str(raw.get("name") or ""),
            description=str(raw.get("description") or ""),
            relative_path=str(raw.get("relative_path") or ""),
            frontmatter_extract=dict(frontmatter),
            content_hash=str(raw.get("content_hash") or ""),
            schedules=dict(schedules),
            schedule_errors={
                str(key): str(value) for key, value in schedule_errors.items()
            },
            webhooks=dict(webhooks),
            webhook_errors={
                str(key): str(value) for key, value in webhook_errors.items()
            },
            scripts=tuple(
                str(name) for name in (raw.get("scripts") or ())
            ),
            stray_files=tuple(
                str(path) for path in (raw.get("stray_files") or ())
            ),
        )

@dataclass(frozen=True, slots=True)
class ResponsibilityScanResult:
    """One whole-root scan: valid entries plus packages that exist but failed.

    ``package_errors`` distinguishes "the folder is there but unreadable or
    malformed" (reconciliation freezes its Schedule rows) from a folder that
    is affirmatively absent (reconciliation retires its rows).
    """

    entries: tuple[ResponsibilityIndexEntry, ...]
    package_errors: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "package_errors", _freeze(dict(self.package_errors))
        )

def _parse_document(raw: bytes) -> tuple[str, dict[str, Any], str, str] | None:
    if len(raw) > RESPONSIBILITY_MD_BYTE_CEILING or not raw.startswith(b"---"):
        return None
    # The byte ceiling only guards read plumbing; the character ceiling is
    # the real parse bound and must hold for terminal-written documents
    # too, or an arbitrarily large charter stays indexed and injected into
    # run prompts. It is deliberately the pre-budget value, not the write
    # budget: charters written before the budget existed must stay indexed
    # (read-lenient) until their next write consolidates them.
    # Bytes ≤ ceiling implies chars ≤ ceiling, so decode only past it.
    if (
        len(raw) > RESPONSIBILITY_MD_CHAR_CEILING
        and len(raw.decode("utf-8")) > RESPONSIBILITY_MD_CHAR_CEILING
    ):
        return None
    match = _FRONTMATTER_END_RE.search(
        raw,
        3,
        min(len(raw), MAX_RESPONSIBILITY_FRONTMATTER_BYTES + 3),
    )
    if match is None or match.end() > MAX_RESPONSIBILITY_FRONTMATTER_BYTES:
        return None
    body = raw[match.end() :]
    if not body.strip():
        return None
    parsed = yaml.safe_load(raw[3 : match.start()].decode("utf-8"))
    if not isinstance(parsed, dict) or not set(parsed).issubset(_ALLOWED_FRONTMATTER):
        return None
    if "name" not in parsed or not {"trigger", "description"} & set(parsed):
        return None
    name = parsed.get("name")
    description = parsed.get("trigger", parsed.get("description"))
    if not isinstance(name, str) or validate_responsibility_name(name) is not None:
        return None
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > MAX_RESPONSIBILITY_DESCRIPTION_LENGTH
    ):
        return None
    for key in ("author", "version"):
        if key in parsed and not isinstance(parsed[key], (str, int, float)):
            return None
    if "lifecycle" in parsed and (
        not isinstance(parsed["lifecycle"], str)
        or parsed["lifecycle"] not in _LIFECYCLE_VALUES
    ):
        return None
    extracted = {
        key: parsed[key]
        for key in ("name", "trigger", "description", "author", "version", "lifecycle")
        if key in parsed
    }
    return name, extracted, description.strip().strip("'\""), body.decode("utf-8")

def _validate_document(content: str, *, expected_name: str) -> str | None:
    try:
        raw = content.encode("utf-8")
    except UnicodeError:
        return "RESPONSIBILITY.md must be UTF-8 text."
    if len(content) > MAX_RESPONSIBILITY_MD_CHARS:
        return (
            f"This write would put RESPONSIBILITY.md at "
            f"{len(content):,}/{MAX_RESPONSIBILITY_MD_CHARS:,} chars. The "
            "charter is a router, it holds no method: move the area's "
            "craft to references/ files, how a service is operated to its "
            "connection manual, and current status to STATE.md, then "
            "retry, all in this turn."
        )
    try:
        parsed = _parse_document(raw)
    except (UnicodeError, yaml.YAMLError, ValueError):
        parsed = None
    if parsed is None or parsed[0] != expected_name:
        return (
            "RESPONSIBILITY.md must have valid frontmatter with name and "
            "trigger; its name must match the directory; and its body "
            "must be non-empty. Optional frontmatter: author, version, and "
            "lifecycle ('ongoing' or 'finite')."
        )
    _, frontmatter, _, body = parsed
    if "trigger" not in frontmatter:
        return (
            "RESPONSIBILITY.md frontmatter uses `trigger:` — the area "
            "and its facets, within 120 characters — in place of "
            "`description:`. Rewrite the frontmatter with trigger and "
            "retry."
        )
    if frontmatter.get("lifecycle") == "finite" and not _has_done_when_heading(
        body
    ):
        return (
            "A finite responsibility (lifecycle: finite) requires a "
            "'## Done when' section in its body stating the done state and "
            "the stop condition."
        )
    return None

def _description_length_notice(content: str) -> str | None:
    """Advisory attached to accepted RESPONSIBILITY.md writes; never blocks."""

    try:
        parsed = _parse_document(content.encode("utf-8"))
    except (UnicodeError, yaml.YAMLError, ValueError):
        return None
    if parsed is None or len(parsed[2]) <= _ROSTER_DESCRIPTION_LENGTH:
        return None
    return (
        f"Saved. The trigger is over {_ROSTER_DESCRIPTION_LENGTH} "
        "characters — shorten it."
    )

def document_lifecycle(document: str) -> str:
    """Best-effort lifecycle read for scheduled-run context assembly.

    Fails toward "ongoing": the run receives the full document either way, so
    the finite operating rule is only ever added on a trusted parse.
    """

    try:
        parsed = _parse_document(document.encode("utf-8"))
    except (UnicodeError, yaml.YAMLError, ValueError):
        return "ongoing"
    if parsed is None:
        return "ongoing"
    return str(parsed[1].get("lifecycle") or "ongoing")

def package_char_limit_for(parts: Sequence[str]) -> int | None:
    """The character budget for a package-relative budgeted-file path."""

    if tuple(parts) == ("RESPONSIBILITY.md",):
        return MAX_RESPONSIBILITY_MD_CHARS
    if tuple(parts) == ("STATE.md",):
        return MAX_STATE_ROOT_CHARS
    if parts and parts[0] == STATE_DIRNAME:
        return MAX_STATE_FILE_CHARS
    if parts and parts[0] == ARCHIVE_DIRNAME:
        return MAX_ARCHIVE_FILE_CHARS
    if parts and parts[0] in {REFERENCES_DIRNAME, CONTEXT_DIRNAME}:
        return MAX_CONTEXT_FILE_CHARS
    return None

def package_byte_ceiling_for(parts: Sequence[str]) -> int | None:
    """The internal byte ceiling for a package-relative budgeted-file path."""

    if tuple(parts) == ("RESPONSIBILITY.md",):
        return RESPONSIBILITY_MD_BYTE_CEILING
    if tuple(parts) == ("STATE.md",):
        return STATE_ROOT_BYTE_CEILING
    if parts and parts[0] == STATE_DIRNAME:
        return STATE_FILE_BYTE_CEILING
    if parts and parts[0] == ARCHIVE_DIRNAME:
        return ARCHIVE_FILE_BYTE_CEILING
    if parts and parts[0] in {REFERENCES_DIRNAME, CONTEXT_DIRNAME}:
        return CONTEXT_FILE_BYTE_CEILING
    return None

def _validate_state(content: str) -> str | None:
    try:
        content.encode("utf-8")
    except UnicodeError:
        return "STATE.md must be UTF-8 text."
    if len(content) > MAX_STATE_ROOT_CHARS:
        return (
            "This write would put STATE.md at "
            f"{len(content):,}/{MAX_STATE_ROOT_CHARS:,} chars. Consolidate "
            "now: delete stale state, merge overlapping notes, retire "
            "finished records to archive/ (move whole files with mv; append "
            "trimmed lines to archive/log.md) — spill a record into a "
            "state/ file (referenced here) only if it still does not fit — "
            "then retry, all in this turn."
        )
    return None

def _validate_state_leaf(relative_path: str, content: str) -> str | None:
    try:
        content.encode("utf-8")
    except UnicodeError:
        return f"{relative_path} must be UTF-8 text."
    if len(content) > MAX_STATE_FILE_CHARS:
        return (
            f"This write would put {relative_path} at "
            f"{len(content):,}/{MAX_STATE_FILE_CHARS:,} chars. Consolidate "
            "now: delete stale detail, merge overlapping notes, retire "
            "finished records to archive/ — divide it into two state/ files "
            "only if it covers two subjects — then retry, all in this turn."
        )
    return None

def _validate_context_file(relative_path: str, content: str) -> str | None:
    try:
        content.encode("utf-8")
    except UnicodeError:
        return f"{relative_path} must be UTF-8 text."
    if len(content) > MAX_CONTEXT_FILE_CHARS:
        return (
            f"This write would put {relative_path} at "
            f"{len(content):,}/{MAX_CONTEXT_FILE_CHARS:,} chars. A "
            "references/ file carries one kind of work: move rules and "
            "boundaries to the charter, how a service is operated to its "
            "connection manual, learned records to state/ — divide it into "
            "two references/ files only if it covers two kinds of work — "
            "then retry, all in this turn."
        )
    return None

def _archive_shard_target(
    relative_path: str, package_fd: int | None = None
) -> str:
    """The next *free* shard name beside an overflowing archive file.

    log.md → log-2.md, customers/acme-2.md → customers/acme-3.md. With a
    package descriptor, occupied suffixes are probed past: recommending an
    existing shard would have the follow-up write_file replace it — losing
    archived history — so the first nonexistent target is returned.
    """

    path = PurePosixPath(relative_path)
    match = re.match(r"^(.*)-(\d+)$", path.stem)
    if match:
        base, suffix = match.group(1), int(match.group(2)) + 1
    else:
        base, suffix = path.stem, 2
    candidate = path.with_name(f"{base}-{suffix}{path.suffix}")
    if package_fd is not None:
        for _ in range(MAX_ARCHIVE_FILES):
            try:
                os.stat(
                    str(candidate), dir_fd=package_fd, follow_symlinks=False
                )
            except OSError:
                break
            suffix += 1
            candidate = path.with_name(f"{base}-{suffix}{path.suffix}")
    return str(candidate)

def _validate_archive_file(
    relative_path: str, content: str, package_fd: int | None = None
) -> str | None:
    try:
        content.encode("utf-8")
    except UnicodeError:
        return f"{relative_path} must be UTF-8 text."
    if len(content) > MAX_ARCHIVE_FILE_CHARS:
        return (
            f"This write would put {relative_path} at "
            f"{len(content):,}/{MAX_ARCHIVE_FILE_CHARS:,} chars. Archive "
            "files shard rather than shrink: start "
            f"{_archive_shard_target(relative_path, package_fd)} and "
            "continue there."
        )
    return None

def _open_root(root: Path | str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(os.fspath(root), flags)

def scan_workspace_responsibilities(
    responsibilities_root: Path | str,
    *,
    only_names: frozenset[str] | None = None,
) -> ResponsibilityScanResult:
    """Scan only direct child packages.

    Malformed or unreadable packages vanish from the roster entries but are
    reported in ``package_errors`` so schedule reconciliation freezes rather
    than retires their rows; only an affirmatively absent folder is absent
    from both.

    ``only_names`` restricts the scan to the named packages: the scoped
    receipt refresh reads one just-written package instead of the whole
    root. The result is authoritative only for those names.
    """

    try:
        root_fd = _open_root(responsibilities_root)
    except FileNotFoundError:
        return ResponsibilityScanResult(entries=(), package_errors={})
    except OSError as exc:
        raise ResponsibilityFilesystemError(
            f"cannot safely open responsibility root: {exc}"
        ) from exc
    records: list[ResponsibilityIndexEntry] = []
    package_errors: dict[str, str] = {}
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for entry in sorted(os.scandir(root_fd), key=lambda item: item.name):
            if entry.name == WORKSPACE_ARCHIVE_DIRNAME:
                continue
            if only_names is not None and entry.name not in only_names:
                continue
            if validate_responsibility_name(entry.name) is not None:
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                package_fd = os.open(entry.name, flags, dir_fd=root_fd)
            except OSError as exc:
                package_errors[entry.name] = f"package unreadable: {exc}"
                continue
            try:
                raw = _read_regular_file_fd(
                    package_fd,
                    "RESPONSIBILITY.md",
                    max_bytes=RESPONSIBILITY_MD_BYTE_CEILING,
                )
                parsed = _parse_document(raw)
                if parsed is None or parsed[0] != entry.name:
                    package_errors[entry.name] = (
                        "RESPONSIBILITY.md metadata is malformed"
                    )
                    continue
                state = _read_regular_file_fd(
                    package_fd,
                    "STATE.md",
                    max_bytes=STATE_ROOT_BYTE_CEILING,
                )
                state.decode("utf-8")
                references_info = None
                for dirname in (REFERENCES_DIRNAME, CONTEXT_DIRNAME):
                    try:
                        references_info = os.stat(
                            dirname, dir_fd=package_fd, follow_symlinks=False
                        )
                    except FileNotFoundError:
                        continue
                    break
                if references_info is None or not stat.S_ISDIR(
                    references_info.st_mode
                ):
                    package_errors[entry.name] = "references: not a directory"
                    continue
                declarations, schedule_errors = _scan_schedule_declarations(
                    package_fd, flags
                )
                webhooks, webhook_errors = _scan_webhook_declarations(package_fd, flags)
                name, frontmatter, description, _ = parsed
                records.append(
                    ResponsibilityIndexEntry(
                        name=name,
                        description=description,
                        relative_path=name,
                        frontmatter_extract=frontmatter,
                        content_hash=hashlib.sha256(raw).hexdigest(),
                        schedules=declarations,
                        schedule_errors=schedule_errors,
                        webhooks=webhooks,
                        webhook_errors=webhook_errors,
                        scripts=_scan_package_scripts(package_fd),
                        stray_files=_scan_stray_files(package_fd),
                    )
                )
            except (
                OSError,
                ResponsibilityFilesystemError,
                UnicodeError,
                yaml.YAMLError,
                ValueError,
            ) as exc:
                package_errors[entry.name] = f"package invalid: {exc}"
                continue
            finally:
                os.close(package_fd)
    finally:
        os.close(root_fd)
    return ResponsibilityScanResult(
        entries=tuple(sorted(records, key=lambda item: item.name)),
        package_errors=package_errors,
    )

def _support_path(file_path: str, *, action: str) -> tuple[str, ...]:
    if not isinstance(file_path, str) or not file_path or "\\" in file_path:
        raise ResponsibilityFilesystemError("file_path is required")
    path = PurePosixPath(file_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ResponsibilityFilesystemError(
            "file_path must stay inside the responsibility"
        )
    parts = tuple(path.parts)
    if action == "patch" and parts == ("RESPONSIBILITY.md",):
        return parts
    if parts == ("STATE.md",) and action in {"patch", "write_file"}:
        return parts
    if (
        len(parts) == 2
        and parts[0] in {SCHEDULES_DIRNAME, WEBHOOKS_DIRNAME}
        and parts[1].endswith(".yaml")
    ):
        return parts
    if (
        len(parts) >= 2
        and parts[0]
        in {REFERENCES_DIRNAME, CONTEXT_DIRNAME, STATE_DIRNAME, ARCHIVE_DIRNAME}
        and parts[-1].endswith(".md")
    ):
        return parts
    if (
        len(parts) == 2
        and parts[0] == "scripts"
        and parts[1].endswith(_SCRIPT_SUFFIXES)
    ):
        return parts
    raise ResponsibilityFilesystemError(
        "support files must be STATE.md, Markdown files below state/ or "
        "archive/, schedules/<name>.yaml, webhooks/<name>.yaml, "
        "scripts/<name>.sh|.bash|.py, or Markdown files below references/"
    )

def _count_tree_files(package_fd: int, dirname: str) -> int:
    """Count regular files under one package subdirectory, symlink-safe."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        base_fd = os.open(dirname, flags, dir_fd=package_fd)
    except OSError:
        return 0
    count = 0
    stack = [base_fd]
    while stack:
        fd = stack.pop()
        try:
            for entry in os.scandir(fd):
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(os.open(entry.name, flags, dir_fd=fd))
                    elif entry.is_file(follow_symlinks=False):
                        count += 1
                except OSError:
                    continue
        except OSError:
            pass
        finally:
            os.close(fd)
    return count

def _leaf_file_exists(package_fd: int, parts: Sequence[str]) -> bool:
    try:
        info = os.stat(
            "/".join(parts), dir_fd=package_fd, follow_symlinks=False
        )
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode)

def _validate_leaf_count(package_fd: int, parts: Sequence[str]) -> str | None:
    """Runaway-loop backstop: reject a write that would create a new file
    past the directory cap; rewriting an existing file never triggers it."""

    if _leaf_file_exists(package_fd, parts):
        return None
    if parts[0] == STATE_DIRNAME:
        if _count_tree_files(package_fd, STATE_DIRNAME) >= MAX_STATE_FILES:
            return (
                f"state/ already holds {MAX_STATE_FILES} files — its cap. "
                "Retire finished state/ files to archive/ or merge "
                "overlapping ones (and drop their references in STATE.md), "
                "then retry, all in this turn."
            )
    elif parts[0] == ARCHIVE_DIRNAME:
        if _count_tree_files(package_fd, ARCHIVE_DIRNAME) >= MAX_ARCHIVE_FILES:
            return (
                f"archive/ already holds {MAX_ARCHIVE_FILES} files — its "
                "cap. Merge small records into fewer archive/ files (up to "
                f"their {MAX_ARCHIVE_FILE_CHARS:,}-char bound), then retry, "
                "all in this turn."
            )
    return None

def read_responsibility_package(
    responsibilities_root: Path | str,
    reference: str,
) -> dict[str, Any]:
    """Resolve one package at fire time, returning raw required documents."""

    name = normalize_responsibility_reference(reference)
    try:
        root_fd = _open_root(responsibilities_root)
    except FileNotFoundError:
        return {"found": False, "name": name}
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            package_fd = os.open(name, flags, dir_fd=root_fd)
        except OSError:
            return {"found": False, "name": name}
        try:
            errors: list[str] = []
            documents: dict[str, str] = {}
            for filename, limit in (
                ("RESPONSIBILITY.md", RESPONSIBILITY_MD_BYTE_CEILING),
                ("STATE.md", STATE_ROOT_BYTE_CEILING),
            ):
                try:
                    documents[filename] = _read_regular_file_fd(
                        package_fd, filename, max_bytes=limit
                    ).decode("utf-8")
                except (OSError, UnicodeError, ValueError) as exc:
                    errors.append(f"{filename}: {exc}")
            references_info = None
            for dirname in (REFERENCES_DIRNAME, CONTEXT_DIRNAME):
                try:
                    references_info = os.stat(
                        dirname, dir_fd=package_fd, follow_symlinks=False
                    )
                except OSError:
                    continue
                break
            if references_info is None or not stat.S_ISDIR(
                references_info.st_mode
            ):
                errors.append("references: not a directory")
            raw_document = documents.get("RESPONSIBILITY.md")
            valid = False
            if raw_document is not None:
                try:
                    parsed = _parse_document(raw_document.encode("utf-8"))
                    valid = parsed is not None and parsed[0] == name
                except (UnicodeError, yaml.YAMLError, ValueError):
                    valid = False
            if not valid:
                errors.append("RESPONSIBILITY.md metadata is malformed")
            # Top-level state/ entries drive the conditional operating rule;
            # a failed listing reads as empty and simply omits the rule.
            state_entries: list[str] = []
            try:
                state_fd = os.open(STATE_DIRNAME, flags, dir_fd=package_fd)
                try:
                    for entry in sorted(
                        os.scandir(state_fd), key=lambda item: item.name
                    ):
                        if entry.name.startswith("."):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            state_entries.append(f"{entry.name}/")
                        elif entry.is_file(follow_symlinks=False):
                            state_entries.append(entry.name)
                finally:
                    os.close(state_fd)
            except OSError:
                state_entries = []
            return {
                "found": True,
                "name": name,
                "responsibility_document": documents.get("RESPONSIBILITY.md", ""),
                "state_document": documents.get("STATE.md", ""),
                "state_entries": state_entries,
                "malformed": bool(errors),
                "errors": errors,
            }
        finally:
            os.close(package_fd)
    finally:
        os.close(root_fd)
