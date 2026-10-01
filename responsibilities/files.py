"""Responsibility validation shared by native write and patch operations."""
from pathlib import Path, PurePosixPath
from typing import Iterable
import os
import hermes_yaml as yaml

from responsibilities.common import (
    MAX_PACKAGE_SCRIPT_BYTES, ResponsibilityFilesystemError,
    get_responsibilities_root, validate_responsibility_name,
)
from responsibilities.declarations import validate_schedule_declaration, validate_webhook_declaration
from responsibilities.packages import (
    _description_length_notice, _support_path, _validate_archive_file,
    _validate_context_file, _validate_document, _validate_leaf_count,
    _validate_state, _validate_state_leaf, package_char_limit_for,
    package_byte_ceiling_for, _parse_document,
)


def package_path(path: str):
    root = Path(os.path.abspath(get_responsibilities_root()))
    candidate = Path(os.path.abspath(path))
    try:
        parts = candidate.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) < 2 or parts[0] == ".archive":
        return None
    if validate_responsibility_name(parts[0]):
        raise ResponsibilityFilesystemError("Invalid responsibility package name.")
    for parent in (candidate, *candidate.parents):
        if parent.is_symlink():
            raise ResponsibilityFilesystemError("Responsibility paths must not follow symlinks.")
        if parent == root:
            break
    return root / parts[0], parts[1:]


def validate_write(path: str, content: str, *, moving_from: str | None = None) -> str | None:
    target = package_path(path)
    if target is None:
        return None
    package, parts = target
    relative = "/".join(parts)
    from tools.skill_provenance import is_unattended_review

    if is_unattended_review() and parts[0] in {"schedules", "webhooks"}:
        return "Reviews cannot edit schedule or webhook declarations. Record the proposed change in STATE.md."
    if parts == ("RESPONSIBILITY.md",):
        return _validate_document(content, expected_name=package.name)
    _support_path(relative, action="write_file")
    if not (package / "RESPONSIBILITY.md").is_file():
        return "Create RESPONSIBILITY.md before writing the responsibility's supporting files."
    source_target = package_path(moving_from) if moving_from else None
    same_tree_move = bool(source_target and source_target[0] == package
                          and source_target[1][0] == parts[0] and Path(moving_from).is_file())
    if parts[0] in {"state", "archive"} and not same_tree_move:
        fd = os.open(package, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            error = _validate_leaf_count(fd, parts)
        finally:
            os.close(fd)
        if error:
            return error
    if parts == ("STATE.md",):
        return _validate_state(content)
    validators = {
        "state": _validate_state_leaf, "archive": _validate_archive_file,
        "references": _validate_context_file, "context": _validate_context_file,
    }
    if parts[0] in validators:
        return validators[parts[0]](relative, content)
    if parts[0] == "scripts":
        if len(content.encode("utf-8")) > MAX_PACKAGE_SCRIPT_BYTES:
            return f"{relative} exceeds {MAX_PACKAGE_SCRIPT_BYTES} bytes."
        return None
    validator = {"schedules": validate_schedule_declaration, "webhooks": validate_webhook_declaration}[parts[0]]
    return validator(parts[-1], content.encode("utf-8"))[1]


def finish_write(path: str, content: str) -> str | None:
    target = package_path(path)
    if target is None:
        return None
    package, parts = target
    if parts == ("RESPONSIBILITY.md",):
        (package / "references").mkdir(exist_ok=True)
        try:
            with (package / "STATE.md").open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            pass
        return _description_length_notice(content)
    return None


MAX_PACKAGE_LISTING_FILES = 512

def usage_gauge(char_count: int, char_limit: int) -> str:
    """Memory-tool usage format. The percentage is deliberately not clamped
    so a legacy over-budget file reads honestly (e.g. '215% — …')."""

    pct = int((char_count / char_limit) * 100) if char_limit else 0
    return f"{pct}% — {char_count:,}/{char_limit:,} chars"

def _budgeted_file_usage(
    path: Path, *, char_limit: int, byte_ceiling: int
) -> str | None:
    """The usage gauge for one on-disk budgeted file, or ``None``.

    Bounded read: a terminal-created file may be arbitrarily large, and
    the gauge must never load it whole. Past the ceiling the file is
    off-contract — extras are omitted and the plain read result stands on
    its own.
    """

    try:
        with open(path, "rb") as handle:
            raw = handle.read(byte_ceiling + 1)
        if len(raw) > byte_ceiling:
            return None
        return usage_gauge(len(raw.decode("utf-8")), char_limit)
    except (OSError, UnicodeError):
        return None

def _list_package_files(
    package_dir: Path, *, subdirs: Iterable[str]
) -> dict[str, list[str]]:
    """Bounded, dotfile-free recursive listing of package subdirectories."""

    listing: dict[str, list[str]] = {}
    remaining = MAX_PACKAGE_LISTING_FILES
    for subdir in subdirs:
        if remaining <= 0:
            break
        base = package_dir / subdir
        files: list[str] = []
        stack = [base]
        while stack and remaining > 0:
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    entries = sorted(it, key=lambda item: item.name)
            except OSError:
                continue
            for item in entries:
                if item.name.startswith("."):
                    continue
                try:
                    if item.is_dir(follow_symlinks=False):
                        stack.append(Path(item.path))
                    elif item.is_file(follow_symlinks=False):
                        files.append(
                            str(PurePosixPath(subdir)
                                / Path(item.path).relative_to(base))
                        )
                        remaining -= 1
                        if remaining <= 0:
                            break
                except OSError:
                    continue
        if files:
            listing[subdir] = sorted(files)
    return listing

def _list_top_level_entries(package_dir: Path, subdir: str) -> list[str]:
    """Top-level entries of one package subdirectory, directories marked
    with a trailing slash and never recursed — the scheme-level view the
    reference discipline teaches, bounded like every advisory listing."""

    entries: list[str] = []
    try:
        with os.scandir(package_dir / subdir) as it:
            items = sorted(it, key=lambda item: item.name)
    except OSError:
        return []
    for item in items[:MAX_PACKAGE_LISTING_FILES]:
        if item.name.startswith("."):
            continue
        try:
            if item.is_dir(follow_symlinks=False):
                entries.append(f"{subdir}/{item.name}/")
            elif item.is_file(follow_symlinks=False):
                entries.append(f"{subdir}/{item.name}")
        except OSError:
            continue
    return entries


def read_extras(path: str) -> dict:
    target = package_path(path)
    if target is None:
        return {}
    package, parts = target
    limit = package_char_limit_for(parts)
    ceiling = package_byte_ceiling_for(parts)
    if limit is None or ceiling is None:
        return {}
    extras = {"responsibility": package.name}
    usage = _budgeted_file_usage(Path(path), char_limit=limit, byte_ceiling=ceiling)
    if usage is not None:
        extras["usage"] = usage
    if parts == ("RESPONSIBILITY.md",):
        with Path(path).open("rb") as handle:
            raw = handle.read(ceiling + 1)
        try:
            parsed = _parse_document(raw)
        except (ValueError, UnicodeError, yaml.YAMLError):
            parsed = None
        if parsed is not None:
            extras["description"] = parsed[2]
        recursive = _list_package_files(package, subdirs=("schedules", "webhooks", "references", "context"))
        linked = {}
        for folder in ("state", "schedules", "webhooks", "scripts", "references", "context", "archive"):
            entries = recursive.get(folder, []) if folder in {"schedules", "webhooks", "references", "context"} else _list_top_level_entries(package, folder)
            if folder == "state" and (package / "STATE.md").is_file():
                entries.insert(0, "STATE.md")
            if entries:
                linked[folder] = entries
        if linked:
            extras["linked_files"] = linked
    return extras



def mutation_receipt(path):
    target = package_path(path)
    if not target:
        return {key: value for key, value in read_extras(path).items() if key in {"responsibility", "usage"}}
    package, parts = target
    if parts[0] == "schedules":
        from responsibilities.schedules import reconcile
        errors = reconcile()
        return {"warning": errors[package.name + "/" + "/".join(parts)]} if package.name + "/" + "/".join(parts) in errors else {}
    if parts[0] == "webhooks":
        from responsibilities.webhook_store import reconcile, receipt
        reconcile()
        return receipt(package.name, Path(parts[-1]).stem)
    return {key: value for key, value in read_extras(path).items() if key in {"responsibility", "usage"}}


def validate_move(src: str, dst: str):
    target = package_path(dst)
    if target is None:
        return None, None
    if Path(src).is_symlink():
        return None, "Responsibility paths must not follow symlinks."
    ceiling = package_byte_ceiling_for(target[1]) or MAX_PACKAGE_SCRIPT_BYTES
    with Path(src).open('rb') as handle:
        raw = handle.read(ceiling + 1)
    if len(raw) > ceiling:
        return None, f"Move exceeds the destination's {ceiling:,}-byte limit."
    content = raw.decode('utf-8')
    return content, validate_write(dst, content, moving_from=src)
