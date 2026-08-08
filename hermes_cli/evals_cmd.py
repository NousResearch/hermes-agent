"""CLI implementation for ``hermes evals``.

This is deliberately an edge command rather than a model tool: task mining and
validation should not enlarge every agent request's tool schema.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from hermes_cli.evals_core import (
    build_candidate_from_trace,
    score_run_artifact,
    validate_manifest,
)


_MAX_MANIFEST_BYTES = 2 * 1024 * 1024


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _open_session_db():
    from hermes_state import SessionDB

    return SessionDB()


def _default_candidate_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "evals" / "candidates"


def load_manifest(path: Path) -> Mapping[str, Any]:
    """Load one bounded YAML task manifest without constructing Python objects."""

    path = path.expanduser()
    if not path.is_file():
        raise ValueError(f"manifest is not a file: {path}")
    size = path.stat().st_size
    if size > _MAX_MANIFEST_BYTES:
        raise ValueError(
            f"manifest exceeds {_MAX_MANIFEST_BYTES} byte safety limit: {path}"
        )
    try:
        loaded = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read YAML manifest {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"manifest root must be a mapping: {path}")
    return loaded


def _atomic_private_write(
    path: Path,
    rendered: str,
    *,
    force: bool,
    label: str,
) -> Path:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing to write through symlink: {path}")

    tmp_path: Path | None = None
    try:
        fd, raw_tmp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        tmp_path = Path(raw_tmp)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        if force:
            # os.replace replaces a symlink entry rather than following it.
            os.replace(tmp_path, path)
        else:
            # Link publication is atomic and cannot overwrite a file that
            # appears after a check-then-write race.
            try:
                os.link(tmp_path, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite existing {label}: {path}"
                ) from exc
            tmp_path.unlink()
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
    return path


def write_manifest(path: Path, manifest: Mapping[str, Any], *, force: bool = False) -> Path:
    """Atomically write a private YAML manifest, refusing symlink targets."""

    rendered = yaml.safe_dump(
        dict(manifest),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return _atomic_private_write(
        path,
        rendered,
        force=force,
        label="manifest",
    )


def load_run_artifact(path: Path) -> Mapping[str, Any]:
    path = path.expanduser()
    if not path.is_file():
        raise ValueError(f"run artifact is not a file: {path}")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError(f"run artifact exceeds {_MAX_MANIFEST_BYTES} byte safety limit: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON run artifact {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"run artifact root must be a mapping: {path}")
    return loaded


def write_score(path: Path, result: Mapping[str, Any], *, force: bool = False) -> Path:
    rendered = json.dumps(dict(result), indent=2, sort_keys=True) + "\n"
    return _atomic_private_write(path, rendered, force=force, label="score")


def _manifest_paths(path: Path) -> Iterable[Path]:
    path = path.expanduser()
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise ValueError(f"path does not exist: {path}")
    yield from sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.casefold() in {".yaml", ".yml"}
    )


def _cmd_mine(args) -> int:
    db = _open_session_db()
    try:
        resolved = db.resolve_session_id(args.session_id)
        if not resolved:
            print(f"Error: session ID or prefix is missing or ambiguous: {args.session_id}")
            return 2
        session = db.get_session(resolved)
        if not session:
            print(f"Error: session not found: {resolved}")
            return 2
        messages = db.get_messages(resolved)
        manifest = build_candidate_from_trace(
            session,
            messages,
        )
    except ValueError as exc:
        print(f"Error: cannot mine session: {exc}")
        return 2
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()

    output = getattr(args, "output", None)
    path = Path(output) if output else _default_candidate_dir() / f"{manifest['id']}.yaml"
    try:
        write_manifest(path, manifest, force=bool(getattr(args, "force", False)))
    except (OSError, ValueError) as exc:
        print(f"Error: could not write candidate: {exc}")
        return 1

    print(f"Candidate written: {path}")
    print(f"Signals found: {len(manifest.get('signals', []))}")
    print(
        "Review required: inspect the sanitized candidate, define task-specific "
        "checks, then set status: approved."
    )
    return 0


def _cmd_validate(args) -> int:
    try:
        paths = list(_manifest_paths(Path(args.path)))
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2
    if not paths:
        print("Error: no YAML manifests found")
        return 2

    failed = 0
    for path in paths:
        try:
            manifest = load_manifest(path)
        except ValueError as exc:
            print(f"INVALID {path}: {exc}")
            failed += 1
            continue
        result = validate_manifest(manifest)
        if result.errors:
            print(f"INVALID {path}")
            for error in result.errors:
                print(f"  error: {error}")
            failed += 1
            continue
        if result.warnings:
            label = "NOT READY" if getattr(args, "ready", False) else "VALID CANDIDATE"
            print(f"{label} {path}")
            for warning in result.warnings:
                print(f"  warning: {warning}")
            if getattr(args, "ready", False):
                failed += 1
        else:
            print(f"READY {path}")
    return 1 if failed else 0


def _cmd_score(args) -> int:
    try:
        manifest = load_manifest(Path(args.task))
        run = load_run_artifact(Path(args.run))
        result = score_run_artifact(manifest, run)
    except ValueError as exc:
        print(f"Error: cannot score run: {exc}")
        return 2

    output = getattr(args, "output", None)
    if output:
        try:
            write_score(
                Path(output),
                result,
                force=bool(getattr(args, "force", False)),
            )
        except (OSError, ValueError) as exc:
            print(f"Error: could not write score: {exc}")
            return 1

    status = str(result["status"])
    print(
        f"{status.upper().replace('_', ' ')} {result['task_id']} "
        f"({result['deterministic']['passed']}/{result['deterministic']['total']} deterministic checks)"
    )
    if not output:
        print(json.dumps(result, indent=2, sort_keys=True))
    if status == "passed":
        return 0
    if status == "needs_judge":
        return 3
    return 1


def cmd_evals(args) -> int:
    action = getattr(args, "evals_action", None)
    if action == "mine":
        return _cmd_mine(args)
    if action == "validate":
        return _cmd_validate(args)
    if action == "score":
        return _cmd_score(args)
    print("Use `hermes evals --help` to see available actions.")
    return 2
