"""Job-local native preparation admission; never a portable environment cache."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
import re
import tomllib
from pathlib import Path

from pm.store import current_target, tree_digest
from scripts.build.inputs import AgentInputs, RESOURCE_ENV


def prepared_path(out: Path) -> Path:
    return out.with_name(out.name + ".prepared.json")


@contextmanager
def preparation_lock(out: Path):
    from hermes_cli.runtime_state import _lock
    from pm.filesystem import is_junction

    if out != out.resolve():
        raise ValueError("symlinked native output")
    for name in ("hermes-agent", "tools", "venv", "pm-runtime", "uv-cache", "bin",
                 "enabled-features.json", "manifest.json"):
        path = out / name
        if path.is_symlink() or (path.exists() and is_junction(path)):
            raise ValueError(f"symlinked native output: {path}")
    with out.with_name(out.name + ".prepare.lock").open("a+b") as lock:
        if not _lock(lock.fileno(), wait=False):
            raise ValueError("native output is already in use")
        yield


def _digest(path: Path) -> str:
    return tree_digest(path) if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest(code: Path) -> str:
    """Ignore only assembly-owned products, leaving admitted source immutable."""
    project = tomllib.loads((code / "pyproject.toml").read_text(encoding="utf-8-sig"))["project"]
    dist = re.sub(r"[-_.]+", "_", project["name"])
    generated = {"install-stamp.json", "hermes_cli/tui_dist", "hermes_cli/web_dist",
                 f"{dist}-{project['version']}.dist-info"}
    files = {}
    for directory, dirs, names in os.walk(code):
        dirs[:] = [name for name in dirs if name != "__pycache__"
                   and (Path(directory) / name).relative_to(code).as_posix() not in generated]
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                files[path.relative_to(code).as_posix()] = os.readlink(path)
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(code).as_posix()
            if relative not in generated:
                files[relative] = os.readlink(path) if path.is_symlink() else _digest(path)
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _owned(path: Path, out: Path) -> None:
    if not path.is_absolute() or not path.exists() or not path.resolve().is_relative_to(out):
        raise ValueError(f"missing or escaped prepared path: {path}")


def _check_links(path: Path, out: Path) -> None:
    from pm.filesystem import is_junction

    for directory, dirs, files in os.walk(path):
        for name in dirs + files:
            entry = Path(directory) / name
            if entry.is_symlink() or is_junction(entry):
                _owned(entry, out)


def _input_paths(inputs: AgentInputs, out: Path) -> list[Path]:
    expected = {"code": out / "hermes-agent", "project": out / "hermes-agent/pyproject.toml",
                "tools": out / "tools", "environment": out / "venv", "pm_runtime": out / "pm-runtime",
                "features": out / "enabled-features.json"}
    if (any(getattr(inputs, key) != path for key, path in expected.items())
            or inputs.placement != "contained" or inputs.repo != "hermes-agent"
            or inputs.frontends or inputs.stamp or inputs.command_dir or inputs.env
            or inputs.resources != {name: inputs.code / name for name in RESOURCE_ENV}
            or not inputs.site_packages.is_relative_to(inputs.environment)):
        raise ValueError("native prepared input layout changed")
    paths = [inputs.code, inputs.tools, inputs.environment, inputs.pm_runtime,
             inputs.python, inputs.features, out / "uv-cache"]
    for path in [inputs.project, inputs.site_packages, *inputs.resources.values(), *paths]:
        _owned(path, out)
    return paths


def publish_prepared(out: Path, source: Path, revision: str, inputs: AgentInputs) -> Path:
    from hermes_cli.runtime_state import _atomic_bytes
    from pm.paths import lockfile_path

    inputs.validate(out)
    paths = _input_paths(inputs, out)
    digests = {}
    for path in paths:
        _owned(path, out)
        _check_links(path, out)
        digests[path.relative_to(out).as_posix()] = _source_digest(path) if path == inputs.code else _digest(path)
    data = {"schema": 1, "out": str(out), "source": str(source), "revision": revision,
            "lock": _digest(lockfile_path()), "inputs": asdict(inputs), "digests": digests}
    prepared = prepared_path(out)
    _atomic_bytes(prepared, (json.dumps(data, default=str, indent=2) + "\n").encode())
    return prepared


def load_prepared(prepared: Path) -> AgentInputs:
    from pm.paths import lockfile_path

    try:
        data = json.loads(prepared.read_text(encoding="utf-8-sig"))
        out = Path(data["out"])
        if (prepared.is_symlink() or out != out.resolve() or data["schema"] != 1
                or prepared != prepared_path(out) or data["lock"] != _digest(lockfile_path())
                or not re.fullmatch(r"[a-f0-9]{40}", data["revision"])):
            raise ValueError("native preparation identity changed")
        inputs = AgentInputs.from_dict(data["inputs"])
        if inputs.target != current_target() or not inputs.ref:
            raise ValueError("native preparation target/ref changed")
        inputs.validate(out)
        required = {path.relative_to(out).as_posix() for path in _input_paths(inputs, out)}
        if set(data["digests"]) != required:
            raise ValueError("native preparation inventory changed")
        for name, digest in data["digests"].items():
            path = out / name
            _owned(path, out)
            _check_links(path, out)
            actual = _source_digest(path) if path == inputs.code else _digest(path)
            if actual != digest:
                raise ValueError(f"prepared bytes changed: {name}")
        return inputs
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid native preparation; run preparation again: {exc}") from exc