#!/usr/bin/env python3
"""Install and attest exact release-local Cloud Muncho runtime dependencies.

The production release is deliberately self-contained: it does not resolve
``agent-browser`` through PATH/npx, import an arbitrary DDGS installation, or
download a browser after activation.  This packager installs npm bytes from the
committed package lock, Python wheels from the committed uv lock with hash
checking, and one fixed Chrome-for-Testing archive whose URL and digest are
part of this source contract.

The implementation is packaged in the Hermes wheel so an immutable release
executes the code it just installed, rather than a source-tree helper.  The
source-side script is only a compatibility CLI wrapper.

No credential is read or recorded by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway.canonical_writer_release_contract import (
    RUNTIME_DEPENDENCY_MANIFEST_RELATIVE_PATH,
    RUNTIME_DEPENDENCY_NPM_CACHE_RELATIVE_PATH,
)


MANIFEST_SCHEMA = "muncho-production-runtime-dependencies.v1"
PREPARATION_SCHEMA = "muncho-production-runtime-dependency-preparation.v1"
MANIFEST_RELATIVE_PATH = RUNTIME_DEPENDENCY_MANIFEST_RELATIVE_PATH
CANARY_RELEASE_BASE = Path("/opt/muncho-canary-releases")
REVISION = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

AGENT_BROWSER_VERSION = "0.26.0"
AGENT_BROWSER_INTEGRITY = (
    "sha512-pdqSfjwbFSp+qnwlb2g23e9wXveIOfMi19xpPA9xZUbzEAUp6W4YBZj6Ybj8z4M7WkcbGDDYc+oDIHDt9R3EDQ=="
)
AGENT_BROWSER_WRAPPER = Path("node_modules/agent-browser/bin/agent-browser.js")
AGENT_BROWSER_NATIVE = Path(
    "node_modules/agent-browser/bin/agent-browser-linux-x64"
)
AGENT_BROWSER_SHIM = Path("node_modules/.bin/agent-browser")
AGENT_BROWSER_CONFIG = Path(
    "ops/muncho/runtime/dependencies/agent-browser.json"
)
AGENT_BROWSER_CONFIG_BYTES = b"{}\n"
AGENT_BROWSER_CONFIG_ROOT_UID = 0
AGENT_BROWSER_CONFIG_ROOT_GID = 0
NODE_VERSION = "24.18.0"
NODE_URL = f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.xz"
NODE_ARCHIVE_SHA256 = "55aa7153f9d88f28d765fcdad5ae6945b5c0f98a36881703817e4c450fa76742"
NODE_ARCHIVE_SIZE = 31_511_588
NODE_ROOT = Path("ops/muncho/runtime/dependencies/node-linux-x64")
NODE_EXECUTABLE = NODE_ROOT / "bin/node"
NPM_EXECUTABLE = NODE_ROOT / "bin/npm"
NPM_CACHE_RELATIVE_PATH = RUNTIME_DEPENDENCY_NPM_CACHE_RELATIVE_PATH
NPM_CACHE_MAX_ENTRIES = 50_000
NPM_CACHE_MAX_BYTES = 512 * 1024 * 1024

CHROME_VERSION = "150.0.7871.114"
CHROME_URL = (
    "https://storage.googleapis.com/chrome-for-testing-public/"
    f"{CHROME_VERSION}/linux64/chrome-linux64.zip"
)
CHROME_ARCHIVE_SHA256 = (
    "03963c0dd9bf91e9b0e760cff37680f9b92ff42758182286382787622323cf9d"
)
CHROME_ARCHIVE_SIZE = 187_336_525
CHROME_ROOT = Path("ops/muncho/runtime/dependencies/chrome-linux64")
CHROME_EXECUTABLE = CHROME_ROOT / "chrome"

SUPPORTED_IMPLEMENTATION = "CPython"
SUPPORTED_PYTHON = (3, 11)
SUPPORTED_MACHINE = "x86_64"
SUPPORTED_SYSTEM = "Linux"

# Closed DDGS dependency graph selected by the ``ddgs`` extra on CPython 3.11
# Linux x86_64.  Every version and every accepted wheel hash is read from and
# checked against uv.lock before pip is invoked with --require-hashes.
DDGS_LOCKED_DISTRIBUTIONS = {
    "anyio": "4.12.1",
    "brotli": "1.2.0",
    "certifi": "2026.5.20",
    "click": "8.3.1",
    "ddgs": "9.14.4",
    "fake-useragent": "2.2.0",
    "h11": "0.16.0",
    "h2": "4.3.0",
    "hpack": "4.1.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "hyperframe": "6.1.0",
    "idna": "3.15",
    "lxml": "6.1.1",
    "primp": "1.3.1",
    "socksio": "1.0.0",
    "typing-extensions": "4.15.0",
}


class RuntimeDependencyError(RuntimeError):
    """One stable, non-secret release dependency failure."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular(
    path: Path,
    *,
    maximum: int,
    allow_empty: bool = False,
    expected: os.stat_result | None = None,
) -> bytes:
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeDependencyError("runtime_dependency_source_unavailable") from exc
    if (
        resolved != path
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not (0 if allow_empty else 1) <= before.st_size <= maximum
    ):
        raise RuntimeDependencyError("runtime_dependency_source_invalid")
    if expected is not None and _regular_identity(before) != _regular_identity(expected):
        raise RuntimeDependencyError("runtime_dependency_source_raced")
    payload = path.read_bytes()
    after = path.lstat()
    if (
        len(payload) != before.st_size
        or _regular_identity(before) != _regular_identity(after)
    ):
        raise RuntimeDependencyError("runtime_dependency_source_raced")
    return payload


def _release_location(
    value: Path,
    revision: str,
    release_address: Path | None = None,
) -> tuple[Path, Path]:
    if REVISION.fullmatch(revision or "") is None:
        raise RuntimeDependencyError("runtime_dependency_revision_invalid")
    try:
        release = value.resolve(strict=True)
    except OSError as exc:
        raise RuntimeDependencyError("runtime_dependency_release_unavailable") from exc
    try:
        address = (
            release
            if release_address is None
            else release_address.resolve(strict=False)
        )
    except OSError as exc:
        raise RuntimeDependencyError(
            "runtime_dependency_release_address_invalid"
        ) from exc
    production_address = address.name == f"hermes-agent-{revision[:12]}"
    canary_address = address.name == revision and address.parent == CANARY_RELEASE_BASE
    production_staging = (
        production_address
        and release.parent == address.parent
        and re.fullmatch(
            rf"\.hermes-agent-{revision[:12]}\.tmp\.[1-9][0-9]*",
            release.name,
        )
        is not None
    )
    if (
        release != value
        or (release_address is not None and release_address != address)
        or not (production_address or canary_address)
        or (release != address and not production_staging)
        or (canary_address and release != address)
    ):
        raise RuntimeDependencyError("runtime_dependency_release_address_invalid")
    marker = _read_regular(release / ".codex-source-commit", maximum=128)
    if marker != (revision + "\n").encode("ascii"):
        raise RuntimeDependencyError("runtime_dependency_release_identity_invalid")
    return release, address


def _release_root(value: Path, revision: str) -> Path:
    """Compatibility helper for an already-final exact release address."""

    release, _address = _release_location(value, revision)
    return release


def _release_interpreter(release: Path) -> Path:
    """Bind packaging to the exact interpreter that invoked this script.

    The release publisher owns the virtual-environment directory name.  The
    dependency package does not guess between multiple names: it requires this
    script to be executed by one ``<release>/<single-env>/bin/python`` and uses
    that exact environment for install and attestation.
    """

    interpreter = Path(sys.executable)
    if not interpreter.is_absolute():
        raise RuntimeDependencyError("runtime_dependency_interpreter_invalid")
    try:
        relative = interpreter.relative_to(release)
    except ValueError as exc:
        raise RuntimeDependencyError(
            "runtime_dependency_interpreter_not_release_local"
        ) from exc
    if (
        len(relative.parts) != 3
        or relative.parts[1] != "bin"
        or relative.parts[2] not in {"python", "python3", "python3.11"}
    ):
        raise RuntimeDependencyError("runtime_dependency_interpreter_invalid")
    return interpreter


def _validate_supported_platform() -> None:
    if (
        platform.python_implementation() != SUPPORTED_IMPLEMENTATION
        or sys.version_info[:2] != SUPPORTED_PYTHON
        or platform.machine() != SUPPORTED_MACHINE
        or platform.system() != SUPPORTED_SYSTEM
    ):
        raise RuntimeDependencyError("runtime_dependency_platform_unsupported")


def _locked_python_requirements(release: Path) -> tuple[str, Mapping[str, Any]]:
    pyproject_raw = _read_regular(release / "pyproject.toml", maximum=2 * 1024 * 1024)
    uv_raw = _read_regular(release / "uv.lock", maximum=8 * 1024 * 1024)
    try:
        pyproject = tomllib.loads(pyproject_raw.decode("utf-8", errors="strict"))
        lock = tomllib.loads(uv_raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeDependencyError("runtime_dependency_python_lock_invalid") from exc
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    if extras.get("ddgs") != ["ddgs==9.14.4"]:
        raise RuntimeDependencyError("runtime_dependency_python_pin_invalid")
    packages = {
        item.get("name"): item
        for item in lock.get("package", [])
        if isinstance(item, Mapping) and item.get("name") in DDGS_LOCKED_DISTRIBUTIONS
    }
    if set(packages) != set(DDGS_LOCKED_DISTRIBUTIONS):
        raise RuntimeDependencyError("runtime_dependency_python_lock_incomplete")
    lines: list[str] = []
    identities: dict[str, Any] = {}
    for name, expected_version in sorted(DDGS_LOCKED_DISTRIBUTIONS.items()):
        item = packages[name]
        wheels = item.get("wheels")
        if item.get("version") != expected_version or not isinstance(wheels, list):
            raise RuntimeDependencyError("runtime_dependency_python_lock_drifted")
        hashes = sorted(
            {
                str(wheel.get("hash", "")).removeprefix("sha256:")
                for wheel in wheels
                if isinstance(wheel, Mapping)
                and isinstance(wheel.get("url"), str)
                and str(wheel["url"]).endswith(".whl")
                and isinstance(wheel.get("hash"), str)
                and str(wheel["hash"]).startswith("sha256:")
            }
        )
        if not hashes or any(SHA256.fullmatch(value) is None for value in hashes):
            raise RuntimeDependencyError("runtime_dependency_python_wheel_hash_invalid")
        lines.append(
            f"{name}=={expected_version} "
            + " ".join(f"--hash=sha256:{value}" for value in hashes)
        )
        identities[name] = {"version": expected_version, "wheel_sha256": hashes}
    contract = {
        "pyproject_sha256": _sha256(pyproject_raw),
        "uv_lock_sha256": _sha256(uv_raw),
        "distributions": identities,
    }
    return "\n".join(lines) + "\n", contract


def _validate_node_lock(release: Path) -> Mapping[str, Any]:
    package_raw = _read_regular(release / "package.json", maximum=1024 * 1024)
    lock_raw = _read_regular(release / "package-lock.json", maximum=4 * 1024 * 1024)
    try:
        package = json.loads(package_raw.decode("utf-8", errors="strict"))
        lock = json.loads(lock_raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeDependencyError("runtime_dependency_node_lock_invalid") from exc
    item = lock.get("packages", {}).get("node_modules/agent-browser", {})
    if (
        package.get("dependencies", {}).get("agent-browser") != "^0.26.0"
        or item.get("version") != AGENT_BROWSER_VERSION
        or item.get("integrity") != AGENT_BROWSER_INTEGRITY
        or item.get("bin") != {"agent-browser": "bin/agent-browser.js"}
    ):
        raise RuntimeDependencyError("runtime_dependency_agent_browser_pin_invalid")
    return {
        "package_json_sha256": _sha256(package_raw),
        "package_lock_sha256": _sha256(lock_raw),
        "version": AGENT_BROWSER_VERSION,
        "integrity": AGENT_BROWSER_INTEGRITY,
    }


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path,
    timeout: int,
    extra_environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    environment = {
        "HOME": str(cwd),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{cwd / NODE_ROOT / 'bin'}:/usr/bin:/bin",
        "PIP_NO_CACHE_DIR": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if extra_environment:
        environment.update(extra_environment)
    try:
        result = subprocess.run(
            tuple(arguments),
            cwd=cwd,
            env=environment,
            capture_output=True,
            check=False,
            timeout=timeout,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeDependencyError("runtime_dependency_install_execution_failed") from exc
    if result.returncode != 0 or len(result.stdout) > 2 * 1024 * 1024 or len(result.stderr) > 2 * 1024 * 1024:
        raise RuntimeDependencyError("runtime_dependency_install_failed")
    return result


def _remove_bounded_npm_cache(release: Path, cache: Path) -> None:
    """Remove only the exact disposable npm cache after bounding its shape."""

    if cache != release / NPM_CACHE_RELATIVE_PATH:
        raise RuntimeDependencyError("runtime_dependency_npm_cache_address_invalid")
    if not os.path.lexists(cache):
        return
    root = cache.lstat()
    if not stat.S_ISDIR(root.st_mode) or stat.S_ISLNK(root.st_mode):
        raise RuntimeDependencyError("runtime_dependency_npm_cache_invalid")
    entries = 0
    total_bytes = 0
    for current, directories, files in os.walk(
        cache,
        topdown=True,
        followlinks=False,
    ):
        for name in (*directories, *files):
            entries += 1
            if entries > NPM_CACHE_MAX_ENTRIES:
                raise RuntimeDependencyError("runtime_dependency_npm_cache_oversized")
            item = Path(current) / name
            state = item.lstat()
            if stat.S_ISDIR(state.st_mode):
                continue
            if (
                not stat.S_ISREG(state.st_mode)
                or stat.S_ISLNK(state.st_mode)
                or state.st_nlink != 1
            ):
                raise RuntimeDependencyError("runtime_dependency_npm_cache_invalid")
            total_bytes += state.st_size
            if total_bytes > NPM_CACHE_MAX_BYTES:
                raise RuntimeDependencyError("runtime_dependency_npm_cache_oversized")
    shutil.rmtree(cache)
    if os.path.lexists(cache):
        raise RuntimeDependencyError("runtime_dependency_npm_cache_cleanup_failed")


def _install_python(release: Path, requirements: str) -> None:
    interpreter = _release_interpreter(release)
    _read_regular(interpreter.resolve(strict=True), maximum=64 * 1024 * 1024)
    # ``uv sync`` deliberately removes installers that are not part of the
    # application lock.  Re-seed the pinned managed Python's bundled pip
    # before using the hash-locked, binary-only runtime dependency contract.
    _run(
        (
            str(interpreter),
            "-I",
            "-m",
            "ensurepip",
            "--upgrade",
            "--default-pip",
        ),
        cwd=release,
        timeout=120,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".muncho-ddgs-", suffix=".txt", dir=release
    )
    temporary = Path(temporary_name)
    try:
        os.write(descriptor, requirements.encode("ascii"))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _run(
            (
                str(interpreter),
                "-I",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-input",
                "--no-deps",
                "--only-binary=:all:",
                "--require-hashes",
                "--force-reinstall",
                "--requirement",
                str(temporary),
            ),
            cwd=release,
            timeout=600,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _install_node(release: Path) -> None:
    cache = release / NPM_CACHE_RELATIVE_PATH
    if os.path.lexists(cache) or os.path.lexists(release / ".npm"):
        raise RuntimeDependencyError("runtime_dependency_npm_cache_collision")
    os.mkdir(cache, 0o700)
    try:
        _run(
            (
                str(release / NPM_EXECUTABLE),
                "ci",
                "--ignore-scripts",
                "--omit=dev",
                "--workspaces=false",
                "--no-audit",
                "--no-fund",
            ),
            cwd=release,
            timeout=600,
            extra_environment={
                "npm_config_cache": str(cache),
                "npm_config_update_notifier": "false",
            },
        )
        native = release / AGENT_BROWSER_NATIVE
        wrapper = release / AGENT_BROWSER_WRAPPER
        if not native.is_file() or not wrapper.is_file():
            raise RuntimeDependencyError("runtime_dependency_agent_browser_missing")
        native.chmod(0o555)
        wrapper.chmod(0o555)
    finally:
        _remove_bounded_npm_cache(release, cache)
    if os.path.lexists(release / ".npm"):
        raise RuntimeDependencyError("runtime_dependency_npm_cache_leaked")


def _install_agent_browser_config(release: Path) -> None:
    """Install the one inert config accepted by the browser controller.

    ``agent-browser`` otherwise auto-discovers configuration in the process
    home and working directory.  The production controller always passes this
    exact release-local file explicitly, so packaging owns its bytes and mode
    just like the Node, native launcher, and Chrome identities.
    """

    path = release / AGENT_BROWSER_CONFIG
    path.parent.mkdir(parents=True, mode=0o755, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            remaining = memoryview(AGENT_BROWSER_CONFIG_BYTES)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise RuntimeDependencyError(
                        "runtime_dependency_agent_browser_config_write_failed"
                    )
                remaining = remaining[written:]
            # A setgid parent can assign its group to a newly-created file
            # even when that is not the process effective group (notably on
            # macOS temporary test roots).  Bind the final identity through
            # the already-open, nofollow descriptor before the atomic rename.
            os.fchown(descriptor, os.geteuid(), os.getegid())  # windows-footgun: ok — Linux release-packager boundary
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except (OSError, RuntimeDependencyError) as exc:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if isinstance(exc, RuntimeDependencyError):
            raise
        raise RuntimeDependencyError(
            "runtime_dependency_agent_browser_config_install_failed"
        ) from exc


def _agent_browser_config_identity(
    release: Path,
    *,
    expected_uid: int,
    expected_gid: int,
) -> Mapping[str, Any]:
    """Attest the exact inert config at one explicit ownership phase."""

    path = release / AGENT_BROWSER_CONFIG
    try:
        state = path.lstat()
        payload = _read_regular(path, maximum=64)
    except (OSError, RuntimeDependencyError) as exc:
        raise RuntimeDependencyError(
            "runtime_dependency_agent_browser_config_invalid"
        ) from exc
    if (
        payload != AGENT_BROWSER_CONFIG_BYTES
        or state.st_nlink != 1
        or state.st_uid != expected_uid
        or state.st_gid != expected_gid
        or stat.S_IMODE(state.st_mode) != 0o444
    ):
        raise RuntimeDependencyError(
            "runtime_dependency_agent_browser_config_invalid"
        )
    return {
        "path": str(path),
        "sha256": _sha256(payload),
        "owner_uid": state.st_uid,
        "group_gid": state.st_gid,
        "mode": "0444",
        "regular_one_link": True,
    }


def _install_node_runtime(release: Path) -> None:
    archive = release / "ops/muncho/runtime/dependencies/node-linux-x64.tar.xz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(NODE_URL, timeout=60) as response, archive.open("wb") as output:
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > NODE_ARCHIVE_SIZE:
                    raise RuntimeDependencyError("runtime_dependency_node_archive_invalid")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except (OSError, RuntimeDependencyError) as exc:
        archive.unlink(missing_ok=True)
        if isinstance(exc, RuntimeDependencyError):
            raise
        raise RuntimeDependencyError("runtime_dependency_node_download_failed") from exc
    if size != NODE_ARCHIVE_SIZE or digest.hexdigest() != NODE_ARCHIVE_SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeDependencyError("runtime_dependency_node_archive_invalid")
    target = release / NODE_ROOT
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, mode=0o755)
    archive_prefix = f"node-v{NODE_VERSION}-linux-x64"
    try:
        with tarfile.open(archive, mode="r:xz") as bundle:
            members = bundle.getmembers()
            if not 1 <= len(members) <= 10_000:
                raise RuntimeDependencyError("runtime_dependency_node_archive_invalid")
            total = 0
            for member in members:
                path = Path(member.name)
                if (
                    not path.parts
                    or path.parts[0] != archive_prefix
                    or ".." in path.parts
                ):
                    raise RuntimeDependencyError("runtime_dependency_node_archive_invalid")
                relative = Path(*path.parts[1:])
                if not relative.parts:
                    continue
                destination = target / relative
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    destination.chmod(0o755)
                    continue
                if member.issym():
                    link = Path(member.linkname)
                    try:
                        resolved_link = (destination.parent / link).resolve(strict=False)
                        resolved_link.relative_to(target)
                    except (OSError, ValueError):
                        raise RuntimeDependencyError("runtime_dependency_node_archive_invalid")
                    if link.is_absolute():
                        raise RuntimeDependencyError("runtime_dependency_node_archive_invalid")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.symlink_to(member.linkname)
                    continue
                if not member.isfile() or member.size > 256 * 1024 * 1024:
                    raise RuntimeDependencyError("runtime_dependency_node_archive_invalid")
                total += member.size
                if total > 512 * 1024 * 1024:
                    raise RuntimeDependencyError("runtime_dependency_node_archive_invalid")
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeDependencyError("runtime_dependency_node_archive_invalid")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("xb") as output:
                    copied = 0
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        copied += len(chunk)
                        output.write(chunk)
                    if copied != member.size:
                        raise RuntimeDependencyError("runtime_dependency_node_archive_invalid")
                destination.chmod(0o555 if member.mode & 0o111 else 0o444)
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeDependencyError("runtime_dependency_node_extract_failed") from exc
    archive.unlink()


def _install_chrome(release: Path) -> None:
    archive = release / "ops/muncho/runtime/dependencies/chrome-linux64.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(CHROME_URL, timeout=60) as response, archive.open("wb") as output:
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > CHROME_ARCHIVE_SIZE:
                    raise RuntimeDependencyError("runtime_dependency_chrome_archive_invalid")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except (OSError, RuntimeDependencyError) as exc:
        try:
            archive.unlink()
        except FileNotFoundError:
            pass
        if isinstance(exc, RuntimeDependencyError):
            raise
        raise RuntimeDependencyError("runtime_dependency_chrome_download_failed") from exc
    if size != CHROME_ARCHIVE_SIZE or digest.hexdigest() != CHROME_ARCHIVE_SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeDependencyError("runtime_dependency_chrome_archive_invalid")
    target = release / CHROME_ROOT
    if target.exists():
        import shutil

        shutil.rmtree(target)
    target.mkdir(parents=True, mode=0o755)
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if not 1 <= len(members) <= 1024:
                raise RuntimeDependencyError("runtime_dependency_chrome_archive_invalid")
            for member in members:
                path = Path(member.filename)
                if not path.parts or path.parts[0] != "chrome-linux64" or ".." in path.parts:
                    raise RuntimeDependencyError("runtime_dependency_chrome_archive_invalid")
                relative = Path(*path.parts[1:])
                if not relative.parts:
                    continue
                destination = target / relative
                mode = member.external_attr >> 16
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    destination.chmod(0o755)
                    continue
                if not stat.S_ISREG(mode) or member.file_size > 512 * 1024 * 1024:
                    raise RuntimeDependencyError("runtime_dependency_chrome_archive_invalid")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("xb") as output:
                    copied = 0
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        copied += len(chunk)
                        output.write(chunk)
                    if copied != member.file_size:
                        raise RuntimeDependencyError("runtime_dependency_chrome_archive_invalid")
                destination.chmod(0o555 if mode & 0o111 else 0o444)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeDependencyError("runtime_dependency_chrome_extract_failed") from exc
    archive.unlink()


def _tree_identity(path: Path, *, maximum_files: int, maximum_bytes: int) -> Mapping[str, Any]:
    records: list[Mapping[str, Any]] = []
    total = 0
    for item in sorted(path.rglob("*")):
        relative = str(item.relative_to(path))
        state = item.lstat()
        if stat.S_ISDIR(state.st_mode):
            records.append({"path": relative, "kind": "directory", "mode": stat.S_IMODE(state.st_mode)})
            continue
        if stat.S_ISLNK(state.st_mode):
            target = os.readlink(item)
            records.append({"path": relative, "kind": "symlink", "target": target})
            continue
        if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
            raise RuntimeDependencyError("runtime_dependency_tree_invalid")
        total += state.st_size
        if len(records) >= maximum_files or total > maximum_bytes:
            raise RuntimeDependencyError("runtime_dependency_tree_oversized")
        payload = _read_regular(
            item,
            maximum=max(1, state.st_size),
            allow_empty=True,
            expected=state,
        )
        records.append(
            {
                "path": relative,
                "kind": "file",
                "mode": stat.S_IMODE(state.st_mode),
                "size": state.st_size,
                "sha256": _sha256(payload),
            }
        )
    if not records:
        raise RuntimeDependencyError("runtime_dependency_tree_empty")
    return {
        "file_count": sum(item["kind"] == "file" for item in records),
        "total_bytes": total,
        "tree_sha256": _sha256(_canonical_bytes(records)),
    }


def _distribution_identity(
    name: str,
    release: Path,
) -> Mapping[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeDependencyError("runtime_dependency_distribution_missing") from exc
    version = distribution.version
    expected = DDGS_LOCKED_DISTRIBUTIONS[name]
    if version != expected or not distribution.files:
        raise RuntimeDependencyError("runtime_dependency_distribution_drifted")
    records: list[Mapping[str, Any]] = []
    for entry in sorted(distribution.files, key=str):
        try:
            located = Path(distribution.locate_file(entry)).resolve(strict=True)
            state = located.lstat()
        except OSError as exc:
            raise RuntimeDependencyError(
                "runtime_dependency_source_unavailable"
            ) from exc
        try:
            relative = located.relative_to(_release_interpreter(release).parents[1])
        except ValueError as exc:
            raise RuntimeDependencyError("runtime_dependency_distribution_escaped") from exc
        if stat.S_ISDIR(state.st_mode):
            continue
        payload = _read_regular(
            located,
            maximum=64 * 1024 * 1024,
            allow_empty=True,
            expected=state,
        )
        records.append(
            {"path": str(relative), "size": len(payload), "sha256": _sha256(payload)}
        )
    if not records:
        raise RuntimeDependencyError("runtime_dependency_distribution_empty")
    return {
        "version": version,
        "file_count": len(records),
        "files_sha256": _sha256(_canonical_bytes(records)),
    }


def _manifest_value(
    release_root: Path,
    revision: str,
    *,
    release_address: Path | None = None,
) -> Mapping[str, Any]:
    """Recompute the complete dependency identity without changing the release."""

    release, address = _release_location(
        release_root,
        revision,
        release_address,
    )
    _validate_supported_platform()
    _requirements, python_lock = _locked_python_requirements(release)
    node_lock = _validate_node_lock(release)
    wrapper = release / AGENT_BROWSER_WRAPPER
    native = release / AGENT_BROWSER_NATIVE
    shim = release / AGENT_BROWSER_SHIM
    chrome = release / CHROME_EXECUTABLE
    for path in (wrapper, native, chrome):
        state = path.lstat()
        if not stat.S_ISREG(state.st_mode) or not os.access(path, os.X_OK):
            raise RuntimeDependencyError("runtime_dependency_executable_invalid")
    if not shim.is_symlink() or shim.resolve(strict=True) != wrapper:
        raise RuntimeDependencyError("runtime_dependency_agent_browser_shim_invalid")
    _agent_browser_config_identity(
        release,
        expected_uid=AGENT_BROWSER_CONFIG_ROOT_UID,
        expected_gid=AGENT_BROWSER_CONFIG_ROOT_GID,
    )
    config_payload = AGENT_BROWSER_CONFIG_BYTES
    browser_version = _run((str(shim), "--version"), cwd=release, timeout=10)
    chrome_version = _run((str(chrome), "--version"), cwd=release, timeout=10)
    node = release / NODE_EXECUTABLE
    npm = release / NPM_EXECUTABLE
    interpreter = _release_interpreter(release)
    node_version = _run((str(node), "--version"), cwd=release, timeout=10)
    npm_version = _run((str(npm), "--version"), cwd=release, timeout=10)
    if browser_version.stdout.decode("ascii", errors="strict").strip() != f"agent-browser {AGENT_BROWSER_VERSION}":
        raise RuntimeDependencyError("runtime_dependency_agent_browser_version_invalid")
    if chrome_version.stdout.decode("ascii", errors="strict").strip() != f"Google Chrome for Testing {CHROME_VERSION}":
        raise RuntimeDependencyError("runtime_dependency_chrome_version_invalid")
    if node_version.stdout.decode("ascii", errors="strict").strip() != f"v{NODE_VERSION}":
        raise RuntimeDependencyError("runtime_dependency_node_version_invalid")
    unsigned = {
        "schema": MANIFEST_SCHEMA,
        "release_revision": revision,
        "release_address": str(address),
        "platform": {
            "implementation": SUPPORTED_IMPLEMENTATION,
            "python_major_minor": list(SUPPORTED_PYTHON),
            "system": SUPPORTED_SYSTEM,
            "machine": SUPPORTED_MACHINE,
        },
        "source": {
            **node_lock,
            **{key: python_lock[key] for key in ("pyproject_sha256", "uv_lock_sha256")},
            "chrome_url": CHROME_URL,
            "chrome_archive_sha256": CHROME_ARCHIVE_SHA256,
            "chrome_archive_size": CHROME_ARCHIVE_SIZE,
            "node_url": NODE_URL,
            "node_archive_sha256": NODE_ARCHIVE_SHA256,
            "node_archive_size": NODE_ARCHIVE_SIZE,
        },
        "agent_browser": {
            "version": AGENT_BROWSER_VERSION,
            "config_path": str(address / AGENT_BROWSER_CONFIG),
            "config_sha256": _sha256(config_payload),
            "wrapper_path": str(address / AGENT_BROWSER_WRAPPER),
            "wrapper_sha256": _sha256(_read_regular(wrapper, maximum=1024 * 1024)),
            "native_path": str(address / AGENT_BROWSER_NATIVE),
            "native_sha256": _sha256(_read_regular(native, maximum=64 * 1024 * 1024)),
            "package_tree": _tree_identity(
                release / "node_modules/agent-browser",
                maximum_files=4096,
                maximum_bytes=256 * 1024 * 1024,
            ),
            "node_path": str(address / NODE_EXECUTABLE),
            "node_version": node_version.stdout.decode("ascii", errors="strict").strip(),
            "node_sha256": _sha256(
                _read_regular(node, maximum=256 * 1024 * 1024)
            ),
            "npm_path": str(address / NPM_EXECUTABLE),
            "npm_version": npm_version.stdout.decode("ascii", errors="strict").strip(),
            "npm_target_sha256": _sha256(
                _read_regular(npm.resolve(strict=True), maximum=16 * 1024 * 1024)
            ),
            "node_tree": _tree_identity(
                release / NODE_ROOT,
                maximum_files=10_000,
                maximum_bytes=512 * 1024 * 1024,
            ),
        },
        "chrome": {
            "version": CHROME_VERSION,
            "executable_path": str(address / CHROME_EXECUTABLE),
            "executable_sha256": _sha256(_read_regular(chrome, maximum=512 * 1024 * 1024)),
            "tree": _tree_identity(
                release / CHROME_ROOT,
                maximum_files=2048,
                maximum_bytes=1024 * 1024 * 1024,
            ),
        },
        "python": {
            "interpreter_path": str(
                address / interpreter.relative_to(release)
            ),
            "distributions": {
                name: _distribution_identity(name, release)
                for name in sorted(DDGS_LOCKED_DISTRIBUTIONS)
            },
            "lock_contract_sha256": _sha256(_canonical_bytes(python_lock)),
        },
        "secret_material_recorded": False,
    }
    return {**unsigned, "manifest_sha256": _sha256(_canonical_bytes(unsigned))}


def build_manifest(
    release_root: Path,
    revision: str,
    *,
    release_address: Path | None = None,
) -> Mapping[str, Any]:
    """Recompute and atomically install the dependency identity manifest."""

    release, address = _release_location(
        release_root,
        revision,
        release_address,
    )
    manifest = _manifest_value(
        release,
        revision,
        release_address=address,
    )
    path = release / MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, mode=0o755, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(manifest) + b"\n")
    temporary.chmod(0o444)
    os.replace(temporary, path)
    return manifest


def verify_manifest(
    release_root: Path,
    revision: str,
    *,
    release_address: Path | None = None,
) -> Mapping[str, Any]:
    release, address = _release_location(
        release_root,
        revision,
        release_address,
    )
    path = release / MANIFEST_RELATIVE_PATH
    raw = _read_regular(path, maximum=2 * 1024 * 1024)
    try:
        expected = json.loads(raw.decode("ascii", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeDependencyError("runtime_dependency_manifest_invalid") from exc
    if not isinstance(expected, Mapping) or raw != _canonical_bytes(expected) + b"\n":
        raise RuntimeDependencyError("runtime_dependency_manifest_invalid")
    # Verification must be observational.  In particular it must never replace
    # a tampered manifest with freshly computed bytes before reporting drift.
    observed = _manifest_value(
        release,
        revision,
        release_address=address,
    )
    if observed != expected:
        raise RuntimeDependencyError("runtime_dependency_manifest_drifted")
    return expected


def install_release_dependencies(
    release_root: Path,
    revision: str,
    *,
    release_address: Path | None = None,
) -> Mapping[str, Any]:
    prepare_release_dependencies(
        release_root,
        revision,
        release_address=release_address,
    )
    return build_manifest(
        release_root,
        revision,
        release_address=release_address,
    )


def prepare_release_dependencies(
    release_root: Path,
    revision: str,
    *,
    release_address: Path | None = None,
) -> Mapping[str, Any]:
    """Build dependencies as the release owner, before the root config seal.

    This phase never claims a production manifest.  The caller must
    mechanically chown only the exact inert browser config to root:root and
    then invoke :func:`build_manifest` as the release owner.
    """

    release, address = _release_location(
        release_root,
        revision,
        release_address,
    )
    _validate_supported_platform()
    requirements, _contract = _locked_python_requirements(release)
    _validate_node_lock(release)
    _install_python(release, requirements)
    _install_node_runtime(release)
    _install_node(release)
    _install_chrome(release)
    _install_agent_browser_config(release)
    config = _agent_browser_config_identity(
        release,
        expected_uid=os.geteuid(),  # windows-footgun: ok — Linux release-packager boundary
        expected_gid=os.getegid(),  # windows-footgun: ok — Linux release-packager boundary
    )
    unsigned = {
        "schema": PREPARATION_SCHEMA,
        "release_revision": revision,
        "release_address": str(address),
        "agent_browser_config": config,
        "root_seal_required": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "preparation_sha256": _sha256(_canonical_bytes(unsigned)),
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package exact Muncho runtime dependencies")
    parser.add_argument(
        "command",
        choices=("prepare", "install", "build-manifest", "verify"),
    )
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--release-address")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    try:
        release_address = (
            None
            if args.release_address is None
            else Path(args.release_address)
        )
        if args.command == "prepare":
            value = prepare_release_dependencies(
                Path(args.release_root),
                args.revision,
                release_address=release_address,
            )
        elif args.command == "install":
            value = install_release_dependencies(
                Path(args.release_root),
                args.revision,
                release_address=release_address,
            )
        elif args.command == "build-manifest":
            value = build_manifest(
                Path(args.release_root),
                args.revision,
                release_address=release_address,
            )
        else:
            value = verify_manifest(
                Path(args.release_root),
                args.revision,
                release_address=release_address,
            )
    except (OSError, RuntimeDependencyError):
        return 2
    digest_name = (
        "preparation_sha256"
        if value["schema"] == PREPARATION_SCHEMA
        else "manifest_sha256"
    )
    print(_canonical_bytes({
        "schema": value["schema"],
        "release_revision": value["release_revision"],
        digest_name: value[digest_name],
        "secret_material_recorded": False,
    }).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
