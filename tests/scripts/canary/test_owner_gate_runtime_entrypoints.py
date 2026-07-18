from __future__ import annotations

import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest


ROOT = Path(__file__).parents[3]
ENTRYPOINT_ROOT = ROOT / "ops/muncho/owner-gate/bin"
REVISION = "a" * 40
ENTRYPOINTS = (
    ("muncho-owner-gate-intake", 29102),
    ("muncho-passkey-v2-authority", 29102),
    ("muncho-passkey-v2-executor", 29103),
    ("muncho-passkey-v2-web", 29101),
    ("muncho-owner-gate-install", 0),
    ("muncho-owner-gate-activate-storage", 0),
)


def _validator(name: str) -> Callable[..., Path]:
    path = ENTRYPOINT_ROOT / name
    namespace = runpy.run_path(
        str(path),
        run_name=f"owner_gate_entrypoint_test_{name.replace('-', '_')}",
    )
    return namespace["_validated_release"]


def _flags(**changes: Any) -> SimpleNamespace:
    values = {
        "isolated": 1,
        "ignore_environment": 1,
        "no_user_site": 1,
        "safe_path": True,
        "dont_write_bytecode": 1,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _root_lstat(
    path: os.PathLike[str] | str,
    *,
    wrong_owner: Path | None = None,
) -> SimpleNamespace:
    state = os.lstat(path)
    resolved = Path(path).resolve(strict=False)
    return SimpleNamespace(
        st_mode=state.st_mode,
        st_uid=1 if wrong_owner is not None and resolved == wrong_owner else 0,
        st_gid=0,
        st_nlink=state.st_nlink,
    )


def _layout(
    tmp_path: Path,
    name: str,
    *,
    revision: str = REVISION,
) -> tuple[Path, Path, Path, Path]:
    install = tmp_path / "opt/muncho-owner-gate"
    releases = install / "releases"
    release = releases / revision
    entrypoint = release / "bin" / name
    interpreter = release / "venv/bin/python"
    entrypoint.parent.mkdir(parents=True)
    interpreter.parent.mkdir(parents=True)
    entrypoint.write_text("# fixed test entrypoint\n", encoding="utf-8")
    interpreter.write_bytes(b"fixed copied python test identity")
    install.chmod(0o755)
    releases.chmod(0o755)
    release.chmod(0o555)
    entrypoint.chmod(0o555)
    interpreter.chmod(0o755)
    return releases, release, entrypoint, interpreter


def _call(
    validator: Callable[..., Path],
    *,
    releases: Path,
    entrypoint: Path,
    interpreter: Path,
    uid: int,
    lstat_fn: Callable[[os.PathLike[str] | str], Any] = _root_lstat,
    flags: Any | None = None,
    executable: Path | None = None,
    getuid_fn: Callable[[], int] | None = None,
    geteuid_fn: Callable[[], int] | None = None,
    getgid_fn: Callable[[], int] | None = None,
    getegid_fn: Callable[[], int] | None = None,
) -> Path:
    return validator(
        entrypoint=entrypoint,
        executable=interpreter if executable is None else executable,
        releases_root=releases,
        lstat_fn=lstat_fn,
        getuid_fn=(lambda: uid) if getuid_fn is None else getuid_fn,
        geteuid_fn=(lambda: uid) if geteuid_fn is None else geteuid_fn,
        getgid_fn=(lambda: uid) if getgid_fn is None else getgid_fn,
        getegid_fn=(lambda: uid) if getegid_fn is None else getegid_fn,
        flags=_flags() if flags is None else flags,
    )


@pytest.mark.parametrize(("name", "uid"), ENTRYPOINTS)
def test_runtime_entrypoint_accepts_only_its_exact_release_identity(
    tmp_path: Path,
    name: str,
    uid: int,
) -> None:
    releases, release, entrypoint, interpreter = _layout(tmp_path, name)
    assert _call(
        _validator(name),
        releases=releases,
        entrypoint=entrypoint,
        interpreter=interpreter,
        uid=uid,
    ) == release

    if name != "muncho-owner-gate-install":
        current = releases.parent / "current"
        current.symlink_to(release, target_is_directory=True)
        assert _call(
            _validator(name),
            releases=releases,
            entrypoint=current / "bin" / name,
            interpreter=current / "venv/bin/python",
            uid=uid,
        ) == release


def test_runtime_installer_accepts_only_exact_private_staging_name(
    tmp_path: Path,
) -> None:
    name = "muncho-owner-gate-install"
    revision = f".{REVISION}.bootstrap"
    releases, release, entrypoint, interpreter = _layout(
        tmp_path,
        name,
        revision=revision,
    )
    release.chmod(0o700)

    assert _call(
        _validator(name),
        releases=releases,
        entrypoint=entrypoint,
        interpreter=interpreter,
        uid=0,
    ) == release


@pytest.mark.parametrize(
    "revision",
    (f"{REVISION}.bootstrap", f".{REVISION}", f".{REVISION}.bootstrap.extra"),
)
def test_runtime_installer_rejects_ambiguous_staging_names(
    tmp_path: Path,
    revision: str,
) -> None:
    name = "muncho-owner-gate-install"
    releases, release, entrypoint, interpreter = _layout(
        tmp_path,
        name,
        revision=revision,
    )
    release.chmod(0o700)

    with pytest.raises(SystemExit, match="runtime_entrypoint_invalid"):
        _call(
            _validator(name),
            releases=releases,
            entrypoint=entrypoint,
            interpreter=interpreter,
            uid=0,
        )


@pytest.mark.parametrize("revision", ["A" * 40, "g" * 40, "a" * 39, "a" * 41])
def test_runtime_entrypoint_rejects_noncanonical_release_revision(
    tmp_path: Path,
    revision: str,
) -> None:
    name, uid = ENTRYPOINTS[0]
    releases, _, entrypoint, interpreter = _layout(
        tmp_path, name, revision=revision
    )
    with pytest.raises(SystemExit, match="runtime_entrypoint_invalid"):
        _call(
            _validator(name),
            releases=releases,
            entrypoint=entrypoint,
            interpreter=interpreter,
            uid=uid,
        )


@pytest.mark.parametrize(("name", "uid"), ENTRYPOINTS)
def test_runtime_entrypoint_rejects_wrong_effective_identity(
    tmp_path: Path,
    name: str,
    uid: int,
) -> None:
    releases, _, entrypoint, interpreter = _layout(tmp_path, name)
    with pytest.raises(SystemExit, match="runtime_entrypoint_invalid"):
        _call(
            _validator(name),
            releases=releases,
            entrypoint=entrypoint,
            interpreter=interpreter,
            uid=uid,
            geteuid_fn=lambda: uid + 1,
        )


@pytest.mark.parametrize("case", ["symlink", "hardlink", "mode", "owner"])
def test_runtime_entrypoint_rejects_file_identity_drift(
    tmp_path: Path,
    case: str,
) -> None:
    name, uid = ENTRYPOINTS[2]
    releases, release, entrypoint, interpreter = _layout(tmp_path, name)
    wrong_owner: Path | None = None
    if case == "symlink":
        target = entrypoint.with_name("real-executor")
        entrypoint.rename(target)
        entrypoint.symlink_to(target.name)
    elif case == "hardlink":
        os.link(entrypoint, entrypoint.with_name("second-link"))
    elif case == "mode":
        entrypoint.chmod(0o755)
    else:
        wrong_owner = entrypoint.resolve()

    def lstat_fn(path: os.PathLike[str] | str) -> SimpleNamespace:
        return _root_lstat(path, wrong_owner=wrong_owner)

    with pytest.raises(SystemExit, match="runtime_entrypoint_invalid"):
        _call(
            _validator(name),
            releases=releases,
            entrypoint=entrypoint,
            interpreter=interpreter,
            uid=uid,
            lstat_fn=lstat_fn,
        )
    assert release.exists()


@pytest.mark.parametrize(
    "flags",
    [_flags(isolated=0), _flags(dont_write_bytecode=0), _flags(safe_path=False)],
)
def test_runtime_entrypoint_rejects_missing_isolated_runtime_flags(
    tmp_path: Path,
    flags: SimpleNamespace,
) -> None:
    name, uid = ENTRYPOINTS[3]
    releases, _, entrypoint, interpreter = _layout(tmp_path, name)
    with pytest.raises(SystemExit, match="runtime_entrypoint_invalid"):
        _call(
            _validator(name),
            releases=releases,
            entrypoint=entrypoint,
            interpreter=interpreter,
            uid=uid,
            flags=flags,
        )


def test_runtime_entrypoint_rejects_interpreter_from_another_release(
    tmp_path: Path,
) -> None:
    name, uid = ENTRYPOINTS[1]
    releases, _, entrypoint, interpreter = _layout(tmp_path, name)
    other = tmp_path / "other-python"
    other.write_bytes(b"other interpreter")
    other.chmod(0o755)
    with pytest.raises(SystemExit, match="runtime_entrypoint_invalid"):
        _call(
            _validator(name),
            releases=releases,
            entrypoint=entrypoint,
            interpreter=interpreter,
            executable=other,
            uid=uid,
        )


@pytest.mark.parametrize("case", ["symlink", "hardlink", "mode", "owner"])
def test_runtime_entrypoint_rejects_interpreter_file_identity_drift(
    tmp_path: Path,
    case: str,
) -> None:
    name, uid = ENTRYPOINTS[3]
    releases, _, entrypoint, interpreter = _layout(tmp_path, name)
    wrong_owner: Path | None = None
    if case == "symlink":
        target = interpreter.with_name("python-copy")
        interpreter.rename(target)
        interpreter.symlink_to(target.name)
    elif case == "hardlink":
        os.link(interpreter, interpreter.with_name("python-second-link"))
    elif case == "mode":
        interpreter.chmod(0o555)
    else:
        wrong_owner = interpreter.resolve()

    def lstat_fn(path: os.PathLike[str] | str) -> SimpleNamespace:
        return _root_lstat(path, wrong_owner=wrong_owner)

    with pytest.raises(SystemExit, match="runtime_entrypoint_invalid"):
        _call(
            _validator(name),
            releases=releases,
            entrypoint=entrypoint,
            interpreter=interpreter,
            uid=uid,
            lstat_fn=lstat_fn,
        )


@pytest.mark.parametrize("case", ["release_mode", "release_owner", "base_mode"])
def test_runtime_entrypoint_rejects_release_directory_identity_drift(
    tmp_path: Path,
    case: str,
) -> None:
    name, uid = ENTRYPOINTS[0]
    releases, release, entrypoint, interpreter = _layout(tmp_path, name)
    wrong_owner: Path | None = None
    if case == "release_mode":
        release.chmod(0o755)
    elif case == "release_owner":
        wrong_owner = release.resolve()
    else:
        releases.chmod(0o555)

    def lstat_fn(path: os.PathLike[str] | str) -> SimpleNamespace:
        return _root_lstat(path, wrong_owner=wrong_owner)

    with pytest.raises(SystemExit, match="runtime_entrypoint_invalid"):
        _call(
            _validator(name),
            releases=releases,
            entrypoint=entrypoint,
            interpreter=interpreter,
            uid=uid,
            lstat_fn=lstat_fn,
        )
