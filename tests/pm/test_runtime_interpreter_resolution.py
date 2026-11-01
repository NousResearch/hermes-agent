"""PM's runtime must never be staged with an interpreter its own lock refuses.

The bootstrap fallback in ``pm.runtime.runtime_python`` used to pair the staged
``uv`` with the AMBIENT interpreter. On a host whose base venv is an older
minor than PM pins, uv then refuses the runtime graph outright ("The requested
interpreter resolved to Python 3.11.15, which is incompatible with the
project's Python requirement"), and the environment the sync built is abandoned.
The fallback must resolve a MANAGED interpreter of the pinned minor instead.
"""
import os
from pathlib import Path
import sys

import pytest


def _fake_uv(tmp_path: Path, managed: Path | None) -> Path:
    """A uv that reports *managed*, or nothing at all."""
    uv = tmp_path / "uv"
    if managed is None:
        uv.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    else:
        uv.write_text(
            f'#!/bin/sh\n[ "$1 $2 $3" = "python find --managed-python" ] && echo "{managed}"\n',
            encoding="utf-8",
        )
    uv.chmod(0o755)
    return uv


def _fake_python(path: Path) -> Path:
    """A spawnable interpreter standing in for a managed one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    real = Path(sys.executable)
    path.write_text(f'#!/bin/sh\nexec "{real}" "$@"\n', encoding="utf-8")
    path.chmod(0o755)
    return path


def test_bootstrap_resolves_a_managed_interpreter_when_the_ambient_one_is_too_old(tmp_path, monkeypatch):
    """The ambient interpreter is the wrong minor: a managed one must replace it."""
    from pm.runtime import _bootstrap_python

    monkeypatch.setattr(sys, "version_info", (3, 11, 15, "final", 0))
    managed = _fake_python(tmp_path / "managed-python")
    uv = _fake_uv(tmp_path, managed)

    resolved = _bootstrap_python(uv)
    assert resolved == managed, "a managed interpreter of the pinned minor must win"
    assert resolved != Path(sys.executable)


def test_bootstrap_keeps_the_ambient_interpreter_when_it_has_the_pinned_minor(tmp_path, monkeypatch):
    """A host whose base venv already has the pinned minor needs no lookup."""
    from pm.runtime import _pinned_python_minor, _bootstrap_python

    monkeypatch.setattr(sys, "version_info", (3, 14, 9, "final", 0))
    uv = _fake_uv(tmp_path, None)

    resolved = _bootstrap_python(uv)
    assert resolved == Path(sys.executable)
    assert _pinned_python_minor("3.14.9+20251021") == "3.14"


def test_bootstrap_falls_back_to_the_ambient_interpreter_when_nothing_else_is_there(tmp_path, monkeypatch):
    """With neither a store entry nor a uv-managed one, the ambient one stands.

    Better than refusing outright: the ambient interpreter may already satisfy
    the lock, and the stage that follows reports a real diagnosis if it does not.
    """
    from pm.runtime import _bootstrap_python

    monkeypatch.setattr(sys, "version_info", (3, 11, 15, "final", 0))
    uv = _fake_uv(tmp_path, None)

    resolved = _bootstrap_python(uv)
    assert resolved == Path(sys.executable)


def test_bootstrap_prefers_the_pinned_store_interpreter(tmp_path, monkeypatch):
    """PM's own pinned interpreter, when the store still holds it, comes first."""
    from pm.lock import Lockfile
    from pm.paths import lockfile_path
    from pm.registry import get_package
    from pm.runtime import _bootstrap_python
    from pm.store import current_target

    monkeypatch.setattr(sys, "version_info", (3, 11, 15, "final", 0))
    version = Lockfile(lockfile_path()).version("python")
    assert version, "the PM lock pins an interpreter"

    store = tmp_path / "store"
    entry = store / get_package("python").store_entry(version, current_target())
    binary = entry / "bin" / "python3"
    binary.parent.mkdir(parents=True)
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    binary.chmod(0o755)

    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    resolved = _bootstrap_python(_fake_uv(tmp_path, None))
    assert resolved == binary, "the pinned managed interpreter must win"


def test_the_ambient_interpreter_is_refused_by_the_runtime_lock(tmp_path, monkeypatch):
    """The failure this fixes, stated as a fact about the lock itself.

    PM's runtime graph requires a Python minor the ambient interpreter on a
    stale base venv does not provide, so uv refuses the lock.
    """
    repo = Path(__file__).resolve().parents[2]
    lock = (repo / "pm" / "uv.lock").read_text(encoding="utf-8")
    assert 'requires-python = "==3.14.*"' in lock


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell wrappers")
def test_runtime_python_never_pairs_the_staged_uv_with_an_incompatible_ambient(tmp_path, monkeypatch):
    """The regression itself: the fallback must not pair uv with the ambient 3.11.

    ``runtime_python`` falls back to the staged ``uv`` when PM's facts do not
    vouch for the pinned toolchain. That fallback is where the ambient
    interpreter used to leak in.
    """
    from pm import runtime

    monkeypatch.setattr(sys, "version_info", (3, 11, 15, "final", 0))
    managed = _fake_python(tmp_path / "managed-python")

    from pm.lock import Lockfile
    from pm.paths import lockfile_path
    from pm.registry import get_package
    from pm.store import current_target

    version = Lockfile(lockfile_path()).version("uv")
    assert version, "the PM lock pins uv"
    entry = tmp_path / "store" / get_package("uv").store_entry(version, current_target())
    entry.mkdir(parents=True, exist_ok=True)
    staged = entry / "uv"
    staged.write_text(
        f'#!/bin/sh\n[ "$1 $2 $3" = "python find --managed-python" ] && echo "{managed}"\nexit 0\n',
        encoding="utf-8",
    )
    staged.chmod(0o755)

    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    captured: dict[str, object] = {}

    def prepare(uv_path, python_path, root, **kwargs):
        captured["uv"] = uv_path
        captured["python"] = python_path
        raise SystemExit(0)

    monkeypatch.setattr(runtime, "prepare_runtime", prepare)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: None)

    with pytest.raises(SystemExit):
        runtime.runtime_python()
    assert captured["python"] == managed, "the staged uv must not carry the ambient interpreter"
