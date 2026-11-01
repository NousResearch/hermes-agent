"""The bionic interpreter pin and the Termux wheel target name one version.

``pm/lock.json`` pins the interpreter the bionic payload ships; the wheelhouse
build scripts describe that same interpreter in ``TARGET_ENV`` so dependency
markers and wheel tags resolve against it. The Termux pool is rolling, so the
pin moves whenever Termux rebuilds python -- and a repin that leaves
``TARGET_ENV`` behind resolves wheels for an interpreter the payload no longer
carries. #141465 is that drift seen from the other side: the pin named an
archive the pool had already deleted, so the rebuild 404'd before anything
else could run.

The values themselves are data (bumped by review); these tests pin the
relationship, and a ``curl -sI`` of the pinned URL is the evidence that the
pool still serves it.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_BIONIC_DEB = re.compile(
    r"/p/python/python_(?P<ver>[0-9]+\.[0-9]+\.[0-9]+)(?:-[0-9]+)?_aarch64\.deb$"
)
_TARGET_ENV_VERSION = re.compile(
    r'"(?:python_full_version|implementation_version)":\s*"([0-9][^"]*)"'
)


def _pinned_interpreter(lock_path: Path) -> str:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    url = lock["packages"]["python"]["artifacts"]["linux-arm64-bionic"]["url"]
    match = _BIONIC_DEB.search(url)
    assert match, f"the python bionic pin does not name a pool .deb: {url}"
    return match.group("ver")


def _load_build_wheels():
    path = REPO_ROOT / "scripts" / "termux" / "build_wheels.py"
    spec = importlib.util.spec_from_file_location("termux_build_wheels", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wheel_target_env_matches_the_pinned_interpreter():
    pinned = _pinned_interpreter(REPO_ROOT / "pm" / "lock.json")
    env = _load_build_wheels().TARGET_ENV
    assert env["python_full_version"] == pinned, (
        "build_wheels TARGET_ENV resolves wheels for "
        f"{env['python_full_version']}, the bionic payload pins {pinned}"
    )
    assert env["implementation_version"] == pinned
    assert env["python_version"] == ".".join(pinned.split(".")[:2])


def test_shell_wheel_target_env_matches_the_pinned_interpreter():
    pinned = _pinned_interpreter(REPO_ROOT / "pm" / "lock.json")
    script = (REPO_ROOT / "scripts" / "termux" / "termux_build.sh").read_text(encoding="utf-8")
    found = _TARGET_ENV_VERSION.findall(script)
    assert found, "termux_build.sh carries no TARGET_ENV version"
    assert set(found) == {pinned}, (
        f"termux_build.sh resolves wheels for {sorted(set(found))}, "
        f"the bionic payload pins {pinned}"
    )


def test_python_fetch_url_reproduces_the_pinned_row():
    """The class and the lock are two spellings of one pin: `pm lock --bump`
    reads the class, the install path reads the row."""
    from pm.registry import get_package

    lock = json.loads((REPO_ROOT / "pm" / "lock.json").read_text(encoding="utf-8"))
    row = lock["packages"]["python"]["artifacts"]["linux-arm64-bionic"]
    package = get_package("python")
    assert package.fetch_url(lock["packages"]["python"]["version"], "linux-arm64-bionic") == row["url"]
