"""Baked bytecode: staging contract and launcher marker behavior."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.bundles.bytecode import MARKER, bake_bytecode

_LIB = f"lib/python{sys.version_info.major}.{sys.version_info.minor}"


def _make_payload(root: Path) -> Path:
    (root / "hermes-agent" / "pkg").mkdir(parents=True)
    (root / "hermes-agent" / "pkg" / "__init__.py").write_text("")
    (root / "hermes-agent" / "pkg" / "mod.py").write_text("X = 1\n")
    site = root / "venv" / _LIB / "site-packages"
    site.mkdir(parents=True)
    (site / "dep.py").write_text("Y = 2\n")
    pm = root / "pm-runtime" / _LIB / "site-packages"
    pm.mkdir(parents=True)
    (pm / "pmdep.py").write_text("Z = 3\n")
    stdlib = root / "tools" / "python-staged" / _LIB
    stdlib.mkdir(parents=True)
    (stdlib / "os.py").write_text("# the landmark CPython finds its standard library by\n")
    (stdlib / "stdmod.py").write_text("W = 4\n")
    return root


def _staged_python(root: Path) -> Path:
    """The payload's interpreter where PM stages it (bin/python3 in the tool entry)."""
    python = root / "tools" / "python-staged" / "bin" / "python3"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n")
    python.chmod(0o755)
    return python


def _files(root: Path) -> dict[str, tuple[int, int]]:
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns, path.stat().st_size)
            for path in root.rglob("*") if path.is_file()}


def _pyc_flags(source: Path) -> int | None:
    """PEP 552 flags of the source's pyc: 0 timestamp, 1 unchecked-hash, 3 checked-hash."""
    from importlib.util import cache_from_source
    pyc = Path(cache_from_source(source))
    return int.from_bytes(pyc.read_bytes()[4:8], "little") if pyc.is_file() else None


@pytest.mark.platforms("posix")
def test_bake_produces_owner_writable_unchecked_hash_pycs(tmp_path):
    root = _make_payload(tmp_path)
    result = bake_bytecode(root, _staged_python(root))
    assert (root / MARKER).read_text().strip() == "unchecked-hash"
    pyc = next((root / "hermes-agent" / "pkg" / "__pycache__").glob("mod*.pyc"))
    # PEP 552 header, little-endian flags at bytes 4..8: value 1 =
    # hash-based (bit0) and unchecked (bit1 clear). Timestamp pycs would be 0;
    # checked-hash would be 3.
    assert pyc.read_bytes()[4:8] == b"\x01\x00\x00\x00"
    # owner write before packaging: Squirrel.Mac's ShipIt clears the
    # quarantine attribute of every bundled file and aborts the whole
    # macOS update with EACCES on any read-only one (#138659)
    assert pyc.stat().st_mode & 0o200
    assert result["modules"] >= 3


@pytest.mark.platforms("posix")
def test_bake_turns_every_existing_timestamp_pyc_into_unchecked_hash(tmp_path):
    """Timestamp pycs already in the tree when the bake runs (uv's --compile-bytecode in
    both site-packages, the build's own stdlib imports) ship unchecked-hash too, and the
    staged interpreter's standard library is baked like the agent tree."""
    root = _make_payload(tmp_path)
    venv_dep = root / "venv" / _LIB / "site-packages" / "dep.py"
    stdlib = root / "tools" / "python-staged" / _LIB
    for source in (venv_dep, stdlib / "stdmod.py"):
        subprocess.run([sys.executable, "-I", "-m", "compileall", "-q", "--invalidation-mode", "timestamp",
                        str(source)], check=True)
    bake_bytecode(root, _staged_python(root))
    flags = {name: _pyc_flags(source) for name, source in {
        "venv dep.py": venv_dep, "stdlib os.py": stdlib / "os.py", "stdlib stdmod.py": stdlib / "stdmod.py"}.items()}
    assert flags == {"venv dep.py": 1, "stdlib os.py": 1, "stdlib stdmod.py": 1}


@pytest.mark.platforms("posix")
def test_bare_payload_python_writes_nothing_after_packaging_moves_mtimes(tmp_path):
    """The sealed-payload contract: a payload Python run without a cache prefix (Desktop's
    ticket bridge) imports from every root after packaging rewrote the sources' mtimes, and
    the payload stays as built. A macOS app that rewrites its own pycs fails its code
    signature, and Gatekeeper then holds its next update behind a prompt."""
    root = _make_payload(tmp_path)
    python = _staged_python(root)
    for site in (root / "venv" / _LIB / "site-packages", root / "pm-runtime" / _LIB / "site-packages"):
        subprocess.run([sys.executable, "-I", "-m", "compileall", "-q", "--invalidation-mode", "timestamp",
                        str(site)], check=True)
    bake_bytecode(root, python)
    later = int(os.stat(root / MARKER).st_mtime) + 86_400
    for source in root.rglob("*.py"):
        os.utime(source, (later, later))
    before = _files(root)
    roots = [root / "hermes-agent", root / "venv" / _LIB / "site-packages",
             root / "pm-runtime" / _LIB / "site-packages", root / "tools" / "python-staged" / _LIB]
    code = f"import sys; sys.path.extend({[str(r) for r in roots]!r}); import pkg.mod, dep, pmdep, stdmod"
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    subprocess.run([str(python), "-P", "-c", code], check=True, env=env, cwd=tmp_path)
    assert _files(root) == before


@pytest.mark.platforms("posix")
def test_bake_refuses_an_interpreter_library_outside_the_payload(tmp_path):
    """The bake never writes into an interpreter the payload does not ship."""
    root = _make_payload(tmp_path / "payload")
    outside = tmp_path / "host-python-lib"
    outside.mkdir()
    (outside / "os.py").write_text("")
    staged = root / "tools" / "python-staged" / _LIB
    for path in staged.iterdir():
        path.unlink()
    staged.rmdir()
    staged.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the payload"):
        bake_bytecode(root, _staged_python(root))
    assert not (outside / "__pycache__").exists()


@pytest.mark.platforms("posix")
def test_bake_fails_closed_on_uncompilable_module(tmp_path):
    """Strict compileall: an unparseable module fails the whole bake instead of
    silently shipping cold-compile-every-launch bytecode."""
    import subprocess as _sp
    root = _make_payload(tmp_path)
    (root / "hermes-agent" / "pkg" / "broken.py").write_text("this is (( not python\n")
    with pytest.raises(_sp.CalledProcessError):
        bake_bytecode(root, _staged_python(root))


@pytest.mark.platforms("posix")
def test_uncovered_reports_parseable_module_without_pyc(tmp_path):
    """The coverage gate: a parseable module lacking bytecode is reported
    before it can ship as a cold-compile stall."""
    from importlib.util import cache_from_source
    root = _make_payload(tmp_path)
    bake_bytecode(root, _staged_python(root))
    from scripts.bundles.bytecode import _uncovered
    missing, unparseable = _uncovered(root / "hermes-agent")
    assert missing == [] and unparseable == 0
    pyc = Path(cache_from_source(root / "hermes-agent" / "pkg" / "mod.py"))
    pyc.unlink()
    missing, _ = _uncovered(root / "hermes-agent")
    assert [m.name for m in missing] == ["mod.py"]


def test_bake_rejects_missing_import_root(tmp_path):
    (tmp_path / "venv").mkdir()
    with pytest.raises(FileNotFoundError):
        bake_bytecode(tmp_path, Path(sys.executable))


def _load_wrapper():
    """Import the wrapper with placeholders substituted, like the build does."""
    from scripts.build.launchers import render_wrapper

    text = render_wrapper("stubmod.entry:main", "../hermes-agent", "../venv/Lib/site-packages")
    namespace: dict = {"__name__": "launcher_wrapper_under_test"}
    wrapper = Path("scripts/build/launcher_wrapper.py")
    exec(compile(text, str(wrapper), "exec"), namespace)
    return namespace


def test_launcher_skips_user_cache_redirect_when_marker_present(tmp_path):
    ns = _load_wrapper()
    payload = _make_payload(tmp_path)
    (payload / MARKER).write_text("unchecked-hash\n")
    here = payload / "bin"
    here.mkdir()
    environ = {"HOME": str(tmp_path / "userhome")}
    original = sys.pycache_prefix
    try:
        ns["configure"](str(here), environ=environ)
        # Baked payload: no prefix redirect — imports read the baked
        # source-adjacent dirs (Python's default lookup), and the prefix
        # would relocate those READS away from the payload.
        assert "PYTHONPYCACHEPREFIX" not in environ
    finally:
        sys.pycache_prefix = original


def test_launcher_keeps_user_cache_redirect_without_marker(tmp_path):
    ns = _load_wrapper()
    payload = _make_payload(tmp_path)
    here = payload / "bin"
    here.mkdir()
    environ = {"HOME": str(tmp_path / "userhome")}
    original = sys.pycache_prefix
    try:
        ns["configure"](str(here), environ=environ)
        assert environ["PYTHONPYCACHEPREFIX"] == str(
            tmp_path / "userhome" / ".cache" / "hermes-pycache")
    finally:
        sys.pycache_prefix = original
