"""Live Windows proof for PM under ``LongPathsEnabled = 0`` (#130242, #130232).

Runs ONLY on a real Windows host whose registry value
``HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\\LongPathsEnabled`` is 0
(the Windows default). The on-demand ``wine2e/**`` lane
(``.github/workflows/windows-venv-e2e.yml``) sets it in an earlier step, so this
test's process starts with long paths off, and selects the file with
``-m integration`` (the marker keeps it out of the ordinary OS lane, whose runner
has long paths on and could not make the claim).

Nothing is mocked: a ``Store`` rooted under a temp directory drives the real
scratch / copy / digest / publish / removal path a bundled-copy install takes
(``pm.install._copy_verified_source`` plus ``_install``), over a tree whose files
and one directory lie beyond MAX_PATH. The test reaches its own fixtures through
``\\\\?\\`` spellings it builds itself, so a check never trusts the very calls
under test (plain ``Path.exists`` silently answers False beyond 260 characters).
"""

from __future__ import annotations

import json
import ntpath
import os
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.platforms("windows"),  # live long-path behaviour exists only on Windows
    # Requires a host configured with long paths off; the wine2e lane selects it.
    pytest.mark.integration,
]

MAX_PATH = 260
VERBATIM = "\\\\?\\"
SEGMENT = "s" * 60


def _long_paths_enabled() -> bool:
    import ctypes

    ntdll = ctypes.WinDLL("ntdll")
    ntdll.RtlAreLongPathsEnabled.restype = ctypes.c_ubyte
    return bool(ntdll.RtlAreLongPathsEnabled())


def _verbatim(path: Path | str) -> Path:
    """The test's own independent spelling for checks beyond MAX_PATH."""
    text = ntpath.abspath(os.fspath(path))
    return Path(text if text.startswith(VERBATIM) else VERBATIM + text)


def _plain(path: Path | str) -> str:
    text = os.fspath(path)
    return text[len(VERBATIM):] if text.startswith(VERBATIM) else text


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _build_tree(entry: Path, plain_base_length: int) -> dict[str, bytes]:
    """Files beyond MAX_PATH, one directory beyond MAX_PATH, and a short binary.

    ``plain_base_length`` is the plain length of the shortest root the tree
    will sit under, so every deep path exceeds MAX_PATH wherever it lands.
    """
    deep = ""
    while plain_base_length + len(deep) <= MAX_PATH + 10:
        deep = f"{deep}/{SEGMENT}" if deep else SEGMENT
    files = {
        "bin/tool.exe": b"not a real binary",
        "short.txt": b"short",
        # File whose full path crosses MAX_PATH inside a directory that does not.
        f"{deep.rsplit('/', 1)[0]}/{'f' * 80}.txt": b"file beyond MAX_PATH",
        # Directory whose own path is beyond MAX_PATH, holding two files.
        f"{deep}/inner/a.txt": b"inside a directory beyond MAX_PATH",
        f"{deep}/inner/b.py": b"print('deep')\n",
    }
    for rel, data in files.items():
        target = _verbatim(entry) / Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return files


def _assert_tree(entry: Path, files: dict[str, bytes]) -> None:
    root = _verbatim(entry)
    for rel, data in files.items():
        assert (root / Path(rel)).read_bytes() == data, rel
    found = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert found == sorted(files)


def test_store_lifecycle_beyond_max_path_with_long_paths_disabled(tmp_path):
    # A green run must not be a false proof: long paths really are off here.
    assert sys.platform == "win32"
    assert not _long_paths_enabled(), (
        "LongPathsEnabled is on for this process; set HKLM\\SYSTEM\\CurrentControlSet\\"
        "Control\\FileSystem LongPathsEnabled=0 before the interpreter starts")
    probe = _verbatim(tmp_path) / ("p" * 120) / ("q" * 120) / "probe.txt"
    probe.parent.mkdir(parents=True)
    probe.write_bytes(b"x")
    assert len(_plain(probe)) > MAX_PATH
    with pytest.raises(OSError):
        os.stat(_plain(probe))  # the host really cuts plain spellings at MAX_PATH
    shutil.rmtree(_verbatim(tmp_path) / ("p" * 120))

    from pm.install import _remove_entry
    from pm.lock import Facts
    from pm.package import Package, compose_env
    from pm.store import Store, current_target, tree_digest

    class Tool(Package):
        name = "longpath-probe"

        def binary(self, entry: Path, target: str) -> Path:
            return entry / "bin" / "tool.exe"

    package, target, version = Tool(), current_target(), "1.0.0"
    entry_name = package.store_entry(version, target)
    source_root, store_root = tmp_path / "src", tmp_path / "dst"
    source, store = Store(source_root), Store(store_root)

    files = _build_tree(source_root / entry_name,
                        len(_plain(tmp_path / "dst" / entry_name)))
    expected = tree_digest(_verbatim(source_root / entry_name))

    with store.scratch() as scratch:
        staged = scratch / "staged"
        # The copy _copy_verified_source performs, then its digest check.
        shutil.copytree(source.entry(entry_name), staged, symlinks=True)
        assert tree_digest(staged) == expected
        assert tree_digest(source.entry(entry_name)) == expected
        _assert_tree(staged, files)
        # Bytes scratch() must clean up itself (an abandoned extraction).
        abandoned = scratch / "abandoned"
        shutil.copytree(source.entry(entry_name), abandoned, symlinks=True)
        published = store.publish(staged, entry_name)
    assert not _verbatim(scratch).exists(), "scratch() left bytes behind beyond MAX_PATH"
    assert not [p for p in _verbatim(store_root).iterdir() if p.name.startswith(".staging-")]

    _assert_tree(store_root / entry_name, files)
    assert tree_digest(published) == expected

    # Records and child environments are exits: they carry the ordinary spelling.
    facts = Facts(tmp_path / "facts.json")
    facts.record(package.name, version, entry_name, package.env(published, target), store.root,
                 target=target, artifacts=["0" * 64], digest=tree_digest(published))
    assert facts.installed(package.name, version, store.root, (target, ("0" * 64,)))
    on_disk = json.loads((tmp_path / "facts.json").read_text(encoding="utf-8"))
    assert not [s for s in _strings(on_disk) if VERBATIM in s]
    child_env = compose_env([facts.env_for(package.name, store.root)], base={})
    assert not [v for v in child_env.values() if VERBATIM in v], child_env
    assert ntpath.normcase(child_env["PATH"]) == ntpath.normcase(
        _plain(store_root / entry_name / "bin"))

    _remove_entry(store, entry_name)
    assert not _verbatim(store_root / entry_name).exists(), "removal left bytes beyond MAX_PATH"
