"""Behaviour of the shared Node sidecar resolver.

The resolver answers one question: which directory does a Node sidecar run
from? A sidecar needs its node_modules, and Node's ESM resolver only looks in
ancestor directories of the importing file, so the entry file and the deps
must live in one tree. NODE_PATH does not work for ESM, and both in-tree
sidecars are "type": "module".

That constraint is why a read-only install tree with no baked deps has to copy
the sidecar somewhere writable. Nothing else Node offers gets the two into the
same tree.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gateway.sidecar_runtime import resolve_sidecar


def _make_sidecar(root: Path, *, files=("index.mjs", "helper.mjs")) -> Path:
    """A sidecar source dir with a lockfile and a couple of ESM helpers."""
    src = root / "src"
    src.mkdir(parents=True)
    (src / "package.json").write_text(json.dumps({"name": "sc", "type": "module"}))
    (src / "package-lock.json").write_text(json.dumps({"lockfileVersion": 3}))
    for name in files:
        (src / name).write_text(f"// {name}\n")
    return src


def _install_deps(src: Path, *, current: bool = True) -> None:
    """Give the sidecar a node_modules with npm's install marker."""
    nm = src / "node_modules"
    (nm / "somepkg").mkdir(parents=True)
    (nm / "somepkg" / "index.js").write_text("module.exports = 1;\n")
    marker = nm / ".package-lock.json"
    marker.write_text("{}")
    lock = src / "package-lock.json"
    if current:
        # Marker newer than the lockfile: the install matches.
        os.utime(marker, (lock.stat().st_atime + 10, lock.stat().st_mtime + 10))
    else:
        os.utime(marker, (lock.stat().st_atime - 10, lock.stat().st_mtime - 10))


class TestRungOrder:
    """The four rungs, in the order the resolver must try them."""

    def test_env_override_is_used_as_is(self, tmp_path, monkeypatch):
        """An operator override wins over every other rung.

        It is the escape hatch for a layout Hermes cannot predict, so the
        resolver must not second-guess it.
        """
        src = _make_sidecar(tmp_path)
        override = tmp_path / "elsewhere"
        override.mkdir()
        monkeypatch.setenv("HERMES_TESTSC_SIDECAR_DIR", str(override))
        assert resolve_sidecar("testsc", src) == override

    def test_writable_source_runs_in_place(self, tmp_path, monkeypatch):
        """A dev checkout installs and runs in the source tree."""
        monkeypatch.delenv("HERMES_TESTSC_SIDECAR_DIR", raising=False)
        src = _make_sidecar(tmp_path)
        assert resolve_sidecar("testsc", src) == src

    def test_readonly_source_with_current_deps_runs_in_place(
        self, tmp_path, monkeypatch
    ):
        """Baked deps that match the lockfile need no writable directory.

        This is the Nix store and the built container image. The sidecar
        never writes inside its own directory, so read-only is fine.
        """
        monkeypatch.delenv("HERMES_TESTSC_SIDECAR_DIR", raising=False)
        src = _make_sidecar(tmp_path)
        _install_deps(src, current=True)
        monkeypatch.setattr(
            "gateway.sidecar_runtime.dir_writable", lambda p: False
        )
        assert resolve_sidecar("testsc", src) == src

    def test_readonly_source_with_stale_deps_mirrors(self, tmp_path, monkeypatch):
        """A lockfile newer than the install means the deps must be rebuilt.

        The source tree cannot take the write, so the sidecar moves to the
        durable directory where npm can run.
        """
        monkeypatch.delenv("HERMES_TESTSC_SIDECAR_DIR", raising=False)
        src = _make_sidecar(tmp_path)
        _install_deps(src, current=False)
        home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(
            "gateway.sidecar_runtime.dir_writable", lambda p: False
        )
        got = resolve_sidecar("testsc", src)
        assert got != src
        assert got == home / "sidecars" / "testsc"

    def test_readonly_source_with_no_deps_mirrors(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_TESTSC_SIDECAR_DIR", raising=False)
        src = _make_sidecar(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setattr(
            "gateway.sidecar_runtime.dir_writable", lambda p: False
        )
        assert resolve_sidecar("testsc", src) != src


class TestMirrorCarriesTheWholeSidecar:
    """The mirror must be complete. A manifest of files is not enough.

    The Photon resolver kept a list of files to copy. The list drifted twice:
    it named a file that had been deleted, and it omitted two helpers that
    index.mjs imports. Both faults appear only on a read-only install, which
    is the one place nobody runs by hand.
    """

    @pytest.fixture
    def mirrored(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_TESTSC_SIDECAR_DIR", raising=False)
        src = _make_sidecar(
            tmp_path, files=("index.mjs", "helper.mjs", "deep-helper.mjs")
        )
        (src / "sub").mkdir()
        (src / "sub" / "nested.mjs").write_text("// nested\n")
        _install_deps(src, current=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setattr(
            "gateway.sidecar_runtime.dir_writable", lambda p: False
        )
        return src, resolve_sidecar("testsc", src)

    def test_every_source_file_reaches_the_mirror(self, mirrored):
        """Whatever the source holds, the mirror holds.

        Asserted against the source tree itself, not against a second list,
        so a new helper cannot be forgotten.
        """
        src, mirror = mirrored
        want = {
            p.relative_to(src)
            for p in src.rglob("*")
            if p.is_file() and "node_modules" not in p.parts
        }
        got = {
            p.relative_to(mirror)
            for p in mirror.rglob("*")
            if p.is_file() and "node_modules" not in p.parts
        }
        assert want <= got, f"missing from the mirror: {sorted(want - got)}"

    def test_node_modules_is_not_copied(self, mirrored):
        """npm owns the mirror's deps. Copying them wastes time and space."""
        _src, mirror = mirrored
        assert not (mirror / "node_modules" / "somepkg").exists()


class TestRefresh:
    """An image update changes a source file. The mirror must follow."""

    def _mirror_twice(self, tmp_path, monkeypatch, mutate):
        monkeypatch.delenv("HERMES_TESTSC_SIDECAR_DIR", raising=False)
        src = _make_sidecar(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setattr(
            "gateway.sidecar_runtime.dir_writable", lambda p: False
        )
        mirror = resolve_sidecar("testsc", src)
        # Deps installed in the mirror by npm — these must survive a refresh.
        (mirror / "node_modules" / "somepkg").mkdir(parents=True)
        (mirror / "node_modules" / "somepkg" / "index.js").write_text("1")
        mutate(src)
        return src, resolve_sidecar("testsc", src)

    def test_a_changed_source_file_is_recopied(self, tmp_path, monkeypatch):
        def mutate(src):
            (src / "index.mjs").write_text("// version two\n")

        _src, mirror = self._mirror_twice(tmp_path, monkeypatch, mutate)
        assert (mirror / "index.mjs").read_text() == "// version two\n"

    def test_a_new_source_file_appears(self, tmp_path, monkeypatch):
        def mutate(src):
            (src / "brand-new.mjs").write_text("// new helper\n")

        _src, mirror = self._mirror_twice(tmp_path, monkeypatch, mutate)
        assert (mirror / "brand-new.mjs").exists()

    def test_the_mirrors_node_modules_survives(self, tmp_path, monkeypatch):
        """A refresh must not delete the deps npm installed in the mirror.

        Wiping them turns every image update into a reinstall, on the
        installs least able to afford one.
        """
        def mutate(src):
            (src / "index.mjs").write_text("// version two\n")

        _src, mirror = self._mirror_twice(tmp_path, monkeypatch, mutate)
        assert (mirror / "node_modules" / "somepkg" / "index.js").exists()


class TestFailureIsNotFatal:
    def test_an_unwritable_mirror_falls_back_to_the_source(
        self, tmp_path, monkeypatch
    ):
        """Returning the read-only source loses installs, not the process.

        The caller's own readiness check then reports the real error, which
        is more use than a traceback out of a path resolver.
        """
        monkeypatch.delenv("HERMES_TESTSC_SIDECAR_DIR", raising=False)
        src = _make_sidecar(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setattr(
            "gateway.sidecar_runtime.dir_writable", lambda p: False
        )

        def _boom(*a, **k):
            raise OSError("read-only file system")

        monkeypatch.setattr("gateway.sidecar_runtime.shutil.copytree", _boom)
        assert resolve_sidecar("testsc", src) == src


class TestEpochMtimeSources:
    """A Nix store source has mtime = epoch on every file.

    copy2 would stamp the mirror's lockfile with that epoch mtime, so npm's
    install marker (stamped at install time) always postdates it and
    deps_are_current() never reports stale — the exact WhatsApp keep-old-
    node_modules bug this resolver exists to fix. The copy must stamp "now".
    """

    def test_a_content_change_with_epoch_mtimes_marks_deps_stale(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("HERMES_TESTSC_SIDECAR_DIR", raising=False)
        src = _make_sidecar(tmp_path)
        lock = src / "package-lock.json"
        for p in src.rglob("*"):
            os.utime(p, (1, 1))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setattr(
            "gateway.sidecar_runtime.dir_writable", lambda p: False
        )

        mirror = resolve_sidecar("testsc", src)
        # npm install in the mirror: marker stamped now.
        nm = mirror / "node_modules"
        nm.mkdir()
        (nm / ".package-lock.json").write_text("{}")

        # Upgrade: lockfile CONTENT changes, mtime stays epoch (new store path).
        lock.write_text(json.dumps({"lockfileVersion": 3, "v": 2}))
        os.utime(lock, (2, 2))

        mirror = resolve_sidecar("testsc", src)
        from gateway.sidecar_runtime import deps_are_current

        assert deps_are_current(mirror) is False, (
            "a refreshed lockfile must postdate the previous npm install, "
            "or an upgrade keeps stale node_modules forever"
        )
