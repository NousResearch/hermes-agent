"""Real hard-link reclamation regressions for #129300 and #129307."""
import json
import os
import shutil

import pytest


def _hardlink_fixture(tmp_path, collector, root_name, shared_location="nested"):
    """One selected generation, one busy one, and old generations whose payload
    hard-links a single shared file standing in for a mapped image."""
    from hermes_cli.runtime_state import collect_generations
    from pm.environments import install_state_dir, runtime_facts_path, site_packages
    from pm.runtime import collect_runtime_generations

    repo = tmp_path / root_name / "repo"
    repo.mkdir(parents=True)
    root = install_state_dir(repo) if collector == "dependencies" else tmp_path / root_name / "pm-runtime"
    generations = root / ("environments" if collector == "dependencies" else "generations")
    shared = tmp_path / root_name / "cached.pyd"
    shared.write_bytes(b"shared extension")
    inode = (shared.stat().st_dev, shared.stat().st_ino)

    def make(name):
        generation = generations / name
        if collector == "dependencies":
            venv = generation / "venv"
            venv.mkdir(parents=True)
            (venv / "pyvenv.cfg").write_text("version = 3.11")
            payload = site_packages(venv)
        else:
            payload = generation / "site-packages"
            generation.mkdir(parents=True)
            (generation / "pm-runtime.json").write_text(json.dumps({"inputs": name}))
        payload.mkdir(parents=True)
        os.link(shared, (payload if shared_location == "nested" else generation) / "locked.pyd")
        (payload / "unique.bin").write_bytes(name.encode() * 1024)
        (generation / ".lease-managed").touch()
        return generation, payload

    selected, selected_payload = make("selected")
    busy, _ = make("busy")
    old = [make(f"old-{i}") for i in range(2)]
    if collector == "dependencies":
        selection = runtime_facts_path(repo)
        selection.write_text(json.dumps({"packages": {"venv": {"environment": str(selected / "venv")}}}))
        collect = lambda: collect_generations(repo, min_age_seconds=0)
    else:
        selection = root / "selected.json"
        selection.write_text(json.dumps({"generation": "generations/selected"}))
        collect = lambda: collect_runtime_generations(root)
    return shared, inode, busy, collect, selection, selected_payload, old


@pytest.mark.parametrize("collector", ["dependencies", "pm-runtime"])
@pytest.mark.parametrize("fd_walk", [False, True], ids=["path-walk", "fd-walk"])
@pytest.mark.parametrize("shared_location", ["nested", "root"])
def test_collectors_reclaim_payload_around_shared_locked_inode(
    tmp_path, monkeypatch, collector, fd_walk, shared_location
):
    from hermes_cli.runtime_state import collect_generations, lease_directory
    from pm.environments import install_state_dir, runtime_facts_path, site_packages
    from pm.runtime import collect_runtime_generations

    if fd_walk and not shutil.rmtree.avoids_symlink_attacks:
        pytest.skip("fd-based rmtree unavailable on this host")
    # Exercise both stdlib walkers, including the path-based walker used on Windows.
    if hasattr(shutil, "_rmtree_impl"):
        monkeypatch.setattr(shutil, "_rmtree_impl", shutil._rmtree_safe_fd if fd_walk else shutil._rmtree_unsafe)
    else:
        monkeypatch.setattr(shutil, "_use_fd_functions", fd_walk)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    root = install_state_dir(repo) if collector == "dependencies" else tmp_path / "pm-runtime"
    generations = root / ("environments" if collector == "dependencies" else "generations")
    shared = tmp_path / "cached.pyd"
    shared.write_bytes(b"shared extension")
    inode = (shared.stat().st_dev, shared.stat().st_ino)

    def make(name, managed=True):
        generation = generations / name
        if collector == "dependencies":
            venv = generation / "venv"
            venv.mkdir(parents=True)
            (venv / "pyvenv.cfg").write_text("version = 3.11")
            payload = site_packages(venv)
        else:
            payload = generation / "site-packages"
            generation.mkdir(parents=True)
            (generation / "pm-runtime.json").write_text(json.dumps({"inputs": name}))
        payload.mkdir(parents=True)
        os.link(shared, (payload if shared_location == "nested" else generation) / "locked.pyd")
        (payload / "unique.bin").write_bytes(name.encode() * 1024)
        # Root-level payload must also be reclaimed, regardless of marker iteration order.
        (generation / "unique-root.bin").write_bytes(name.encode() * 1024)
        if managed:
            (generation / ".lease-managed").touch()
        return generation, payload

    selected, selected_payload = make("selected")
    busy, busy_payload = make("busy")
    legacy, legacy_payload = make("legacy", managed=False)
    old = [make(f"old-{i}") for i in range(10)]
    def locked_file(generation, payload):
        return (payload if shared_location == "nested" else generation) / "locked.pyd"

    assert shared.stat().st_nlink == len(old) + 4
    assert all(locked_file(generation, payload).stat().st_ino == shared.stat().st_ino for generation, payload in old)
    if collector == "dependencies":
        selection = runtime_facts_path(repo)
        selection.write_text(json.dumps({"packages": {"venv": {"environment": str(selected / "venv")}}}))
        collect = lambda: collect_generations(repo, min_age_seconds=0)
        assert collect_generations(repo) == []  # Young generations retain the existing grace period.
        assert all((payload / "unique.bin").is_file() for _, payload in old)
    else:
        selection = root / "selected.json"
        selection.write_text(json.dumps({"generation": "generations/selected"}))
        collect = lambda: collect_runtime_generations(root)
    selection_bytes = selection.read_bytes()
    metadata = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for generation, _ in old
        for path in generation.iterdir()
        if path.name in (".lease-managed", "pm-runtime.json")
    }
    protected = {
        path: path.read_bytes()
        for generation in (selected, busy, legacy)
        for path in generation.rglob("*") if path.is_file()
    }
    release = lease_directory(busy)
    real_unlink = os.unlink
    blocked = []

    def locked_unlink(path, *, dir_fd=None):
        info = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        if (info.st_dev, info.st_ino) == inode:
            blocked.append(path)
            raise PermissionError("mapped shared image")
        return real_unlink(path, dir_fd=dir_fd)

    try:
        with monkeypatch.context() as lock:
            lock.setattr(os, "unlink", locked_unlink)
            assert collect() == []  # Partial shells are not reported as removed generations.
            assert len(blocked) == len(old)
            assert all(not (payload / "unique.bin").exists() for _, payload in old)
            assert all(not (generation / "unique-root.bin").exists() for generation, _ in old)
            assert all(locked_file(generation, payload).read_bytes() == shared.read_bytes() for generation, payload in old)
            assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in metadata.items())
            # Retained markers still support real leases and later collection.
            release_old = lease_directory(old[0][0])
            try:
                blocked.clear()
                assert collect() == []
                assert len(blocked) == len(old) - 1
            finally:
                release_old()
        assert set(collect()) == {generation for generation, _ in old}
        assert all(not generation.exists() for generation, _ in old)
        assert shared.stat().st_nlink == 4
        assert selection.read_bytes() == selection_bytes
        assert all(path.read_bytes() == data for path, data in protected.items())
    finally:
        release()
    assert collect() == [busy]
    assert selected_payload.is_dir() and legacy_payload.is_dir()


@pytest.mark.parametrize("collector", ["dependencies", "pm-runtime"])
def test_posix_disposition_reclaims_what_the_unlink_refused(tmp_path, monkeypatch, collector):
    """Windows: a POSIX-semantics disposition (what MSYS ``rm -rf`` uses) strips a name
    the plain unlink refused while an image mapping persists, so collection reclaims
    the whole generation; a refused disposition keeps the existing defer behavior."""
    import sys

    from hermes_cli import runtime_state
    from hermes_cli.runtime_state import lease_directory

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    real_unlink = os.unlink

    def locked_unlink(inode):
        def refuse(path, *, dir_fd=None):
            info = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
            if (info.st_dev, info.st_ino) == inode:
                raise PermissionError("mapped shared image")
            return real_unlink(path, dir_fd=dir_fd)
        return refuse

    if sys.platform != "win32":
        # Direct, unpatched call: off Windows the helper is inert, never a hidden delete.
        inert_shared = _hardlink_fixture(tmp_path, collector, "inert")[0]
        assert runtime_state._unlink_by_posix_disposition(str(inert_shared)) is False
        assert inert_shared.is_file()

    # A successful disposition reclaims the refused entry and its whole generation.
    shared, inode, busy, collect, selection, selected_payload, old = _hardlink_fixture(tmp_path, collector, "reclaim")
    selection_bytes = selection.read_bytes()
    base_nlink = shared.stat().st_nlink
    release = lease_directory(busy)
    try:
        with monkeypatch.context() as lock:
            lock.setattr(os, "unlink", locked_unlink(inode))

            def strip_name(path):
                real_unlink(path)  # emulate the win32 call: the entry leaves the namespace
                return True

            lock.setattr(runtime_state, "_unlink_by_posix_disposition", strip_name)
            assert set(collect()) == {generation for generation, _ in old}
            assert all(not generation.exists() for generation, _ in old)
            assert shared.stat().st_nlink == base_nlink - len(old)
    finally:
        release()
    assert selection.read_bytes() == selection_bytes
    assert selected_payload.is_dir()

    # A refused disposition preserves today's defer behavior: shells, markers, leases.
    shared, inode, busy, collect, selection, selected_payload, old = _hardlink_fixture(tmp_path, collector, "defer")
    release = lease_directory(busy)
    try:
        with monkeypatch.context() as lock:
            lock.setattr(os, "unlink", locked_unlink(inode))
            lock.setattr(runtime_state, "_unlink_by_posix_disposition", lambda path: False)
            assert collect() == []
            assert all(generation.is_dir() for generation, _ in old)
            assert all((generation / ".lease-managed").is_file() for generation, _ in old)
            release_old = lease_directory(old[0][0])
            try:
                assert collect() == []  # retained shells still honor real leases
            finally:
                release_old()
        # With the refusal gone the retained shells collapse into complete removals.
        assert set(collect()) == {generation for generation, _ in old}
        assert selected_payload.is_dir()
    finally:
        release()
    assert collect() == [busy]


@pytest.mark.parametrize("collector", ["dependencies", "pm-runtime"])
@pytest.mark.parametrize("shared_location", ["nested", "root"])
def test_rename_fallback_defers_mapped_bytes_as_tombstones(tmp_path, monkeypatch, collector, shared_location):
    """Windows 10 19045 live data (#129307): every FileDispositionInfoEx flag
    variant is refused while an image section maps the file, and only a
    same-volume rename frees the name. The tombstone fallback must strip refused
    payload names into ``generation/.reclaimed`` — bytes queued, generation
    visibly incomplete — and a later pass finishes once the mapping is released."""
    import sys

    from hermes_cli import runtime_state
    from hermes_cli.runtime_state import lease_directory

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    real_unlink = os.unlink

    def locked_unlink(inode):
        def refuse(path, *, dir_fd=None):
            info = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
            if (info.st_dev, info.st_ino) == inode:
                raise PermissionError("mapped shared image")
            return real_unlink(path, dir_fd=dir_fd)
        return refuse

    def locked_name(generation, payload):
        return (payload if shared_location == "nested" else generation) / "locked.pyd"

    # Off Windows the fallback is inert: a refused unlink defers without renaming.
    shared, inode, busy, collect, selection, selected_payload, old = _hardlink_fixture(
        tmp_path, collector, "inert", shared_location
    )
    release = lease_directory(busy)
    try:
        with monkeypatch.context() as lock:
            lock.setattr(sys, "platform", "linux")
            lock.setattr(os, "unlink", locked_unlink(inode))
            lock.setattr(runtime_state, "_unlink_by_posix_disposition", lambda path: False)
            assert collect() == []
            assert all(generation.is_dir() for generation, _ in old)
            assert all(locked_name(generation, payload).is_file() for generation, payload in old)
            assert not any((generation / ".reclaimed").exists() for generation, _ in old)
    finally:
        release()

    # Tombstone pass: names leave the payload namespace, bytes queue under .reclaimed.
    shared, inode, busy, collect, selection, selected_payload, old = _hardlink_fixture(
        tmp_path, collector, "tombstone", shared_location
    )
    selection_bytes = selection.read_bytes()
    base_nlink = shared.stat().st_nlink
    release = lease_directory(busy)
    try:
        with monkeypatch.context() as lock:
            lock.setattr(sys, "platform", "win32")
            lock.setattr(os, "unlink", locked_unlink(inode))
            lock.setattr(runtime_state, "_unlink_by_posix_disposition", lambda path: False)
            assert collect() == []  # Bytes remain queued: never reported as removed.
            names = {}
            for generation, payload in old:
                tombstones = sorted((generation / ".reclaimed").iterdir())
                assert len(tombstones) == 1
                assert tombstones[0].read_bytes() == shared.read_bytes()  # Bytes intact.
                assert not locked_name(generation, payload).exists()  # Name is gone.
                assert (generation / ".lease-managed").is_file()
                names[generation] = tombstones[0].name
            assert shared.stat().st_nlink == base_nlink  # A rename never touches the inode.
            # A pass while the mapping lives must not re-rename or churn the queue.
            assert collect() == []
            for generation, _ in old:
                still = sorted(path.name for path in (generation / ".reclaimed").iterdir())
                assert still == [names[generation]]
        # Mapping released: the next pass reclaims the tombstones and the shells.
        assert set(collect()) == {generation for generation, _ in old}
        assert all(not generation.exists() for generation, _ in old)
        assert selection.read_bytes() == selection_bytes
        assert selected_payload.is_dir()
    finally:
        release()
    assert collect() == [busy]
