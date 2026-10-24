"""Real hard-link reclamation regressions for #129300 and #129307."""
import json
import os
import shutil

import pytest


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
