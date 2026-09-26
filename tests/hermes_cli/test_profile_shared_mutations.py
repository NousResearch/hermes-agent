"""Per-name lifecycle locks must retain shared-file update and group contracts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import stat
import threading

import pytest
import yaml

from hermes_cli import profile_lifecycle, profiles


@pytest.fixture
def root(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.mark.parametrize("operation", ["metadata", "rename"])
def test_disjoint_mutations_preserve_shared_file_updates(root, monkeypatch, operation):
    import utils

    if operation == "metadata":
        target = root / "profile.yaml"
        target.write_text("{}", encoding="utf-8")
        writer_module, writer_name = utils, "atomic_yaml_write"
        first_call = lambda: profiles.write_profile_meta(root, description="retained description")
        second_call = lambda: profiles.set_profile_display_name("default", "retained name")
    else:
        for name in ("alpha", "beta"):
            profiles.create_profile(name, no_alias=True, no_skills=True)
        target = root / "honcho.json"
        target.write_text(json.dumps({"hosts": {
            "hermes_alpha": {"aiPeer": "alpha"}, "hermes_beta": {"aiPeer": "beta"},
        }}), encoding="utf-8")
        writer_module, writer_name = profiles, "_atomic_write_json"
        first_call = lambda: profiles.rename_profile("alpha", "gamma")
        second_call = lambda: profiles.rename_profile("beta", "delta")

    first_read, second_attempted, release = threading.Event(), threading.Event(), threading.Event()
    second_thread = None
    second_read_before_release = []
    real_write = getattr(writer_module, writer_name)
    real_read = Path.read_text
    real_sleep = profile_lifecycle.time.sleep

    def write(path, data, **kwargs):
        if path == target and threading.current_thread() is not second_thread:
            first_read.set()
            assert release.wait(10)
        return real_write(path, data, **kwargs)

    def read(path, *args, **kwargs):
        data = real_read(path, *args, **kwargs)
        if path == target and threading.current_thread() is second_thread:
            second_read_before_release.append(not release.is_set())
            second_attempted.set()
        return data

    def sleep(seconds):
        if threading.current_thread() is second_thread:
            second_attempted.set()  # Actual OS-lock contention, not scheduler timing.
        real_sleep(seconds)

    def second():
        nonlocal second_thread
        second_thread = threading.current_thread()
        return second_call()

    monkeypatch.setattr(writer_module, writer_name, write)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(profile_lifecycle.time, "sleep", sleep)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_call)
        try:
            assert first_read.wait(10)
            pending = pool.submit(second)
            assert second_attempted.wait(10)
        finally:
            release.set()
        first.result(15)
        pending.result(15)
    assert second_read_before_release == [False]
    if operation == "metadata":
        assert yaml.safe_load(target.read_text()) == {
            "description": "retained description", "display_name": "retained name",
        }
    else:
        assert set(json.loads(target.read_text())["hosts"]) == {"hermes_gamma", "hermes_delta"}
        assert all((root / "profiles" / name).is_dir() for name in ("gamma", "delta"))


@pytest.mark.platforms("linux")
def test_lifecycle_directories_preserve_group_inheritance(root):
    directory = root / "profiles"
    directory.mkdir(mode=0o2770)
    directory.chmod(0o2770)
    for name in ("alpha", "beta"):
        with profile_lifecycle.profile_lifecycle_lease(directory / name):
            pass
        assert (directory / ".locks").stat().st_mode & stat.S_ISGID
    for name in ("alpha", "beta"):
        lock = directory / ".locks" / f"{name}.lock"
        assert lock.stat().st_gid == directory.stat().st_gid
        assert stat.S_IMODE(lock.stat().st_mode) == 0o660
    profile_lifecycle.mark_profile_deleting(directory / "alpha")
    assert (directory / ".deleted").stat().st_mode & stat.S_ISGID
