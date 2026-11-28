"""Behavior contracts for macOS ``/usr/bin/git`` CLT-stub detection in ``hermes_platform.resolver.git``."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from functools import partial
from pathlib import Path
from shlex import quote

import pytest

from hermes_platform.resolver.git import (
    GitResolution,
    _CltGitCache,
    _probe_xcode_select_developer_dir,
    clear_git_usability_cache,
    inspect_git,
    is_git_command,
    is_usable_git_executable,
    resolve_usable_git,
)


def _make_exe(path: Path, body: str = "#!/usr/bin/env bash\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.mark.platforms("posix")
def test_xcode_select_timeout_discards_late_developer_dir(tmp_path):
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    bash = shutil.which("bash")
    assert bash is not None, "bash is required to execute fake xcode-select scripts"
    (stub.parent / "bash").symlink_to(bash)
    developer = tmp_path / "developer"
    _make_exe(developer / "usr" / "bin" / "git")

    def _make_select(name: str, delay: float) -> Path:
        code = f"import time; time.sleep({delay}); print({str(developer)!r})"
        return _make_exe(
            tmp_path / name,
            f"#!/usr/bin/env bash\nexec {quote(sys.executable)} -c {quote(code)}\n",
        )

    fast_select = _make_select("xcode-select-fast", 0)
    fast = inspect_git(
        {"PATH": str(stub.parent)},
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=partial(_probe_xcode_select_developer_dir, xcode_select=str(fast_select)),
        cache=_CltGitCache(),
    )
    assert fast.status == "usable"

    slow_select = _make_select("xcode-select-slow", 3)
    slow = inspect_git(
        {"PATH": str(stub.parent)},
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=partial(_probe_xcode_select_developer_dir, xcode_select=str(slow_select)),
        cache=_CltGitCache(),
    )
    assert slow.status == "clt_stub"


@pytest.mark.platforms("posix")
def test_macos_clt_stub_without_clt_is_rejected(tmp_path):
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    cache = _CltGitCache()
    env = {"PATH": str(stub.parent)}

    res = inspect_git(
        env,
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=lambda _env: "",
        cache=cache,
    )
    assert res == GitResolution(executable=None, status="clt_stub", raw_path=str(stub))
    assert res.usable is False
    assert resolve_usable_git(
        env,
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=lambda _env: "",
        cache=cache,
    ) is None


@pytest.mark.platforms("posix")
def test_macos_clt_stub_rejects_empty_or_nonexecutable_developer_dir(tmp_path):
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    empty_dev = tmp_path / "empty_clt"
    (empty_dev / "usr" / "bin").mkdir(parents=True)

    assert not is_usable_git_executable(
        stub,
        {"DEVELOPER_DIR": str(empty_dev)},
        platform="darwin",
        apple_stub_path=str(stub),
    )
    assert not is_usable_git_executable(
        stub,
        {},
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=lambda _env: str(empty_dev),
        cache=_CltGitCache(),
    )

    nonexec_git = empty_dev / "usr" / "bin" / "git"
    nonexec_git.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    nonexec_git.chmod(0o644)
    assert not is_usable_git_executable(
        stub,
        {"DEVELOPER_DIR": str(empty_dev)},
        platform="darwin",
        apple_stub_path=str(stub),
    )

    dir_dev = tmp_path / "dir_clt"
    (dir_dev / "usr" / "bin" / "git").mkdir(parents=True)
    assert not is_usable_git_executable(
        stub,
        {"DEVELOPER_DIR": str(dir_dev)},
        platform="darwin",
        apple_stub_path=str(stub),
    )


@pytest.mark.platforms("posix")
def test_macos_developer_dir_may_name_the_xcode_app_bundle(tmp_path):
    """xcode-select(1): DEVELOPER_DIR may be the Xcode application directory; the shims use its
    Contents/Developer."""
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    xcode_app = tmp_path / "Xcode.app"
    _make_exe(xcode_app / "Contents" / "Developer" / "usr" / "bin" / "git")
    empty_app = tmp_path / "Empty.app"
    (empty_app / "Contents" / "Developer").mkdir(parents=True)

    def usable(dev_dir: Path) -> bool:
        return is_usable_git_executable(
            stub, {"DEVELOPER_DIR": str(dev_dir)}, platform="darwin", apple_stub_path=str(stub),
        )

    assert usable(xcode_app)
    assert usable(xcode_app / "Contents" / "Developer")
    assert not usable(empty_app)


@pytest.mark.platforms("posix")
def test_macos_symlink_to_apple_stub_is_still_classified_as_stub(tmp_path):
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    alias_dir = tmp_path / "alias_bin"
    alias_dir.mkdir()
    alias = alias_dir / "git"
    alias.symlink_to(stub)

    res = inspect_git(
        {"PATH": str(alias_dir)},
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=lambda _env: "",
        cache=_CltGitCache(),
    )
    assert res == GitResolution(executable=None, status="clt_stub", raw_path=str(alias))


@pytest.mark.platforms("posix")
def test_macos_clt_stub_with_valid_developer_git_is_usable(tmp_path):
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    dev_dir = tmp_path / "clt"
    _make_exe(dev_dir / "usr" / "bin" / "git")

    def fail_probe(_env):
        raise AssertionError("xcode-select must not run when DEVELOPER_DIR is set")

    via_env = inspect_git(
        {"PATH": str(stub.parent), "DEVELOPER_DIR": str(dev_dir)},
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=fail_probe,
        cache=_CltGitCache(),
    )
    assert via_env == GitResolution(executable=str(stub), status="usable", raw_path=str(stub))
    assert via_env.usable is True

    via_probe = inspect_git(
        {"PATH": str(stub.parent)},
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=lambda _env: str(dev_dir),
        cache=_CltGitCache(),
    )
    assert via_probe == GitResolution(executable=str(stub), status="usable", raw_path=str(stub))


@pytest.mark.platforms("posix")
def test_custom_git_on_path_is_usable_even_when_clt_absent(tmp_path):
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    homebrew = _make_exe(tmp_path / "homebrew" / "bin" / "git")

    def fail_probe(_env):
        raise AssertionError("xcode-select must not run for a non-/usr/bin/git executable")

    env = {
        "PATH": f"{homebrew.parent}{os.pathsep}{stub.parent}",
        "DEVELOPER_DIR": str(tmp_path / "nonexistent-clt"),
    }
    res = inspect_git(
        env,
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=fail_probe,
        cache=_CltGitCache(),
    )
    assert res == GitResolution(executable=str(homebrew), status="usable", raw_path=str(homebrew))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("other_platform", ["linux", "win32"])
def test_non_macos_platform_never_runs_clt_probe(tmp_path, other_platform):
    stub = _make_exe(tmp_path / "usr_bin" / "git")

    def fail_probe(_env):
        raise AssertionError("CLT probe must only run on darwin")

    res = inspect_git(
        {"PATH": str(stub.parent), "DEVELOPER_DIR": str(tmp_path / "nonexistent")},
        platform=other_platform,
        apple_stub_path=str(stub),
        developer_dir_probe=fail_probe,
        cache=_CltGitCache(),
    )
    assert res == GitResolution(executable=str(stub), status="usable", raw_path=str(stub))


@pytest.mark.platforms("posix")
def test_cache_ttl_heals_negative_and_caches_positive_and_respects_cached_false(tmp_path):
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    dev_dir = tmp_path / "clt"
    dev_dir.mkdir()
    now = [100.0]
    probe_calls = 0
    clt_ready = [False]

    def probe(_env):
        nonlocal probe_calls
        probe_calls += 1
        return str(dev_dir) if clt_ready[0] else ""

    cache = _CltGitCache(negative_ttl_s=5.0)
    env = {"PATH": str(stub.parent)}

    assert not is_usable_git_executable(
        stub, env, platform="darwin", apple_stub_path=str(stub),
        developer_dir_probe=probe, cache=cache, now_fn=lambda: now[0],
    )
    assert probe_calls == 1

    now[0] = 104.0
    assert not is_usable_git_executable(
        stub, env, platform="darwin", apple_stub_path=str(stub),
        developer_dir_probe=probe, cache=cache, now_fn=lambda: now[0],
    )
    assert probe_calls == 1

    _make_exe(dev_dir / "usr" / "bin" / "git")
    clt_ready[0] = True
    assert is_usable_git_executable(
        stub, env, cached=False, platform="darwin", apple_stub_path=str(stub),
        developer_dir_probe=probe, cache=cache, now_fn=lambda: now[0],
    )
    assert probe_calls == 2

    now[0] = 99999.0
    assert is_usable_git_executable(
        stub, env, cached=True, platform="darwin", apple_stub_path=str(stub),
        developer_dir_probe=probe, cache=cache, now_fn=lambda: now[0],
    )
    assert probe_calls == 2


@pytest.mark.platforms("posix")
def test_negative_cache_expires_after_ttl_without_restart(tmp_path):
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    dev_dir = tmp_path / "clt"
    dev_dir.mkdir()
    now = [50.0]
    calls = 0
    installed = [False]

    def probe(_env):
        nonlocal calls
        calls += 1
        return str(dev_dir) if installed[0] else ""

    cache = _CltGitCache(negative_ttl_s=5.0)
    env = {"PATH": str(stub.parent)}

    assert not is_usable_git_executable(
        stub, env, platform="darwin", apple_stub_path=str(stub),
        developer_dir_probe=probe, cache=cache, now_fn=lambda: now[0],
    )
    assert calls == 1

    _make_exe(dev_dir / "usr" / "bin" / "git")
    installed[0] = True
    now[0] = 55.0
    assert is_usable_git_executable(
        stub, env, platform="darwin", apple_stub_path=str(stub),
        developer_dir_probe=probe, cache=cache, now_fn=lambda: now[0],
    )
    assert calls == 2


@pytest.mark.platforms("posix")
def test_explicit_env_without_developer_dir_does_not_inherit_ambient_developer_dir(tmp_path, monkeypatch):
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    ambient_dev = tmp_path / "ambient_clt"
    _make_exe(ambient_dev / "usr" / "bin" / "git")
    monkeypatch.setenv("DEVELOPER_DIR", str(ambient_dev))

    res = inspect_git(
        {"PATH": str(stub.parent)},
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=lambda _env: "",
        cache=_CltGitCache(),
    )
    assert res == GitResolution(executable=None, status="clt_stub", raw_path=str(stub))


def test_empty_path_reports_missing_not_clt_stub(tmp_path):
    res = inspect_git({"PATH": ""}, platform="darwin", cache=_CltGitCache())
    assert res == GitResolution(executable=None, status="missing", raw_path=None)
    _make_exe(tmp_path / "git")  # an empty PATH searches nothing, not the cwd
    res_cwd = inspect_git({"PATH": ""}, cwd=tmp_path, platform="darwin", cache=_CltGitCache())
    assert res_cwd == GitResolution(executable=None, status="missing", raw_path=None)


@pytest.mark.platforms("posix")
def test_inspect_git_resolves_relative_path_entries_against_cwd(tmp_path, monkeypatch):
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    child.mkdir()
    _make_exe(parent / "bin" / "git")
    monkeypatch.chdir(parent)

    env = {"PATH": f"bin{os.pathsep}{stub.parent}"}
    in_child = inspect_git(
        env,
        cwd=child,
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=lambda _env: "",
        cache=_CltGitCache(),
    )
    assert in_child == GitResolution(executable=None, status="clt_stub", raw_path=str(stub))

    empty_seg_env = {"PATH": f"{os.pathsep}{stub.parent}"}
    child_local = _make_exe(child / "git")
    found_in_child = inspect_git(
        empty_seg_env,
        cwd=child,
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=lambda _env: "",
        cache=_CltGitCache(),
    )
    assert found_in_child.status == "usable"
    assert found_in_child.executable is not None
    assert os.path.realpath(found_in_child.executable) == os.path.realpath(child_local)

    monkeypatch.setenv("PATH", f"bin{os.pathsep}{stub.parent}")
    ambient_in_child = inspect_git(
        None,
        cwd=child,
        platform="darwin",
        apple_stub_path=str(stub),
        developer_dir_probe=lambda _env: "",
        cache=_CltGitCache(),
    )
    assert ambient_in_child == GitResolution(executable=None, status="clt_stub", raw_path=str(stub))


@pytest.mark.platforms("posix")
def test_inspect_git_keeps_a_pathsep_in_cwd_inside_one_search_entry(tmp_path, monkeypatch):
    """A checkout directory may contain ``:``; joining it into PATH entries must not split it."""
    stub = _make_exe(tmp_path / "usr_bin" / "git")
    child = tmp_path / f"a{os.pathsep}b"
    child_git = _make_exe(child / "bin" / "git")
    path = f"bin{os.pathsep}{stub.parent}"

    def _inspect(env):
        return inspect_git(
            env,
            cwd=child,
            platform="darwin",
            apple_stub_path=str(stub),
            developer_dir_probe=lambda _env: "",
            cache=_CltGitCache(),
        )

    explicit = _inspect({"PATH": path})
    monkeypatch.setenv("PATH", path)
    ambient = _inspect(None)
    for res in (explicit, ambient):
        assert res.status == "usable"
        assert res.executable is not None
        assert os.path.realpath(res.executable) == os.path.realpath(child_git)


def test_spawn_guard_refuses_only_the_clt_stub_not_a_git_it_cannot_find(tmp_path):
    """A bare launch searches PATH its own way (CreateProcess reads the parent's PATH, execvp
    follows a relative entry from the child's cwd), so "not found from here" is left to the launch
    to report rather than refused as if it were the stub."""
    from hermes_platform.resolver.git import is_usable_git_command

    empty = tmp_path / "empty"
    empty.mkdir()
    assert is_usable_git_command("git", {"PATH": str(empty)})
    assert is_usable_git_command("git", {"PATH": ""})


def test_is_git_command_matches_only_git():
    assert is_git_command("git") is True
    assert is_git_command("/usr/bin/git") is True
    assert is_git_command("/opt/homebrew/bin/git") is True
    assert is_git_command("python3") is False
    assert is_git_command("/usr/bin/rg") is False
    assert is_git_command("") is False
    assert is_git_command(None) is False


@pytest.mark.platforms("macos")
def test_macos_native_absent_clt_blocks_usr_bin_git(tmp_path):
    clear_git_usability_cache()
    env = {"PATH": "/usr/bin:/bin", "DEVELOPER_DIR": str(tmp_path / "missing-clt")}
    res = inspect_git(env, cached=False)
    assert res == GitResolution(executable=None, status="clt_stub", raw_path="/usr/bin/git")


@pytest.mark.platforms("macos")
def test_macos_native_valid_developer_dir_allows_usr_bin_git(tmp_path):
    clear_git_usability_cache()
    dev_dir = tmp_path / "dev"
    _make_exe(dev_dir / "usr" / "bin" / "git")
    env = {"PATH": "/usr/bin:/bin", "DEVELOPER_DIR": str(dev_dir)}
    res = inspect_git(env, cached=False)
    assert res == GitResolution(executable="/usr/bin/git", status="usable", raw_path="/usr/bin/git")


@pytest.mark.platforms("macos")
def test_macos_native_custom_git_on_path_not_blocked_by_absent_clt(tmp_path):
    clear_git_usability_cache()
    real_git = shutil.which("git")
    if real_git is None:
        pytest.skip("git not installed")
    custom_dir = tmp_path / "homebrew" / "bin"
    custom_dir.mkdir(parents=True)
    custom_git = custom_dir / "git"
    if os.path.realpath(real_git) == "/usr/bin/git":
        proc = subprocess.run(
            ["/usr/bin/xcode-select", "-p"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False, timeout=2,
        )
        backing = Path(proc.stdout.strip()) / "usr" / "bin" / "git"
        if not backing.is_file():
            pytest.skip("no backing git binary on host")
        custom_git.symlink_to(backing)
    else:
        custom_git.symlink_to(real_git)
    env = {
        "PATH": f"{custom_dir}:/usr/bin:/bin",
        "DEVELOPER_DIR": str(tmp_path / "missing-clt"),
    }
    res = inspect_git(env, cached=False)
    assert res == GitResolution(executable=str(custom_git), status="usable", raw_path=str(custom_git))
