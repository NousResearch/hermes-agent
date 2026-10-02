"""Real-home guards are exercised against disposable protected roots only."""
from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import textwrap

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def protected_home(tmp_path, monkeypatch):
    from tests import conftest

    root = tmp_path / "protected"
    root.mkdir()
    (root / "file.txt").write_text("unchanged", encoding="utf-8")
    (root / "empty").mkdir()
    monkeypatch.setattr(conftest, "_REAL_HERMES_ROOT_CANDIDATES", [root])
    yield root
    # Restore access before tmp_path/pytest cleanup even when an assertion fails.
    monkeypatch.setattr(conftest, "_REAL_HERMES_ROOT_CANDIDATES", [])


def _open_close(path):
    with open(path, encoding="utf-8"):
        pass


def _io_open_close(path):
    with io.open(path, encoding="utf-8"):
        pass


def _os_open_close(path):
    descriptor = os.open(path, os.O_RDONLY)
    os.close(descriptor)


_OPERATIONS = {
    "builtin-open": _open_close,
    "io-open": _io_open_close,
    "os-open": _os_open_close,
    "read-text": lambda path: path.read_text(encoding="utf-8"),
    "write-text": lambda path: path.write_text("changed", encoding="utf-8"),
    "read-bytes": lambda path: path.read_bytes(),
    "write-bytes": lambda path: path.write_bytes(b"changed"),
    "stat": lambda path: path.stat(),
    "lstat": lambda path: path.lstat(),
    "unlink": lambda path: path.unlink(),
    "remove": lambda path: os.remove(path),
    "rename-out": lambda path: path.rename(path.parent.parent / "moved.txt"),
    "replace-out": lambda path: path.replace(path.parent.parent / "replaced.txt"),
    "mkdir": lambda path: (path.parent / "new").mkdir(),
    "makedirs": lambda path: os.makedirs(path.parent / "deep" / "child"),
    "rmdir": lambda path: (path.parent / "empty").rmdir(),
    "rmtree": lambda path: shutil.rmtree(path.parent),
    "listdir": lambda path: os.listdir(path.parent),
    "scandir": lambda path: list(os.scandir(path.parent)),
    "sqlite": lambda path: sqlite3.connect(path.parent / "state.db").close(),
}


@pytest.mark.parametrize("operation", _OPERATIONS, ids=_OPERATIONS)
def test_io_guard_denies_protected_roots_before_mutation(protected_home, operation):
    with pytest.raises((AssertionError, pytest.fail.Exception), match="REAL hermes home"):
        _OPERATIONS[operation](protected_home / "file.txt")


def test_rename_cannot_overwrite_a_protected_destination(protected_home, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("external", encoding="utf-8")
    with pytest.raises((AssertionError, pytest.fail.Exception), match="REAL hermes home"):
        source.replace(protected_home / "file.txt")
    assert source.read_text(encoding="utf-8") == "external"


def test_installed_desktop_app_can_be_read_but_never_changed(tmp_path, monkeypatch):
    from tests import conftest

    app = tmp_path / "Applications" / "Hermes.app"
    (app / "Contents").mkdir(parents=True)
    plist = app / "Contents" / "Info.plist"
    plist.write_text("unchanged", encoding="utf-8")
    replacement = tmp_path / "Hermes.app.new"
    replacement.mkdir()
    changes = [lambda: shutil.rmtree(app), plist.unlink, lambda: replacement.replace(app),
               lambda: plist.write_text("clobbered", encoding="utf-8")]
    try:  # Windows without Developer Mode cannot create symlinks
        (tmp_path / "linked").symlink_to(app.parent, target_is_directory=True)
        changes.append(lambda: shutil.rmtree(tmp_path / "linked" / "Hermes.app"))
    except OSError:
        pass
    monkeypatch.setattr(conftest, "_REAL_INSTALLED_GUI_APPS", [os.path.normcase(os.path.realpath(app))])
    assert plist.read_text(encoding="utf-8") == "unchanged"
    for change in changes:
        with pytest.raises(pytest.fail.Exception, match="REAL installed Hermes desktop app"):
            change()
    assert plist.read_text(encoding="utf-8") == "unchanged"


def test_unprotected_paths_and_open_descriptors_still_work(protected_home, tmp_path):
    target = tmp_path / "allowed" / "file.txt"
    target.parent.mkdir()
    target.write_text("ok", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "ok"
    fd = os.open(target, os.O_RDONLY)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        assert stream.read() == "ok"
    moved = target.replace(target.with_name("moved.txt"))
    assert list(moved.parent.iterdir()) == [moved]
    moved.unlink()
    moved.parent.rmdir()
    connection = sqlite3.connect(tmp_path / "allowed.db")
    connection.close()


@pytest.mark.allow_real_home_io
def test_explicit_opt_out_allows_only_the_disposable_canary(protected_home):
    target = protected_home / "file.txt"
    target.write_text("opted out", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "opted out"


def test_close_keeps_a_reused_descriptors_new_owner(tmp_path, monkeypatch):
    from tests.home_io_guard import HomeIOGuard

    first, second = tmp_path / "first", tmp_path / "second"
    first.touch()
    second.touch()
    original_close = os.close
    reopened = []

    def close_and_reopen(fd):
        original_close(fd)
        reopened.append(os.open(second, os.O_RDONLY))

    guard = HomeIOGuard(lambda: [])
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(os, "close", close_and_reopen)
            guard.install(patcher)
            fd = os.open(first, os.O_RDONLY)
            os.close(fd)
            assert reopened == [fd], "the test must exercise descriptor reuse"
            assert guard.directories[fd] == second
    finally:
        for fd in reopened:
            original_close(fd)


def test_path_metadata_exemption_tracks_path_changes(protected_home, monkeypatch):
    from tests.home_io_guard import HomeIOGuard

    target = protected_home / "file.txt"
    guard = HomeIOGuard(lambda: [protected_home])
    monkeypatch.setenv("PATH", str(protected_home))
    guard.check(target, metadata=True)  # executable lookup, not a state read
    monkeypatch.setenv("PATH", str(protected_home.parent))
    with pytest.raises(AssertionError, match="REAL hermes home"):
        guard.check(target, metadata=True)


def test_relative_path_metadata_exemption_tracks_working_directory(protected_home, tmp_path, monkeypatch):
    from tests.home_io_guard import HomeIOGuard

    target = protected_home / "file.txt"
    guard = HomeIOGuard(lambda: [protected_home])
    other = tmp_path / "other"
    other.mkdir()
    (other / "nested").mkdir()
    monkeypatch.setenv("PATH", "../protected")
    monkeypatch.chdir(other)
    guard.check(target, metadata=True)
    monkeypatch.chdir(other / "nested")
    with pytest.raises(AssertionError, match="REAL hermes home"):
        guard.check(target, metadata=True)


def test_checkout_inside_a_guarded_root_is_not_hermes_state():
    """The default install checks the repo out INSIDE the home (install.sh:
    INSTALL_DIR=$HERMES_HOME/hermes-agent): the checkout, its .venv and test
    data are exempt even when the guarded root contains them; siblings under
    that root are still refused."""
    from tests.home_io_guard import HomeIOGuard

    guard = HomeIOGuard(lambda: [PROJECT_ROOT.parent])
    guard.check(PROJECT_ROOT / "tests" / "home_io_guard.py")
    guard.check(PROJECT_ROOT / ".venv" / "bin" / "python", metadata=True)
    with pytest.raises(AssertionError, match="REAL hermes home"):
        guard.check(PROJECT_ROOT.parent / "config.yaml")


def test_hermes_exported_scratch_tmp_is_not_the_test_temp_root(tmp_path):
    """A Hermes-launched shell hands pytest TMPDIR=<home>/cache/scratch (tagged by
    HERMES_SCRATCH_DIR). With that home guarded, honoring it would put the session
    sandbox, basetemp and every tempfile default inside the guarded root; the
    conftest must drop Hermes' own export before anything allocates temp space."""
    home = tmp_path / "home"
    scratch = home / "cache" / "scratch"
    scratch.mkdir(parents=True)
    probe = tmp_path / "test_probe.py"
    probe.write_text(textwrap.dedent(f"""
        import tempfile
        from pathlib import Path

        def test_temp_root_is_outside_the_honored_home():
            home = Path({str(home)!r}).resolve()
            assert not Path(tempfile.gettempdir()).resolve().is_relative_to(home)
            with tempfile.TemporaryDirectory() as made:
                assert not Path(made).resolve().is_relative_to(home)
        """), encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR", "HERMES_TEST_SANDBOX_HOME")}
    env.update(HERMES_HOME=str(home), TMPDIR=str(scratch), HERMES_SCRATCH_DIR=str(scratch))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-p", "no:cacheprovider", "-q", str(probe)],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


@pytest.fixture(params=["sysconfig", "loaded-modules"])
def protected_runtime_library(tmp_path, monkeypatch, request):
    from tests import conftest, home_io_guard

    home = tmp_path / "runtime-home"
    library = home / "runtime" / "Lib"
    library.mkdir(parents=True)
    source = library / "threading.py"
    source.write_text("stdlib source", encoding="utf-8")
    (library / "empty").mkdir()
    state = home / "state.txt"
    state.write_text("private state", encoding="utf-8")
    paths = [source]
    state_paths = [state]
    try:
        alias = tmp_path / "library-link"
        alias.symlink_to(library, target_is_directory=True)
        (library / "state-link").symlink_to(state)
    except OSError:
        request.node.user_properties.append(("symlink_supported", False))
    else:
        paths.append(alias / source.name)
        state_paths.extend([library / "state-link", alias / "state-link"])
        request.node.user_properties.append(("symlink_supported", True))
    # Exercise discovery from runtime facts, never sys.path or a home-wide whitelist.
    with monkeypatch.context() as discovery:
        discovery.setattr(home_io_guard.sysconfig, "get_path", lambda name: str(
            library if request.param == "sysconfig" else tmp_path / "stale-runtime" / name))
        for module in (home_io_guard.shutil, home_io_guard.threading):
            discovery.setattr(module, "__file__", str(
                (library if request.param == "loaded-modules" else tmp_path / module.__name__)
                / (module.__name__ + ".py")))
        prefixes = home_io_guard._stdlib_prefixes()
    monkeypatch.setattr(home_io_guard, "_STDLIB_PREFIX_STRS",
                        home_io_guard._STDLIB_PREFIX_STRS + prefixes)
    original_roots = conftest._REAL_HERMES_ROOT_CANDIDATES
    monkeypatch.setattr(conftest, "_REAL_HERMES_ROOT_CANDIDATES", original_roots + [home])
    try:
        yield home, paths, state_paths
    finally:
        monkeypatch.setattr(conftest, "_REAL_HERMES_ROOT_CANDIDATES", original_roots)


def test_runtime_library_reads_and_metadata_preserve_state_isolation(protected_runtime_library):
    home, paths, state_paths = protected_runtime_library
    for source in paths:
        assert source.read_text(encoding="utf-8") == "stdlib source"
        with open(source, encoding="utf-8") as stream:
            assert stream.read() == "stdlib source"
        fd = os.open(source, os.O_RDONLY)
        try:
            assert os.read(fd, 100) == b"stdlib source"
        finally:
            os.close(fd)
        assert source.stat().st_size == len(b"stdlib source")
        assert source.lstat().st_size == len(b"stdlib source")
        assert os.access(source, os.R_OK)
        assert os.path.realpath(source).endswith(os.path.join("Lib", "threading.py"))
        assert "threading.py" in os.listdir(source.parent)
        with os.scandir(source.parent) as entries:
            assert "threading.py" in {entry.name for entry in entries}
    # realpath walks metadata through protected ancestors, but reading those
    # directories would reveal sibling state and remains forbidden.
    assert (home / "runtime").stat().st_mode
    assert (home / "runtime").resolve() == home / "runtime"
    for state in state_paths:
        for operation in (lambda: state.read_text(), state.stat):
            with pytest.raises(AssertionError, match="REAL hermes home"):
                operation()
    with pytest.raises(AssertionError, match="REAL hermes home"):
        os.listdir(home / "runtime")


def test_runtime_library_exception_never_permits_mutation(protected_runtime_library, tmp_path):
    _, paths, _ = protected_runtime_library
    replacement = tmp_path / "replacement.py"
    replacement.write_text("replacement", encoding="utf-8")
    for source in paths:
        changes = [
            lambda: source.write_text("changed"),
            lambda: open(source, "r+"),
            lambda: open(source.parent / "created.py", "x"),
            lambda: os.open(source, os.O_WRONLY),
            lambda: os.open(source, os.O_RDONLY | os.O_CREAT),
            lambda: os.open(source, os.O_RDONLY | os.O_APPEND),
            lambda: (source.parent / "new").mkdir(),
            lambda: os.makedirs(source.parent / "deep" / "child"),
            lambda: os.chmod(source, 0o600),
            lambda: os.utime(source, None),
            source.unlink,
            lambda: source.replace(tmp_path / "removed.py"),
            lambda: source.rename(tmp_path / "renamed.py"),
            lambda: replacement.replace(source),
            lambda: (source.parent / "empty").rmdir(),
            lambda: shutil.rmtree(source.parent),
            lambda: sqlite3.connect(source.parent / "state.db"),
        ]
        if getattr(os, "O_TEMPORARY", 0):
            changes.append(lambda: os.open(source, os.O_RDONLY | os.O_TEMPORARY))
        for change in changes:
            with pytest.raises(AssertionError, match="REAL hermes home"):
                change()
        assert source.read_text(encoding="utf-8") == "stdlib source"
        assert "created.py" not in os.listdir(source.parent)
    assert replacement.read_text(encoding="utf-8") == "replacement"


@pytest.mark.parametrize("missing_second", [False, True])
def test_loaded_library_requires_two_matching_module_paths(tmp_path, monkeypatch, missing_second):
    from tests import home_io_guard

    monkeypatch.setattr(home_io_guard.sysconfig, "get_path", lambda name: None)
    monkeypatch.setattr(home_io_guard.shutil, "__file__", str(tmp_path / "first" / "shutil.py"))
    monkeypatch.setattr(home_io_guard.threading, "__file__",
                        None if missing_second else str(tmp_path / "second" / "threading.py"))
    assert home_io_guard._stdlib_prefixes() == ()


def test_library_archive_exception_is_exact_and_corroborated(tmp_path, monkeypatch):
    from tests import home_io_guard

    library = tmp_path / "runtime" / "Lib"
    archive = library.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    unrelated = tmp_path / "other-runtime" / archive.name
    monkeypatch.setattr(home_io_guard.sysconfig, "get_path", lambda name: str(library))
    for module in (home_io_guard.shutil, home_io_guard.threading):
        monkeypatch.setattr(module, "__file__", str(library / (module.__name__ + ".py")))
    monkeypatch.setattr(home_io_guard.sys, "path", [str(archive), str(unrelated), str(library.parent)])
    assert set(home_io_guard._stdlib_prefixes()) == {
        os.path.normcase(str(library.resolve())), os.path.normcase(str(archive.resolve()))
    }
    library.mkdir(parents=True)
    archive.write_text("archive fixture", encoding="utf-8")
    sibling = library.parent / "state.db"
    sibling.write_text("private state", encoding="utf-8")
    from tests import conftest
    monkeypatch.setattr(home_io_guard, "_STDLIB_PREFIX_STRS", home_io_guard._stdlib_prefixes())
    monkeypatch.setattr(conftest, "_REAL_HERMES_ROOT_CANDIDATES",
                        conftest._REAL_HERMES_ROOT_CANDIDATES + [library.parent])
    assert archive.stat().st_size == len("archive fixture")
    assert archive.read_text(encoding="utf-8") == "archive fixture"
    for action in (lambda: archive.write_text("changed"), archive.unlink,
                   lambda: sibling.read_text(), lambda: os.listdir(library.parent)):
        with pytest.raises(AssertionError, match="REAL hermes home"):
            action()
