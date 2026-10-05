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


def test_checkout_payload_manifest_may_be_probed_but_never_read():
    """PM probes ``<checkout>/../manifest.json`` (a sealed payload's descriptor) on every
    hermes_bootstrap import. With the checkout inside the home that is the real home's
    ``manifest.json``: an existence probe of exactly that file passes; reading or writing it,
    and probing anything else in the root, are still refused."""
    from tests.home_io_guard import HomeIOGuard

    root = PROJECT_ROOT.parent
    guard = HomeIOGuard(lambda: [root])
    guard.check(root / "manifest.json", metadata=True)
    for refused in (lambda: guard.check(root / "manifest.json"),
                    lambda: guard.check(root / "manifest.json", destructive=True),
                    lambda: guard.check(root / "manifest.json.bak", metadata=True),
                    lambda: guard.check(root / "config.yaml", metadata=True),
                    lambda: guard.check(root / "profiles" / "work" / "manifest.json", metadata=True)):
        with pytest.raises(AssertionError, match="REAL hermes home"):
            refused()


def test_first_bootstrap_import_under_the_guard_stays_out_of_the_home(tmp_path):
    """A test's first real OpenAI client imports hermes_bootstrap lazily, after the autouse
    guard is armed; its import-time dependency selection probed the checkout's sibling
    manifest.json and tripped the guard whenever the checkout lives inside the guarded home
    (the default install). Guard the checkout's parent as the real home and import it fresh."""
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent(f"""
        import sys
        from pathlib import Path
        import pytest
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        from tests.home_io_guard import HomeIOGuard

        patcher = pytest.MonkeyPatch()
        HomeIOGuard(lambda: [Path({str(PROJECT_ROOT.parent)!r})]).install(patcher)
        assert "hermes_bootstrap" not in sys.modules
        import hermes_bootstrap  # noqa: F401
        patcher.undo()
        print("bootstrap-imported")
    """), encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "HERMES_RUNTIME_DIR")}
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    result = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True,
                            cwd=str(tmp_path), env=env, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("bootstrap-imported")


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
