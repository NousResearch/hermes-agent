"""Exercise remote mutation programs against synthetic nested .hermes scratch trees."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from hermes_cli.ssh_workspace_fs import SshWorkspaceFs, SshWorkspaceFsError


def _shell_path(path: Path) -> str:
    return path.as_posix()


@pytest.fixture
def shell_workspace(tmp_path):
    bash = shutil.which("bash")
    if bash is None or shutil.which("python3") is None:
        pytest.skip("Bash and remote Python 3 are required for mutation programs")
    home = tmp_path / "synthetic" / ".hermes" / "home"
    workspace = home / "workspace"
    workspace.mkdir(parents=True)

    class ShellTransport:
        cwd = _shell_path(workspace)
        _remote_home = _shell_path(home)
        timeout = 15

        def execute(self, command, *, cwd=None, stdin_data=None, timeout=None):
            result = subprocess.run(
                [bash, "-c", f"cd -- {shlex.quote(cwd or self.cwd)}\n" + command],
                input=stdin_data.encode() if stdin_data is not None else None,
                capture_output=True,
                timeout=timeout or self.timeout,
                env={"HOME": str(home), "PATH": os.environ["PATH"], "LANG": "C.UTF-8"},
                check=False,
            )
            return {
                "returncode": result.returncode,
                "output": (result.stdout + result.stderr).decode("utf-8", errors="replace"),
            }

    return SshWorkspaceFs(ShellTransport())


@pytest.fixture
def project(shell_workspace):
    path = Path(shell_workspace.cwd) / "project"
    path.mkdir()
    return path


def _denied(operation, code="EACCES"):
    with pytest.raises(SshWorkspaceFsError) as raised:
        operation()
    assert raised.value.code == code


@pytest.mark.platforms("posix")
def test_nonrecursive_file_delete_finishes_without_directory_removal(shell_workspace, project):
    target = project / "file with spaces.txt"
    shell_workspace.create(str(target), directory=False)
    assert target.is_file()
    assert shell_workspace.delete(str(target), recursive=False) == str(target)
    assert not target.exists()


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.platforms("posix")
def test_directory_create_and_delete_execute_quoted_target(shell_workspace, project, recursive):
    target = project / "folder with 'quotes' and spaces"
    assert shell_workspace.create(str(target), directory=True) == str(target)
    assert target.is_dir()
    if recursive:
        (target / "child.txt").write_bytes(b"remove")
    assert shell_workspace.delete(str(target), recursive=recursive) == str(target)
    assert not target.exists()


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("dangling", [False, True])
@pytest.mark.platforms("posix")
def test_link_delete_preserves_its_referent(shell_workspace, project, dangling, recursive):
    protected = project / "mcp-tokens"
    protected.mkdir()
    sentinel = protected / "sentinel.json"
    sentinel.write_bytes(b"keep")
    link = project / "link"
    link.symlink_to(project / "missing" if dangling else protected, target_is_directory=True)
    shell_workspace.delete(str(link), recursive=recursive)
    assert not link.is_symlink()
    assert sentinel.read_bytes() == b"keep"


@pytest.mark.platforms("linux")
def test_create_and_rename_collisions_preserve_existing_entries(shell_workspace, project):
    source = project / "source"
    source.write_bytes(b"source")
    destination = project / "destination"
    destination.write_bytes(b"keep")
    _denied(lambda: shell_workspace.create(str(destination), directory=False), "EEXIST")
    _denied(lambda: shell_workspace.create(str(destination), directory=True), "EEXIST")
    _denied(lambda: shell_workspace.rename(str(source), destination.name), "EEXIST")
    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"keep"
    destination.unlink()
    destination.symlink_to(project / "missing")
    _denied(lambda: shell_workspace.create(str(destination), directory=False), "EEXIST")
    _denied(lambda: shell_workspace.rename(str(source), destination.name), "EEXIST")
    assert destination.is_symlink()
    assert not (project / "missing").exists()
    destination.unlink()
    destination.mkdir()
    _denied(lambda: shell_workspace.rename(str(source), destination.name), "EEXIST")
    assert destination.is_dir()
    assert source.read_bytes() == b"source"


@pytest.mark.platforms("posix")
def test_create_refuses_fifo_without_opening_it(shell_workspace, project):
    target = project / "fifo"
    os.mkfifo(target)
    _denied(lambda: shell_workspace.create(str(target), directory=False), "EEXIST")
    assert target.is_fifo()


@pytest.mark.platforms("posix")
def test_simultaneous_creates_have_one_winner(shell_workspace, project):
    barrier = Barrier(2)
    target = project / "new"

    def create(_):
        backend = SshWorkspaceFs(shell_workspace._env)
        barrier.wait(timeout=10)
        try:
            backend.create(str(target), directory=False)
        except SshWorkspaceFsError as exc:
            return exc.code
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, range(2)))
    assert sorted(results) == ["EEXIST", "ok"]
    assert target.read_bytes() == b""


@pytest.mark.platforms("linux")
def test_simultaneous_renames_have_one_winner_without_overwrite(shell_workspace, project):
    barrier = Barrier(2)
    for name in ("first", "second"):
        (project / name).write_text(name, encoding="utf-8")

    def rename(name):
        backend = SshWorkspaceFs(shell_workspace._env)
        barrier.wait(timeout=10)
        try:
            backend.rename(str(project / name), "winner")
        except SshWorkspaceFsError as exc:
            return name, exc.code
        return name, "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(rename, ("first", "second")))
    assert sorted(results.values()) == ["EEXIST", "ok"]
    winner = next(name for name, result in results.items() if result == "ok")
    loser = next(name for name, result in results.items() if result == "EEXIST")
    assert (project / "winner").read_text(encoding="utf-8-sig") == winner
    assert (project / loser).read_text(encoding="utf-8-sig") == loser
    assert not (project / winner).exists()


@pytest.mark.parametrize("operation", ["create", "rename", "delete", "write"])
@pytest.mark.parametrize("relative", [".ENV.local", "Config.YAML", "pairing/device.json", "mcp-tokens/nested/item"])
@pytest.mark.platforms("posix")
def test_sensitive_entries_and_descendants_are_never_mutated(shell_workspace, project, operation, relative):
    target = project / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"keep")
    actions = {
        "create": lambda: shell_workspace.create(str(target), directory=False),
        "rename": lambda: shell_workspace.rename(str(target), "ordinary"),
        "delete": lambda: shell_workspace.delete(str(target), recursive=True),
        "write": lambda: shell_workspace.write_text(str(target), "changed", max_bytes=100),  # windows-footgun: ok — adapter API, not Path
    }
    _denied(actions[operation])
    assert target.read_bytes() == b"keep"


@pytest.mark.parametrize("operation", ["rename", "delete"])
@pytest.mark.platforms("posix")
def test_parent_mutation_scans_sensitive_descendants(shell_workspace, project, operation):
    descendant = project / "nested" / "deeper" / ".env"
    descendant.parent.mkdir(parents=True)
    descendant.write_bytes(b"keep")
    action = (lambda: shell_workspace.rename(str(project), "renamed")) if operation == "rename" else (
        lambda: shell_workspace.delete(str(project), recursive=True)
    )
    _denied(action)
    assert descendant.read_bytes() == b"keep"
    assert not project.with_name("renamed").exists()


@pytest.mark.parametrize("relative", ["auth.json/child.txt", "pairing/child.txt", ".env/child.txt"])
@pytest.mark.platforms("posix")
def test_create_cannot_enter_sensitive_directories(shell_workspace, project, relative):
    target = project / relative
    target.parent.mkdir()
    _denied(lambda: shell_workspace.create(str(target), directory=False))
    assert not target.exists()


@pytest.mark.platforms("posix")
def test_rename_checks_destination_and_noop_source_policy(shell_workspace, project):
    source = project / "ordinary"
    source.write_text("keep", encoding="utf-8")
    _denied(lambda: shell_workspace.rename(str(source), "auth.json"))
    assert source.read_text(encoding="utf-8-sig") == "keep"
    protected = project / "config.yaml"
    protected.write_text("keep", encoding="utf-8")
    _denied(lambda: shell_workspace.rename(str(protected), protected.name))
    assert protected.read_text(encoding="utf-8-sig") == "keep"
    assert shell_workspace.rename(str(source), source.name) == str(source)
    _denied(lambda: shell_workspace.rename(str(project / "missing"), "missing"), "ENOENT")


@pytest.mark.platforms("linux")
def test_rename_checks_mapped_destination_descendants(shell_workspace):
    home = Path(shell_workspace._env._remote_home)
    source = home / "staging"
    sentinel = source / "gh" / "sentinel.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"keep")
    _denied(lambda: shell_workspace.rename(str(source), ".config"))
    assert sentinel.read_bytes() == b"keep"
    assert not (home / ".config").exists()
    assert shell_workspace.rename(str(source), "ordinary") == str(home / "ordinary")
    assert (home / "ordinary" / "gh" / "sentinel.txt").read_bytes() == b"keep"


@pytest.mark.parametrize("kind,expected", [("existing", 0o754), ("new", 0o664), ("link", 0o664)])
@pytest.mark.platforms("posix")
def test_atomic_save_modes_under_remote_umask(shell_workspace, project, kind, expected):
    original_execute = shell_workspace._env.execute

    def with_umask(command, **kwargs):
        return original_execute("umask 002\n" + command, **kwargs)

    shell_workspace._env.execute = with_umask
    target = project / "file"
    referent = project / "referent"
    if kind == "existing":
        target.write_bytes(b"old")
        target.chmod(0o754)
    elif kind == "link":
        referent.write_bytes(b"keep")
        referent.chmod(0o600)
        target.symlink_to(referent)
    shell_workspace.write_text(str(target), "new", max_bytes=100)  # windows-footgun: ok — adapter API
    assert not target.is_symlink()
    assert target.stat().st_mode & 0o7777 == expected
    assert target.read_bytes() == b"new"
    if kind == "link":
        assert referent.read_bytes() == b"keep"
        assert referent.stat().st_mode & 0o7777 == 0o600


@pytest.mark.parametrize("operation", ["rename", "delete"])
@pytest.mark.platforms("posix")
def test_unreadable_subtree_scan_fails_closed(shell_workspace, project, operation):
    nested = project / "nested"
    nested.mkdir()
    (nested / "child").write_text("keep", encoding="utf-8")
    nested.chmod(0)
    try:
        # Root or DAC-bypass environments cannot exercise a permission-denied scan.
        try:
            list(nested.iterdir())
        except PermissionError:
            pass
        else:
            pytest.skip("This identity can read mode-000 directories")
        action = (lambda: shell_workspace.rename(str(project), "renamed")) if operation == "rename" else (
            lambda: shell_workspace.delete(str(project), recursive=True)
        )
        _denied(action)
        assert project.is_dir()
        assert not project.with_name("renamed").exists()
    finally:
        nested.chmod(0o700)
    assert (nested / "child").read_text(encoding="utf-8-sig") == "keep"


@pytest.mark.parametrize("operation", ["rename", "delete"])
@pytest.mark.parametrize("root_kind", ["workspace", "home", "ancestor"])
@pytest.mark.platforms("posix")
def test_remote_roots_and_ancestors_are_not_movable(shell_workspace, root_kind, operation):
    workspace = Path(shell_workspace.cwd)
    target = {"workspace": workspace, "home": workspace.parent, "ancestor": workspace.parent.parent}[root_kind]
    marker = workspace / "marker"
    marker.write_text("keep", encoding="utf-8")
    action = (lambda: shell_workspace.rename(str(target), "renamed")) if operation == "rename" else (
        lambda: shell_workspace.delete(str(target), recursive=True)
    )
    _denied(action, "EINVAL")
    assert marker.read_text(encoding="utf-8-sig") == "keep"


@pytest.mark.parametrize("operation", ["create", "rename", "delete", "write"])
@pytest.mark.platforms("posix")
def test_symlink_parent_is_refused_even_for_noop_rename(shell_workspace, project, operation):
    actual = project / "actual"
    actual.mkdir()
    target = actual / "file"
    target.write_text("keep", encoding="utf-8")
    link = project / "alias"
    link.symlink_to(actual, target_is_directory=True)
    alias = link / "file"
    actions = {
        "create": lambda: shell_workspace.create(str(alias), directory=False),
        "rename": lambda: shell_workspace.rename(str(alias), alias.name),
        "delete": lambda: shell_workspace.delete(str(alias), recursive=True),
        "write": lambda: shell_workspace.write_text(str(alias), "changed", max_bytes=100),  # windows-footgun: ok — adapter API, not Path
    }
    _denied(actions[operation], "EINVAL")
    assert target.read_text(encoding="utf-8-sig") == "keep"


@pytest.mark.platforms("linux")
def test_rename_and_write_replace_final_link_not_its_directory_referent(shell_workspace, project):
    protected = project / "pairing"
    protected.mkdir()
    sentinel = protected / "device.json"
    sentinel.write_bytes(b"keep")
    link = project / "alias"
    link.symlink_to(protected, target_is_directory=True)
    renamed = Path(shell_workspace.rename(str(link), "renamed"))
    assert renamed.is_symlink()
    assert not link.is_symlink()
    assert shell_workspace.write_text(str(renamed), "replacement", max_bytes=100) == (str(renamed), 11)  # windows-footgun: ok — adapter API, not Path
    assert not renamed.is_symlink()
    assert renamed.read_text(encoding="utf-8-sig") == "replacement"
    assert list(protected.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"keep"


@pytest.mark.platforms("posix")
def test_nonrecursive_nonempty_delete_is_not_reported_as_permission_denial(shell_workspace, project):
    child = project / "child"
    child.write_text("keep", encoding="utf-8")
    _denied(lambda: shell_workspace.delete(str(project), recursive=False), "ENOTEMPTY")
    assert child.read_text(encoding="utf-8-sig") == "keep"


@pytest.mark.platforms("posix")
def test_canonical_credential_names_remain_protected(shell_workspace, project):
    from hermes_cli.web_routers.files import _SENSITIVE_MANAGED_FILE_BASENAMES

    for name in _SENSITIVE_MANAGED_FILE_BASENAMES:
        target = project / name
        target.write_text("keep", encoding="utf-8")
        _denied(lambda: shell_workspace.delete(str(target), recursive=False))
        assert target.read_text(encoding="utf-8-sig") == "keep"


@pytest.mark.parametrize("aliased_home", [False, True])
@pytest.mark.platforms("posix")
def test_remote_home_credential_subtrees_are_protected(shell_workspace, aliased_home):
    home = Path(shell_workspace._env._remote_home)
    if aliased_home:
        alias = home.with_name("home-alias")
        alias.symlink_to(home, target_is_directory=True)
        shell_workspace._env._remote_home = str(alias)
        original_execute = shell_workspace._env.execute

        def alias_home(command, **kwargs):
            return original_execute(f"export HOME={shlex.quote(str(alias))}\n" + command, **kwargs)

        shell_workspace._env.execute = alias_home
    target = home / ".ssh" / "nested" / "key"
    target.parent.mkdir(parents=True)
    target.write_text("synthetic", encoding="utf-8")
    _denied(lambda: shell_workspace.delete(str(target.parent), recursive=True))
    _denied(lambda: shell_workspace.create(str(target.parent / "new"), directory=False))
    assert target.read_text(encoding="utf-8-sig") == "synthetic"


@pytest.mark.platforms("posix")
def test_missing_remote_guard_interpreter_fails_closed(shell_workspace, project):
    empty_bin = project / "empty-bin"
    empty_bin.mkdir()
    original_execute = shell_workspace._env.execute

    def without_python(command, **kwargs):
        return original_execute(f"export PATH={shlex.quote(str(empty_bin))}\n" + command, **kwargs)

    shell_workspace._env.execute = without_python
    target = project / "new"
    _denied(lambda: shell_workspace.create(str(target), directory=False), "EUNSUPPORTED")
    assert not target.exists()
