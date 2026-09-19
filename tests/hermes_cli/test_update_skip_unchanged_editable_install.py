"""``hermes update`` skips the editable reinstall when the pull can't affect it.

``uv pip install -e .`` never audits an editable target — it reinstalls on
every invocation and rewrites the console-script shims each time. On Windows
that rewrite is the only reason the running ``hermes.exe`` gets quarantined,
and a quarantine that loses its race is the ``os error 32`` family. The gate
under test removes the reinstall (and therefore the rename) for any update
that touches none of the files defining the install.

These tests drive a REAL git repository. The predicate is a ``git diff``
pathspec against the pre-pull SHA; mocking git would assert our idea of what
git prints rather than what it does.
"""

import subprocess

import pytest

from hermes_cli.update_cmd import _editable_install_is_current

GIT = ["git"]


@pytest.fixture
def repo(tmp_path):
    """A repo with one commit, standing in for the pre-pull checkout."""
    subprocess.run(GIT + ["init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        GIT + ["config", "user.email", "t@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(GIT + ["config", "user.name", "t"], cwd=tmp_path, check=True)
    subprocess.run(GIT + ["config", "commit.gpgSign", "false"], cwd=tmp_path, check=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'hermes'\n")
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "__init__.py").write_text("")
    (tmp_path / "cli.py").write_text("x = 1\n")
    subprocess.run(GIT + ["add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(GIT + ["commit", "-qm", "base"], cwd=tmp_path, check=True)
    return tmp_path


def _head(cwd):
    return subprocess.run(
        GIT + ["rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _commit(cwd, message):
    subprocess.run(GIT + ["add", "-A"], cwd=cwd, check=True)
    subprocess.run(GIT + ["commit", "-qm", message], cwd=cwd, check=True)


def test_source_only_pull_skips_the_reinstall(repo):
    """The common update: .py churn inside already-mapped packages."""
    before = _head(repo)
    (repo / "cli.py").write_text("x = 2\n")
    (repo / "agent" / "loop.py").write_text("y = 1\n")
    _commit(repo, "source churn")

    assert _editable_install_is_current(GIT, repo, before) is True


def test_new_submodule_in_mapped_package_skips_the_reinstall(repo):
    """A new file inside an existing package resolves through its __path__."""
    before = _head(repo)
    (repo / "agent" / "brand_new.py").write_text("z = 1\n")
    _commit(repo, "new submodule")

    assert _editable_install_is_current(GIT, repo, before) is True


@pytest.mark.parametrize("source,destination", [
    (None, "new_module.py"),
    ("cli.py", None),
    ("cli.py", "renamed.py"),
    ("agent/__init__.py", "new_module.py"),
    ("cli.py", "agent/cli.py"),
])
def test_root_module_inventory_change_forces_reinstall(repo, source, destination):
    """The editable finder must be refreshed when its root module names change."""
    before = _head(repo)
    if source is None:
        (repo / destination).write_text("x = 1\n", encoding="utf-8")
    elif destination is None:
        (repo / source).unlink()
    else:
        (repo / source).rename(repo / destination)
    _commit(repo, "change root module inventory")

    assert _editable_install_is_current(GIT, repo, before) is False


@pytest.mark.parametrize("source,destination,stage,expected", [
    (None, "new_module.py", False, False),
    (None, "new_module.py", True, False),
    ("cli.py", None, False, False),
    ("cli.py", "renamed.py", False, False),
    ("cli.py", "renamed.py", True, False),
    ("cli.py", "cli.py", False, True),
    (None, "agent/new_module.py", False, True),
])
def test_restored_autostash_inventory_controls_reinstall(repo, source, destination, stage, expected):
    """Reinstall decisions include local inventory restored after a source-only pull."""
    before = _head(repo)
    if source is None or source == destination:
        (repo / destination).write_text("x = 2\n", encoding="utf-8")
    elif destination is None:
        (repo / source).unlink()
    else:
        (repo / source).rename(repo / destination)
    if stage:
        subprocess.run(GIT + ["add", "-A"], cwd=repo, check=True)
    subprocess.run(GIT + ["stash", "push", "--include-untracked"], cwd=repo, check=True)
    (repo / "agent" / "update.py").write_text("x = 3\n", encoding="utf-8")
    _commit(repo, "source-only update")
    subprocess.run(GIT + ["stash", "pop", "--index"], cwd=repo, check=True)

    assert _editable_install_is_current(GIT, repo, before) is expected


@pytest.mark.parametrize(
    "filename",
    ["pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in", "uv.lock"],
)
def test_touching_a_file_that_defines_the_install_forces_the_reinstall(repo, filename):
    """Dependencies, entry points and the static module list all live here."""
    before = _head(repo)
    (repo / filename).write_text("# changed\n")
    _commit(repo, f"touch {filename}")

    assert _editable_install_is_current(GIT, repo, before) is False


def test_source_churn_alongside_a_pyproject_edit_still_reinstalls(repo):
    """The gate must not be fooled by burying the pyproject diff in noise."""
    before = _head(repo)
    (repo / "cli.py").write_text("x = 3\n")
    (repo / "pyproject.toml").write_text("[project]\nname = 'hermes'\ndeps = []\n")
    _commit(repo, "mixed")

    assert _editable_install_is_current(GIT, repo, before) is False


def test_missing_pre_pull_sha_fails_closed(repo):
    """No SHA (ZIP swap, capture failure) means we cannot prove anything."""
    assert _editable_install_is_current(GIT, repo, None) is False
    assert _editable_install_is_current(GIT, repo, "") is False


def test_unresolvable_pre_pull_sha_fails_closed(repo):
    """A shallow checkout whose base commit isn't present reinstalls as before."""
    assert _editable_install_is_current(GIT, repo, "0" * 40) is False


def test_unusable_git_fails_closed(repo):
    """A git that cannot be executed must not be read as 'nothing changed'."""
    before = _head(repo)
    assert _editable_install_is_current(["definitely-not-git"], repo, before) is False
