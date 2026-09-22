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
from pathlib import Path

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


def _finder(repo, mapping, *, layout="lib/python3.11/site-packages", annotated=False):
    site = repo / "venv" / Path(layout)
    site.mkdir(parents=True)
    body = (
        "MAPPING: dict[str, str] = " + repr(mapping) + "\n"
        if annotated
        else "MAPPING = " + repr(mapping) + "\n"
    )
    (site / "__editable___hermes_agent_0_21_4_finder.py").write_text(body, encoding="utf-8")
    return site


def _declare_packages(repo, include):
    (repo / "pyproject.toml").write_text(
        "[tool.setuptools.packages.find]\ninclude = " + repr(include) + "\n",
        encoding="utf-8",
    )
    _commit(repo, "declare packages")


@pytest.mark.parametrize("annotated", [False, True])
def test_stale_map_refuses_the_skip(repo, monkeypatch, annotated):
    """A missing top-level name refuses the skip for both finder spellings."""
    _declare_packages(repo, ["agent", "agent.*", "newpkg", "newpkg.*"])
    before = _head(repo)
    (repo / "newpkg").mkdir()
    (repo / "newpkg" / "__init__.py").write_text("")
    _commit(repo, "add package")
    _finder(repo, {"agent": str(repo / "agent"), "cli": str(repo / "cli")}, annotated=annotated)
    monkeypatch.setattr("hermes_cli.update_cmd_deps.project_venv_dir", lambda _: repo / "venv")
    assert _editable_install_is_current(GIT, repo, before) is False


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("layout", ["lib/python3.11/site-packages", "Lib/site-packages"])
def test_covered_map_skips_source_churn(repo, monkeypatch, layout, annotated):
    """A map that already names the checkout still skips source edits."""
    _declare_packages(repo, ["agent", "agent.*"])
    _finder(
        repo,
        {"agent": str(repo / "agent"), "cli": str(repo / "cli")},
        layout=layout,
        annotated=annotated,
    )
    monkeypatch.setattr("hermes_cli.update_cmd_deps.project_venv_dir", lambda _: repo / "venv")
    before = _head(repo)
    (repo / "agent" / "loop.py").write_text("y = 2\n")
    _commit(repo, "source churn")
    assert _editable_install_is_current(GIT, repo, before) is True


def test_unreadable_finder_fails_closed(repo, monkeypatch):
    before = _head(repo)
    site = repo / "venv" / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    (site / "__editable___hermes_agent_0_21_4_finder.py").write_text("MAPPING = (\n", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.update_cmd_deps.project_venv_dir", lambda _: repo / "venv")
    assert _editable_install_is_current(GIT, repo, before) is False


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
