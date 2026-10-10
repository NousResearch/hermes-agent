"""Frontend build staging residue must be gitignored so a leftover staging dir
cannot make every dirty-tree check report a phantom local change (#125147).

The frontend product builds stage into a sibling scratch dir of the output and
publish it when complete:

- web: ``scripts/build/frontend-common.mjs`` `withProduct()` creates
  ``hermes_cli/.web_dist-build-*`` next to the ignored ``hermes_cli/web_dist/``
- tui (explicit ``--out .../hermes_cli/tui_dist``): same ``withProduct()``
  mechanism, scratch ``hermes_cli/.tui_dist-build-*``
- tui (developer build, no ``--out``): ``scripts/build/tui.mjs`` stages via
  ``mkdtempSync(join(repoRoot, 'ui-tui/.dist-'))``

On Windows the ``rmSync`` cleanup can fail when a file is briefly locked
(AV scanners, a running gateway serving assets), leaving the staging dir as
untracked files. ``hermes_cli/source_stamp.py`` and the update maintenance path
run the dirty-tree check with untracked files included, so leftover residue
reports a permanently dirty tree (``.dirty`` version suffix) and trips the
updater's autostash on every update. ``apps/desktop/.dist-build*`` is already
ignored for the desktop build's scratch (#125147 extends the same coverage).
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Staging dirs exactly as the builds create them: mkdtemp-style random suffixes
# under the output's parent, with published-looking content inside.
STAGING_RESIDUE = (
    "hermes_cli/.web_dist-build-Ab12cd/product/index.html",
    "hermes_cli/.web_dist-build-Ab12cd/product/assets/app.js",
    "hermes_cli/.tui_dist-build-Ef34gh/product/dist/entry.js",
    "ui-tui/.dist-Ij56kl/entry.js",
    "ui-tui/.dist-Ij56kl/package.json",
)


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )


@pytest.fixture
def checkout_repo(tmp_path: Path) -> Path:
    """A real git repo standing in for a checkout, with the tracked .gitignore set.

    Built in a subdirectory of tmp_path: the suite-wide HERMES_HOME isolation
    fixture materialises its own tree in tmp_path itself, which is not part of
    this repo's story.
    """
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    shutil.copyfile(REPO_ROOT / ".gitignore", repo / ".gitignore")
    (repo / "ui-tui").mkdir()
    shutil.copyfile(REPO_ROOT / "ui-tui" / ".gitignore", repo / "ui-tui" / ".gitignore")
    (repo / "app.py").write_text("print('hermes')\n")
    _run_git(repo, "add", ".gitignore", "ui-tui/.gitignore", "app.py")
    _run_git(
        repo,
        "-c", "user.email=t@t", "-c", "user.name=t",
        "commit", "-qm", "init",
    )
    return repo


def test_build_staging_residue_is_ignored(checkout_repo):
    """`git status --porcelain` must stay empty with leftover build staging dirs
    present, so the source stamp and the updater's autostash never see them."""
    for rel in STAGING_RESIDUE:
        path = checkout_repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"staged build output")
    status = _run_git(
        checkout_repo, "status", "--porcelain", "--untracked-files=all"
    )
    assert status.stdout == "", status.stdout


def test_final_outputs_stay_ignored_with_staging_entries(checkout_repo):
    """The contract runs both ways: the published outputs (``web_dist/``,
    ``tui_dist/``, ``ui-tui/dist/``) must remain ignored — normalizing
    ``tui_dist/*`` to ``tui_dist/`` must not un-ignore the output itself."""
    outputs = (
        "hermes_cli/web_dist/index.html",
        "hermes_cli/tui_dist/entry.js",
        "ui-tui/dist/entry.js",
    )
    for rel in outputs:
        path = checkout_repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"published output")
    status = _run_git(
        checkout_repo, "status", "--porcelain", "--untracked-files=all"
    )
    assert status.stdout == "", status.stdout
