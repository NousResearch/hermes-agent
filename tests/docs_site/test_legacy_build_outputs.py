"""Old docs builds must not make an upgraded source checkout look modified."""

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_legacy_docs_outputs_stay_ignored_but_catalog_changes_remain_visible(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    shutil.copyfile(ROOT / ".gitignore", repo / ".gitignore")
    catalog = repo / "website/static/api/model-catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text('{"version": 1}\n', encoding="utf-8")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
        ).stdout

    # No forced add: the compatibility artifact must remain trackable.
    git("add", ".gitignore", "website/static/api/model-catalog.json")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
    for relative in [
        "website/build/index.html",
        "website/.docusaurus/client-manifest.json",
        "website/static/api/skills.json",
        "website/static/api/plugins.json",
        "website/static/llms.txt",
        "website/static/llms-full.txt",
    ]:
        artifact = repo / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("generated output\n", encoding="utf-8")

    assert git("status", "--porcelain", "--untracked-files=all") == ""
    catalog.write_text('{"version": 2}\n', encoding="utf-8")
    assert git("status", "--porcelain", "--untracked-files=all").strip() == "M website/static/api/model-catalog.json"


def test_other_website_paths_stay_trackable(tmp_path):
    """The upgrade E2E publishes website/e2e-hostile-pause-i.txt with a plain
    `git add`; only the named Docusaurus leftovers may be ignored."""
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    shutil.copyfile(ROOT / ".gitignore", repo / ".gitignore")
    fixture = repo / "website/e2e-hostile-pause-i.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("release i\n", encoding="utf-8")
    added = subprocess.run(
        ["git", "-C", str(repo), "add", "--", "website/e2e-hostile-pause-i.txt"],
        check=False, capture_output=True, text=True,
    )
    assert added.returncode == 0, added.stderr
