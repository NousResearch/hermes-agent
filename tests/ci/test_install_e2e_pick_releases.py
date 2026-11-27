"""install-e2e's pick-releases step resolves an explicit install-ref from origin on tagless checkouts."""
import json
import subprocess
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from tests.ci import workflow_steps

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "install-e2e.yml"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _tagless_clone(root: Path) -> Path:
    origin = root / "origin.git"
    _git(root, "init", "--bare", "-b", "main", str(origin))
    seed = root / "seed"
    _git(root, "init", "-b", "main", str(seed))
    _git(seed, "config", "user.name", "fixture")
    _git(seed, "config", "user.email", "fixture@example.invalid")
    (seed / "README.md").write_text("old\n", encoding="utf-8")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-m", "v2026.3.12")
    _git(seed, "tag", "v2026.3.12")
    desktop_dir = seed / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}\n", encoding="utf-8")
    _git(seed, "add", "apps/desktop/package.json")
    _git(seed, "commit", "-m", "v0.23.2")
    _git(seed, "tag", "v0.23.2")
    _git(seed, "push", str(origin), "main", "refs/tags/*:refs/tags/*")
    clone = root / "tagless"
    _git(root, "clone", "--no-local", "--no-tags", str(origin), str(clone))
    return clone


def _pick_step() -> dict:
    workflow = YAML(typ="base").load(WORKFLOW.read_text(encoding="utf-8"))
    return next(step for step in workflow["jobs"]["pick-releases"]["steps"] if step.get("id") == "pick")


@pytest.mark.platforms("linux")
@pytest.mark.parametrize(("install_ref", "has_desktop"), [
    ("v2026.3.12", False),
    ("v0.23.2", True),
])
def test_explicit_install_ref_fetches_missing_tag_from_origin(tmp_path, install_ref, has_desktop):
    clone = _tagless_clone(tmp_path)
    ctx = {"inputs": {"tag-count": "3", "install-ref": install_ref, "exclude-ref": "", "release": False}}
    out = workflow_steps.outputs(_pick_step(), ctx, cwd=clone)
    assert json.loads(out["tags"]) == [{"ref": install_ref, "desktop": has_desktop}]
    assert _git(clone, "tag", "--list").stdout.split() == [install_ref]
