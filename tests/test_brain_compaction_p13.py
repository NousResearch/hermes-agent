"""P13 isolation proof for scripts/brain-compaction.py.

Verifies:
- KENSEI_REPO_ROOT parameterisation: REPO resolves under the env root.
- --dry-run suppresses the brain page rewrite (fp.write_text). Read
  paths (_list_brain_pages, _extract_dated_bullets, _find_bullet_blocks)
  run unchanged so the compaction decisions are still computed.
- import-safe: importing the module does not touch the brain dir.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "brain-compaction.py"


def _load_module(monkeypatch, fake_repo: Path, fake_brain: Path):
    monkeypatch.setenv("KENSEI_REPO_ROOT", str(fake_repo))
    monkeypatch.setenv("GBRAIN_REPO", str(fake_brain))
    scripts_dir = REPO_ROOT / "scripts"
    for pth in (str(scripts_dir), str(REPO_ROOT)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    spec = importlib.util.spec_from_file_location(
        "brain_compaction_under_test", str(SCRIPT)
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_layout(tmp_path):
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    fake_brain = tmp_path / "fake_brain"
    fake_brain.mkdir()
    return fake_repo, fake_brain


def test_repo_resolves_under_env_root(monkeypatch, fake_layout):
    fake_repo, _ = fake_layout
    mod = _load_module(monkeypatch, fake_repo, _)
    # REPO is only used for sys.path; just confirm it picked up the env.
    assert str(mod.REPO).startswith(str(fake_repo))


def test_import_is_side_effect_free(monkeypatch, fake_layout, tmp_path):
    fake_repo, fake_brain = fake_layout
    # Brain dir exists but must not be mutated by import.
    page = fake_brain / "notes.md"
    page.write_text("# notes\n")
    _load_module(monkeypatch, fake_repo, fake_brain)
    assert page.read_text() == "# notes\n"


def test_dry_run_does_not_rewrite_brain_page(monkeypatch, fake_layout):
    """With a brain page containing >=5 dated bullets, --dry-run must
    NOT rewrite the page. The compaction decision is still computed."""
    fake_repo, fake_brain = fake_layout
    mod = _load_module(monkeypatch, fake_repo, fake_brain)
    # Build a brain page with 5+ dated bullets (>= default threshold).
    page = fake_brain / "projects" / "test.md"
    page.parent.mkdir(parents=True)
    original = (
        "# Test Project\\n\\n"
        "- 2026-01-01: first fact\\n"
        "- 2026-01-02: second fact\\n"
        "- 2026-01-03: third fact\\n"
        "- 2026-01-04: fourth fact\\n"
        "- 2026-01-05: fifth fact\\n"
    )
    page.write_text(original)
    mod._DRY_RUN = True
    rc = mod.main()
    assert rc == 0
    # Page content must be unchanged (no compaction marker added).
    assert page.read_text() == original, "dry-run rewrote the brain page"
    assert "<!-- compacted" not in page.read_text()


def test_dry_run_main_exits_zero_empty_brain(monkeypatch, fake_layout, tmp_path):
    """--dry-run with an empty brain dir must exit 0 ([SILENT])."""
    fake_repo, fake_brain = fake_layout
    env = dict(os.environ)
    env["KENSEI_REPO_ROOT"] = str(fake_repo)
    env["GBRAIN_REPO"] = str(fake_brain)
    env["PYTHONPATH"] = str(REPO_ROOT)
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"dry-run failed: rc={proc.returncode} stderr={proc.stderr!r} stdout={proc.stdout!r}"
    )
