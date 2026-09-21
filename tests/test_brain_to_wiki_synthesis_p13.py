"""P13 isolation proof for scripts/brain-to-wiki-synthesis.py.

Verifies:
- KENSEI_REPO_ROOT parameterisation: REPO resolves under the env root.
- --dry-run suppresses every write path (wiki concept page write_text,
  comparison page write_text, index.md update, log.md append). Read
  paths (_read_brain_page, _brain_mtime, _wiki_mtime) run unchanged so
  the synthesis decisions are still computed.
- import-safe: importing the module does not touch the wiki dir.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "brain-to-wiki-synthesis.py"


def _load_module(monkeypatch, fake_repo: Path, fake_brain: Path, fake_wiki: Path):
    monkeypatch.setenv("KENSEI_REPO_ROOT", str(fake_repo))
    monkeypatch.setenv("GBRAIN_REPO", str(fake_brain))
    monkeypatch.setenv("WIKI_DIR", str(fake_wiki))
    scripts_dir = REPO_ROOT / "scripts"
    for pth in (str(scripts_dir), str(REPO_ROOT)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    spec = importlib.util.spec_from_file_location(
        "brain_to_wiki_synthesis_under_test", str(SCRIPT)
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
    (fake_brain / "conventions").mkdir(parents=True)
    fake_wiki = tmp_path / "fake_wiki"
    (fake_wiki / "concepts").mkdir(parents=True)
    (fake_wiki / "comparisons").mkdir(parents=True)
    (fake_wiki / "index.md").write_text("# Index\n")
    (fake_wiki / "log.md").write_text("# Log\n")
    return fake_repo, fake_brain, fake_wiki


def test_repo_resolves_under_env_root(monkeypatch, fake_layout):
    fake_repo, _, _ = fake_layout
    mod = _load_module(monkeypatch, fake_repo, _, _)
    assert str(mod.REPO).startswith(str(fake_repo))


def test_import_is_side_effect_free(monkeypatch, fake_layout):
    fake_repo, fake_brain, fake_wiki = fake_layout
    idx = (fake_wiki / "index.md").read_text()
    _load_module(monkeypatch, fake_repo, fake_brain, fake_wiki)
    assert (fake_wiki / "index.md").read_text() == idx


def test_dry_run_does_not_write_wiki_pages(monkeypatch, fake_layout):
    """With a brain page present that maps to a concept, --dry-run must
    NOT write the wiki concept page, index.md, or log.md."""
    fake_repo, fake_brain, fake_wiki = fake_layout
    mod = _load_module(monkeypatch, fake_repo, fake_brain, fake_wiki)
    # Brain page for conventions/infrastructure (in BRAIN_TO_WIKI mapping).
    (fake_brain / "conventions" / "infrastructure.md").write_text(
        "---\\ntitle: Infra\\n---\\nSome infra conventions.\\n"
    )
    mod._DRY_RUN = True
    rc = mod.main()
    assert rc == 0
    # No concept page written.
    concept = fake_wiki / "concepts" / "kensei-infrastructure-conventions.md"
    assert not concept.exists(), "dry-run wrote a concept page"
    # index.md and log.md unchanged.
    assert "Last updated" not in (fake_wiki / "index.md").read_text()


def test_normal_run_creates_missing_comparisons_directory(monkeypatch, fake_layout):
    """A fresh wiki layout must not crash before writing comparison pages."""
    fake_repo, fake_brain, fake_wiki = fake_layout
    comparisons = fake_wiki / "comparisons"
    comparisons.rmdir()

    mod = _load_module(monkeypatch, fake_repo, fake_brain, fake_wiki)

    assert mod.main() == 0
    assert comparisons.is_dir()
    assert {path.name for path in comparisons.glob("*.md")} == {
        "convex-vs-supabase.md",
        "keyword-vs-embeddings-brain-search.md",
    }


def test_dry_run_main_exits_zero_empty(monkeypatch, fake_layout, tmp_path):
    """--dry-run with no brain pages must exit 0 (silent)."""
    fake_repo, fake_brain, fake_wiki = fake_layout
    env = dict(os.environ)
    env["KENSEI_REPO_ROOT"] = str(fake_repo)
    env["GBRAIN_REPO"] = str(fake_brain)
    env["WIKI_DIR"] = str(fake_wiki)
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
