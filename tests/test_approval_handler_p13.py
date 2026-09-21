"""P13 isolation proof for scripts/approval_handler.py (discord-approval-handler).

Verifies:
- HERMES_HOME / WIKI_DIR parameterisation: STATE, DOTENV, and WIKI_REPOS
  resolve under the env roots, not /home/kensei/.hermes or /home/kensei/wiki.
- --dry-run suppresses every write path: no Discord POST, no wiki page
  rewrite, no kanban task creation, no state file save. Read paths
  (load_state, discord_api GET, find_wiki) run unchanged.
- import-safe: importing the module does not create the state dir.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "approval_handler.py"


def _load_module(monkeypatch, fake_home: Path, fake_wiki: Path):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    monkeypatch.setenv("WIKI_DIR", str(fake_wiki))
    scripts_dir = REPO_ROOT / "scripts"
    for pth in (str(scripts_dir), str(REPO_ROOT)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    spec = importlib.util.spec_from_file_location(
        "approval_handler_under_test", str(SCRIPT)
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_layout(tmp_path):
    fake_home = tmp_path / "fake_hermes"
    fake_home.mkdir()
    fake_wiki = tmp_path / "fake_wiki"
    (fake_wiki / "repos").mkdir(parents=True)
    return fake_home, fake_wiki


def test_paths_resolve_under_env_roots(monkeypatch, fake_layout):
    fake_home, fake_wiki = fake_layout
    mod = _load_module(monkeypatch, fake_home, fake_wiki)
    assert str(mod.STATE).startswith(str(fake_home))
    assert "approval-handler-state.json" in str(mod.STATE)
    assert mod.WIKI_REPOS == fake_wiki / "repos"


def test_import_is_side_effect_free(monkeypatch, fake_layout):
    fake_home, fake_wiki = fake_layout
    assert not (fake_home / "state").exists()
    _load_module(monkeypatch, fake_home, fake_wiki)
    assert not (fake_home / "state").exists(), "import created the state dir"


def test_dry_run_writes_nothing_and_exits_zero(monkeypatch, fake_layout, tmp_path):
    """--dry-run must not write STATE, must not POST to Discord, and must
    exit 0 even with no token and no wiki repos present."""
    fake_home, fake_wiki = fake_layout
    env = dict(os.environ)
    env["HERMES_HOME"] = str(fake_home)
    env["WIKI_DIR"] = str(fake_wiki)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("DISCORD_BOT_TOKEN", None)
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
    assert not (fake_home / "state" / "approval-handler-state.json").exists()


def test_dry_run_does_not_rewrite_wiki(monkeypatch, fake_layout):
    """With a wiki page present and update_wiki called, --dry-run must
    NOT rewrite the page (adoption_status stays unchanged)."""
    fake_home, fake_wiki = fake_layout
    mod = _load_module(monkeypatch, fake_home, fake_wiki)
    page = fake_wiki / "repos" / "test-repo.md"
    original = (
        "---\\nadoption_status: evaluated\\nupdated: 2026-01-01\\ntags: [gitradar]\\n---\\nbody\\n"
    )
    page.write_text(original)
    mod._DRY_RUN = True
    result = mod.update_wiki(page)
    assert result is True, "dry-run update_wiki should return True (would-update)"
    # Page content must be unchanged.
    assert page.read_text() == original, "dry-run rewrote the wiki page"


def test_dry_run_kanban_task_returns_none(monkeypatch, fake_layout):
    """--dry-run must not call the hermes kanban CLI to create a task."""
    fake_home, fake_wiki = fake_layout
    mod = _load_module(monkeypatch, fake_home, fake_wiki)
    mod._DRY_RUN = True
    assert mod.kanban_task("some/repo") is None
