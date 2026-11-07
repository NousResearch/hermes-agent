"""scripts/ci/check_agents_md_size.py: every AGENTS.md chain an agent can load stays under the cap it is truncated at."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_agents_md_size.py"


def _repo(tmp_path, files):
    (tmp_path / "agent").mkdir(parents=True)
    (tmp_path / "agent" / "subdirectory_hints.py").write_text("_MAX_HINT_CHARS = 32_000\n")
    for rel, chars in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x" * chars)
    (tmp_path / "untracked").mkdir()
    (tmp_path / "untracked" / "AGENTS.md").write_text("u" * 40_000)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "agent", *files], cwd=tmp_path, check=True)
    return tmp_path


def _check(repo):
    return subprocess.run([sys.executable, str(SCRIPT), str(repo)], capture_output=True, text=True, check=False)


def test_budget_is_per_chain_from_the_root_down(tmp_path):
    # Each file alone is small; root + app + app/src together is what an agent working in app/src loads.
    fits = {"AGENTS.md": 12_000, "app/AGENTS.md": 10_000, "app/src/AGENTS.md": 8_000, "lib/AGENTS.md": 18_000}
    ok = _check(_repo(tmp_path / "ok", fits))
    assert ok.returncode == 0, ok.stdout
    nested = _check(_repo(tmp_path / "nested", {**fits, "app/src/AGENTS.md": 8_001}))
    assert nested.returncode == 1
    assert "AGENTS.md + app/AGENTS.md + app/src/AGENTS.md: 30001 chars > 30000 chain cap" in nested.stdout
    assert "lib/" not in nested.stdout and "untracked" not in nested.stdout
    root = _check(_repo(tmp_path / "root", {"AGENTS.md": 12_001}))
    assert root.returncode == 1 and "AGENTS.md: 12001 chars > 12000 cap" in root.stdout
