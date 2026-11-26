"""scripts/ci/check_agents_md_size.py: every AGENTS.md chain an agent can load stays under the cap it is truncated at."""

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_agents_md_size.py"
_SPEC = importlib.util.spec_from_file_location("_check_agents_md_size", SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ROOT_CAP, CHAIN_CAP = _MODULE.ROOT_MAX_CHARS, _MODULE.CHAIN_MAX_CHARS


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
    fits = {"AGENTS.md": ROOT_CAP, "app/AGENTS.md": CHAIN_CAP - ROOT_CAP - 8_000, "app/src/AGENTS.md": 8_000,
            "lib/AGENTS.md": 18_000}
    ok = _check(_repo(tmp_path / "ok", fits))
    assert ok.returncode == 0, ok.stdout
    nested = _check(_repo(tmp_path / "nested", {**fits, "app/src/AGENTS.md": 8_001}))
    assert nested.returncode == 1
    assert f"AGENTS.md + app/AGENTS.md + app/src/AGENTS.md: {CHAIN_CAP + 1} chars > {CHAIN_CAP} chain cap" in nested.stdout
    assert "lib/" not in nested.stdout and "untracked" not in nested.stdout
    root = _check(_repo(tmp_path / "root", {"AGENTS.md": ROOT_CAP + 1}))
    assert root.returncode == 1 and f"AGENTS.md: {ROOT_CAP + 1} chars > {ROOT_CAP} cap" in root.stdout


def test_a_chain_at_the_cap_loads_whole_on_a_128k_model(tmp_path):
    """The cap exists so the loader never truncates: a three-file chain at CHAIN_MAX_CHARS, with the
    ``## <label>`` header the loader puts on each file, still fits its 128k budget."""
    from agent.prompt_builder import _load_agents_md

    area = "hermes_agent/gateway/platforms"
    repo = _repo(tmp_path, {"AGENTS.md": ROOT_CAP, "hermes_agent/gateway/AGENTS.md": CHAIN_CAP - ROOT_CAP - 8_000,
                            f"{area}/AGENTS.md": 8_000})
    assert _check(repo).returncode == 0
    assert "[...truncated" not in _load_agents_md(repo / area, context_length=128_000)
