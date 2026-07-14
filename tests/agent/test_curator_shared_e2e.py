"""Orchestrator-level e2e: run_curator_review drives the shared split.

AC3's remaining clauses, exercised through the REAL orchestrator entry
point against a temp HERMES_HOME git repo (real imports, LLM stubbed):

- the split fires from run_curator_review with consolidate=False (it is
  wired BEFORE/OUTSIDE the _llm_pass `if not consolidate:` early-return)
- the run summary carries the shared: line
- a subsequent skill_manage(action=patch) on the now-lean shared SKILL.md
  succeeds (the original >100 KB failure mode is gone)
- AC1 byte-identity: with include_shared_dirs absent, the same run touches
  nothing in skills-shared/
"""

from __future__ import annotations

import importlib
import json
import subprocess
import textwrap
from pathlib import Path

import pytest


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
    )


def _mk_big_skill(root: Path, group: str, name: str, kb: int = 140) -> Path:
    d = root / group / name
    d.mkdir(parents=True, exist_ok=True)
    filler = ("filler content line for the oversized skill body " * 20).strip()
    parts = [f"---\nname: {name}\ndescription: big\n---\n\nIntro.\n\n"]
    i = 0
    while sum(len(p) for p in parts) < kb * 1024:
        parts.append(f"## Section {i:03d}\n\n" + (filler + "\n") * 12 + "\n")
        i += 1
    (d / "SKILL.md").write_text("".join(parts), encoding="utf-8")
    return d


@pytest.fixture
def orch_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    shared = home / "skills-shared"
    (shared / "smart-home").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    _git(home, "init", "-q")
    _git(home, "config", "user.email", "curator-test@example.com")
    _git(home, "config", "user.name", "Curator Test")

    big = _mk_big_skill(shared, "smart-home", "clanker-e2e")

    (home / "config.yaml").write_text(textwrap.dedent(f"""\
        skills:
          external_dirs:
            - {shared / 'smart-home'}
        curator:
          include_shared_dirs: true
          split_over_kb: 100
          consolidate: false
        """), encoding="utf-8")

    _git(home, "add", "-A")
    _git(home, "commit", "-q", "-m", "baseline")

    import agent.skill_utils as su
    importlib.reload(su)
    import agent.curator_shared as cs
    importlib.reload(cs)
    import tools.skill_usage as usage
    importlib.reload(usage)
    import agent.curator as curator
    importlib.reload(curator)

    monkeypatch.setattr(curator, "_run_llm_review", lambda prompt: {
        "final": "", "summary": "stub", "model": "", "provider": "",
        "tool_calls": [], "error": None,
    })
    monkeypatch.setattr(curator, "_load_config", lambda: {
        "include_shared_dirs": True,
        "split_over_kb": 100,
        "consolidate": False,
    })

    yield {"home": home, "shared": shared, "big": big, "curator": curator}


def test_run_curator_review_splits_shared_with_consolidate_false(orch_env):
    curator = orch_env["curator"]
    summaries = []
    curator.run_curator_review(
        on_summary=summaries.append, synchronous=True, consolidate=False,
    )
    # split fired despite consolidate=False (before the early-return)
    post = (orch_env["big"] / "SKILL.md").stat().st_size
    assert post < 100 * 1024
    log = _git(orch_env["home"], "log", "-1", "--pretty=%s").stdout.strip()
    assert log.startswith("curator: split")
    assert any("shared:" in s for s in summaries)
    # references carve files exist and the manifest records the carve map
    manifest = orch_env["big"] / "references" / ".split-manifest.json"
    assert manifest.exists()
    carves = json.loads(manifest.read_text(encoding="utf-8"))["carves"]
    assert carves
    for c in carves:
        assert (orch_env["big"] / c["file"]).exists()


def test_post_split_skill_manage_patch_succeeds(orch_env, monkeypatch):
    """AC3 final clause: the original failure mode (SKILL.md over the 100 KB
    patch cap) is gone — a real skill_manage(action=patch) lands."""
    curator = orch_env["curator"]
    curator.run_curator_review(synchronous=True, consolidate=False)
    assert (orch_env["big"] / "SKILL.md").stat().st_size < 100 * 1024

    import tools.skill_manager_tool as smt
    importlib.reload(smt)
    result = json.loads(smt.skill_manage(
        action="patch", name="clanker-e2e",
        old_string="Intro.", new_string="Intro. (patched post-split)",
    ))
    assert result.get("success"), result
    assert "(patched post-split)" in (
        orch_env["big"] / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_ac1_flag_off_run_is_no_op_on_shared(orch_env, monkeypatch):
    curator = orch_env["curator"]
    monkeypatch.setattr(curator, "_load_config", lambda: {
        "split_over_kb": 100,  # include_shared_dirs absent → off
        "consolidate": False,
    })
    before = (orch_env["big"] / "SKILL.md").read_bytes()
    curator.run_curator_review(synchronous=True, consolidate=False)
    assert (orch_env["big"] / "SKILL.md").read_bytes() == before
    log = _git(orch_env["home"], "log", "-1", "--pretty=%s").stdout.strip()
    assert log == "baseline"
    st = _git(orch_env["home"], "status", "--porcelain",
              "--untracked-files=all", "--",
              "skills-shared/").stdout.strip()
    # nothing but (possibly) the lockfile
    lines = [ln for ln in st.splitlines() if ".curator.lock" not in ln]
    assert lines == []
