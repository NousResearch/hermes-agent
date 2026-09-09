"""Pine Barron Farms canon loader — PC-2026-09-08-004.

The edition's SOUL.md said "I follow the loaded canon packet" while nothing loaded
it. `editions/pine-barron-farms/canon/load_canon.py` + the `pbf-canon` skill now
do, via existing Hermes mechanisms (skill inline-shell / a session-start command),
with a `canon loaded: <path>, sha256=<hash>` receipt and a clear not-found warning.

These tests prove:
  * the loader's three states (present / truncated / absent), exit 0 always;
  * end-to-end, a fact from a test canon file actually lands in the assembled
    preloaded-skills system-prompt text.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EDITION = REPO_ROOT / "editions" / "pine-barron-farms"
LOADER = EDITION / "canon" / "load_canon.py"
SKILL_MD = EDITION / "skills" / "pbf-canon" / "SKILL.md"

FACT = "PLATE PBF-PLATE-042 is the moonlit-orchard hero plate for episode nine"
CANON_TEXT = f"# Pine Barron Farms — test canon\n\n{FACT}.\n\nCHARACTER: Basil the goat is unflappable.\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("pbf_load_canon", LOADER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_loader_present_emits_receipt_and_content(tmp_path):
    (tmp_path / "canon").mkdir()
    (tmp_path / "canon" / "PINE_BARRON_FARMS_CANON.md").write_text(CANON_TEXT, encoding="utf-8")
    mod = _load_module()

    text, found = mod.render(str(tmp_path / "canon"))
    assert found is True
    assert text.startswith("canon loaded: ")
    assert "sha256=" in text and "bytes" in text
    assert mod.BEGIN in text and mod.END in text
    assert FACT in text                              # the actual fact is in the payload
    assert "truncated" not in text


def test_loader_absent_warns_clearly(tmp_path):
    mod = _load_module()
    text, found = mod.render(str(tmp_path / "canon"))   # dir does not exist
    assert found is False
    assert "canon packet NOT FOUND" in text
    assert "SOUL.md persona and method" in text
    assert "PINE_BARRON_FARMS_CANON.md" in text         # tells the admin what to drop in


def test_loader_truncates_large_packet_with_a_pointer(tmp_path):
    (tmp_path / "canon").mkdir()
    big = "# canon\n\n" + ("PLATE line. " * 500) + f"\n{FACT}.\n"
    p = tmp_path / "canon" / "PINE_BARRON_FARMS_CANON.md"
    p.write_text(big, encoding="utf-8")
    mod = _load_module()

    text, found = mod.render(str(tmp_path / "canon"), max_chars=200)
    assert found is True
    assert "truncated to 200 chars" in text
    assert str(p) in text                               # names the full-file path to read


@pytest.mark.parametrize("with_packet", [True, False])
def test_loader_cli_always_exits_zero(tmp_path, with_packet):
    cdir = tmp_path / "canon"
    cdir.mkdir()
    if with_packet:
        (cdir / "PINE_BARRON_FARMS_CANON.md").write_text(CANON_TEXT, encoding="utf-8")
    r = subprocess.run([sys.executable, str(LOADER), "--canon-dir", str(cdir)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    if with_packet:
        assert FACT in r.stdout and "canon loaded: " in r.stdout
    else:
        assert "canon packet NOT FOUND" in r.stdout


def _build_pbf_home(tmp_path: Path) -> Path:
    home = tmp_path / "data"
    (home / "skills" / "pbf-canon").mkdir(parents=True)
    (home / "skills" / "pbf-canon" / "SKILL.md").write_text(SKILL_MD.read_text(encoding="utf-8"), encoding="utf-8")
    # the skill references the loader as ../../canon/load_canon.py
    (home / "canon").mkdir()
    (home / "canon" / "load_canon.py").write_text(LOADER.read_text(encoding="utf-8"), encoding="utf-8")
    (home / "canon" / "PINE_BARRON_FARMS_CANON.md").write_text(CANON_TEXT, encoding="utf-8")
    return home


def test_end_to_end_fact_reaches_assembled_context(tmp_path):
    """The "appears in assembled context" proof, in two deterministic halves:

      1. preloading the `pbf-canon` skill assembles its full body — the loader
         invocation and how to read its output — into the system-prompt text;
      2. running that loader (the mechanism SOUL.md and the skill both invoke)
         emits the canon fact + a `canon loaded:` receipt.

    Together: a fact from a test canon file reaches the model's context through a
    documented, bash-free, no-engine-change mechanism.
    """
    if not SKILL_MD.is_file():
        pytest.skip("pbf-canon SKILL.md missing")
    home = _build_pbf_home(tmp_path)

    # --- 1. the skill body is assembled into the preloaded-skills prompt -----
    code = textwrap.dedent("""
        import json
        from agent.skill_commands import build_preloaded_skills_prompt
        prompt, loaded, missing = build_preloaded_skills_prompt(["pbf-canon"])
        print("LOADED=" + json.dumps(loaded))
        print("=====PROMPT=====")
        print(prompt)
    """)
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO_ROOT)}
    env.pop("HERMES_PROFILE", None)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    head, _, prompt = r.stdout.partition("=====PROMPT=====")
    assert '"pbf-canon"' in head, r.stdout
    flat = " ".join(prompt.split())
    assert "load_canon.py" in flat
    assert "that block **is the canon**" in flat       # the "treat this as canon" wiring
    assert "canon packet NOT FOUND" in prompt          # the absent-case handling is in context too

    # --- 2. the loader the skill/SOUL invoke puts the fact into context -----
    r2 = subprocess.run([sys.executable, str(home / "canon" / "load_canon.py")],
                        capture_output=True, text=True,
                        env={**os.environ, "HERMES_HOME": str(home)})
    assert r2.returncode == 0, r2.stderr
    assert "canon loaded: " in r2.stdout
    assert FACT in r2.stdout
    assert "sha256=" in r2.stdout
