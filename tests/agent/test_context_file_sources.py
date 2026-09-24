"""Per-file context manifest (``agent/context_file_sources.py``) behind the ``/context`` Rules figure.

The manifest and ``build_context_files_prompt`` share one discovery walk, so the invariant under test is
parity: a file is reported ``loaded`` iff its content appears in the built prompt.
"""

from pathlib import Path

import pytest

from agent.context_file_sources import list_context_file_sources, render_context_file_lines
from agent.prompt_builder import build_context_files_prompt


@pytest.fixture()
def project(tmp_path):
    (tmp_path / ".git").mkdir()
    return tmp_path


def _by_label(sources):
    return {s["label"]: s for s in sources}


def test_manifest_matches_what_the_prompt_actually_loads(project, tmp_path_factory):
    """Every context type present at once; the ladder picks .hermes.md, the chain lists both AGENTS files,
    CLAUDE.md/.cursorrules/.cursor/rules/*.mdc are shadowed, an empty file never wins, SOUL.md rides along."""
    (project / ".hermes.md").write_text("hermes rules")
    (project / "AGENTS.md").write_text("root agents rules")
    sub = project / "pkg"
    sub.mkdir()
    (sub / "AGENTS.override.md").write_text("")  # empty: falls through to AGENTS.md in the same directory
    (sub / "AGENTS.md").write_text("pkg agents rules")
    (sub / "CLAUDE.md").write_text("claude rules")
    (sub / ".cursorrules").write_text("cursor rules")
    (sub / ".cursor" / "rules").mkdir(parents=True)
    (sub / ".cursor" / "rules" / "a.mdc").write_text("mdc rule a")
    home = tmp_path_factory.mktemp("home")
    (home / "SOUL.md").write_text("identity text")

    sources = list_context_file_sources(cwd=str(sub), home_override=home)
    prompt = build_context_files_prompt(cwd=str(sub), home_override=home)

    statuses = {s["label"]: s["status"] for s in sources}
    assert statuses == {
        ".hermes.md": "loaded", "../AGENTS.md": "shadowed", "AGENTS.override.md": "empty", "AGENTS.md": "shadowed",
        "CLAUDE.md": "shadowed", ".cursorrules": "shadowed", ".cursor/rules/a.mdc": "shadowed", "SOUL.md": "loaded",
    }
    for src in sources:
        body = Path(src["path"]).read_text().strip() if src["chars"] else ""
        assert src["loaded"] == (bool(body) and body in prompt), src
    assert all(s["est_tokens"] > 0 for s in sources if s["chars"])

    # Same walk, other winner: drop .hermes.md and the whole AGENTS chain loads while the rest stays shadowed.
    (project / ".hermes.md").unlink()
    statuses = {s["label"]: s["status"] for s in list_context_file_sources(cwd=str(sub), home_override=home)}
    prompt = build_context_files_prompt(cwd=str(sub), home_override=home)
    assert statuses["../AGENTS.md"] == statuses["AGENTS.md"] == "loaded" and "root agents rules" in prompt
    assert statuses["CLAUDE.md"] == "shadowed" and "claude rules" not in prompt


def test_truncated_and_suppressed_statuses_follow_the_builder(project, monkeypatch, tmp_path_factory):
    import agent.prompt_builder as pb

    monkeypatch.setattr(pb, "_get_context_file_max_chars", lambda *_a: 40)
    (project / "AGENTS.md").write_text("x" * 100)
    home = tmp_path_factory.mktemp("home")
    entry = _by_label(list_context_file_sources(cwd=str(project), home_override=home))["AGENTS.md"]
    assert entry["status"] == "truncated" and entry["loaded"] is True
    assert "[...truncated AGENTS.md" in build_context_files_prompt(cwd=str(project), home_override=home)

    # Install-tree guard: a fallback cwd (cwd=None) inside the Hermes tree lists the file but never loads it.
    monkeypatch.setattr("agent.runtime_cwd._is_install_tree", lambda _p: True)
    monkeypatch.chdir(project)
    entry = _by_label(list_context_file_sources(cwd=None, home_override=home))["AGENTS.md"]
    assert entry["status"] == "suppressed" and entry["loaded"] is False
    assert build_context_files_prompt(cwd=None, skip_soul=True) == ""
    assert _by_label(list_context_file_sources(cwd=None, allow_install_tree_fallback=True, home_override=home))[
        "AGENTS.md"]["status"] == "truncated"

    lines = render_context_file_lines(list_context_file_sources(cwd=None, home_override=home))
    assert lines[0] == "Context files" and "AGENTS.md" in lines[1] and "install tree" in lines[1]
    assert render_context_file_lines([]) == []

    # Injection scan: the builder swaps the body for a BLOCKED marker, so the manifest must not say "loaded".
    monkeypatch.setattr(pb, "_get_context_file_max_chars", lambda *_a: 10_000)
    monkeypatch.setattr(pb, "_scan_for_threats", lambda content, scope: ["fake-pattern"] if "evil" in content else [])
    (project / "AGENTS.md").write_text("evil")
    entry = _by_label(list_context_file_sources(cwd=str(project), home_override=home))["AGENTS.md"]
    assert entry["status"] == "blocked" and entry["loaded"] is False
    assert "[BLOCKED: AGENTS.md" in build_context_files_prompt(cwd=str(project), home_override=home)

    # The user's own SOUL.md is flagged but loaded — the manifest must say so and the prompt must carry it.
    (home / "SOUL.md").write_text("evil identity text")
    entries = _by_label(list_context_file_sources(cwd=str(project), home_override=home))
    assert entries["SOUL.md"]["status"] == "flagged" and entries["SOUL.md"]["loaded"] is True
    assert entries["AGENTS.md"]["status"] == "blocked"
    prompt = build_context_files_prompt(cwd=str(project), home_override=home)
    assert "evil identity text" in prompt and "[BLOCKED: SOUL.md" not in prompt and "[BLOCKED: AGENTS.md" in prompt
    assert any("SOUL.md" in line and "review the file" in line
               for line in render_context_file_lines(list(entries.values())))


def test_an_identical_copy_further_down_the_chain_is_not_reported_loaded(project, tmp_path_factory):
    """``_load_agents_md`` drops content it has already emitted (a copied or symlinked AGENTS.md),
    so the duplicate is discovered and read but never reaches the prompt. Reporting it ``loaded``
    overstated both the file count and the token budget shown by ``/context``."""
    (project / "AGENTS.md").write_text("shared team rules")
    sub = project / "pkg"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("shared team rules")  # byte-identical copy
    home = tmp_path_factory.mktemp("home")

    sources = list_context_file_sources(cwd=str(sub), home_override=home, skip_soul=True)
    prompt = build_context_files_prompt(cwd=str(sub), home_override=home, skip_soul=True)

    statuses = {s["label"]: s["status"] for s in sources}
    assert list(statuses.values()) == ["loaded", "duplicate"]
    assert [s["loaded"] for s in sources] == [True, False]
    # The prompt carries the content once; the manifest now agrees on how many sections it built.
    assert prompt.count("shared team rules") == 1
    assert sum(1 for s in sources if s["loaded"]) == prompt.count("## ")


def test_a_distinct_file_further_down_the_chain_still_loads(project, tmp_path_factory):
    """The guard is content identity, not position: a real per-package AGENTS.md is unaffected."""
    (project / "AGENTS.md").write_text("root rules")
    sub = project / "pkg"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("package rules")
    home = tmp_path_factory.mktemp("home")

    sources = list_context_file_sources(cwd=str(sub), home_override=home, skip_soul=True)
    prompt = build_context_files_prompt(cwd=str(sub), home_override=home, skip_soul=True)
    assert all(s["status"] == "loaded" for s in sources)
    for src in sources:
        assert Path(src["path"]).read_text() in prompt


def test_the_duplicate_guard_only_applies_within_the_winning_chain(project, tmp_path_factory):
    """A whole chain beaten by a higher-priority type is ``shadowed`` — the accurate reason —
    not ``duplicate``, even when two of its files are identical."""
    (project / ".hermes.md").write_text("hermes wins")
    (project / "AGENTS.md").write_text("same text")
    sub = project / "pkg"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("same text")
    home = tmp_path_factory.mktemp("home")

    statuses = [s["status"] for s in list_context_file_sources(
        cwd=str(sub), home_override=home, skip_soul=True)]
    assert statuses == ["loaded", "shadowed", "shadowed"]


def test_identical_cursor_rules_are_not_deduplicated(project, tmp_path_factory):
    """``_load_cursorrules`` concatenates every candidate without deduping, so the manifest
    must not borrow the AGENTS.md rule — both copies really do reach the prompt."""
    (project / ".cursor" / "rules").mkdir(parents=True)
    (project / ".cursor" / "rules" / "a.mdc").write_text("duplicated rule")
    (project / ".cursor" / "rules" / "b.mdc").write_text("duplicated rule")
    home = tmp_path_factory.mktemp("home")

    sources = list_context_file_sources(cwd=str(project), home_override=home, skip_soul=True)
    prompt = build_context_files_prompt(cwd=str(project), home_override=home, skip_soul=True)
    assert [s["status"] for s in sources] == ["loaded", "loaded"]
    assert prompt.count("duplicated rule") == 2


def test_a_duplicate_deeper_in_a_three_directory_chain_is_caught(project, tmp_path_factory):
    """Dedupe is against everything already emitted, not just the immediately preceding file."""
    (project / "AGENTS.md").write_text("X rules")
    mid = project / "pkg"
    mid.mkdir()
    (mid / "AGENTS.md").write_text("middle rules")
    deep = mid / "inner"
    deep.mkdir()
    (deep / "AGENTS.md").write_text("X rules")  # identical to the git-root file
    home = tmp_path_factory.mktemp("home")

    sources = list_context_file_sources(cwd=str(deep), home_override=home, skip_soul=True)
    prompt = build_context_files_prompt(cwd=str(deep), home_override=home, skip_soul=True)
    assert [s["status"] for s in sources] == ["loaded", "loaded", "duplicate"]
    assert prompt.count("X rules") == 1
    assert sum(1 for s in sources if s["loaded"]) == prompt.count("## ")


def test_the_duplicate_status_renders_with_a_reason(project, tmp_path_factory):
    """``/context`` shows why the file is not counted rather than an unexplained bullet."""
    (project / "AGENTS.md").write_text("shared rules")
    sub = project / "pkg"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("shared rules")
    home = tmp_path_factory.mktemp("home")

    lines = render_context_file_lines(
        list_context_file_sources(cwd=str(sub), home_override=home, skip_soul=True))
    assert any("identical to an earlier file in the chain" in line for line in lines)
    assert not any("•" in line for line in lines)  # no unmapped-status fallback glyph
