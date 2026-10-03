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


# ── provenance labels are spelled the same on every platform ────────────────


def test_chain_labels_use_forward_slashes_on_every_platform(project, tmp_path_factory):
    """The label is a provenance heading in the system prompt ("## ../AGENTS.md"), not a path to
    open. Building it with ``os.path.relpath`` alone made the prompt — and the ``/context``
    listing — read "..\\AGENTS.md" on Windows, while the sibling ``.cursor/rules`` labels next to
    it are spelled with forward slashes, so one manifest carried both conventions."""
    (project / "AGENTS.md").write_text("root rules")
    mid = project / "pkg"
    mid.mkdir()
    (mid / "AGENTS.md").write_text("pkg rules")
    deep = mid / "inner"
    deep.mkdir()
    (deep / "AGENTS.md").write_text("deep rules")
    home = tmp_path_factory.mktemp("home")

    sources = list_context_file_sources(cwd=str(deep), home_override=home, skip_soul=True)
    labels = [s["label"] for s in sources]
    assert labels == ["../../AGENTS.md", "../AGENTS.md", "AGENTS.md"]
    assert not any("\\" in label for label in labels)

    # The same spelling is what the model actually sees.
    prompt = build_context_files_prompt(cwd=str(deep), home_override=home, skip_soul=True)
    headings = [line for line in prompt.splitlines() if line.startswith("## ")]
    assert headings == ["## ../../AGENTS.md", "## ../AGENTS.md", "## AGENTS.md"]


def test_every_discovered_label_is_slash_spelled(project, tmp_path_factory):
    """One convention across kinds: the `.cursor/rules/*.mdc` labels were already slash-spelled."""
    import agent.prompt_builder as pb

    (project / "AGENTS.md").write_text("root rules")
    sub = project / "pkg"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("pkg rules")
    (sub / ".cursor" / "rules").mkdir(parents=True)
    (sub / ".cursor" / "rules" / "a.mdc").write_text("mdc rule")

    labels = [label for _kind, label, _path, _content in pb.discover_context_files(sub.resolve())]
    assert "../AGENTS.md" in labels and ".cursor/rules/a.mdc" in labels
    assert not any("\\" in label for label in labels)
