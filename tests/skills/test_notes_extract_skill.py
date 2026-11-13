"""Tests for the notes-extract skill (scan + idempotent managed-region upsert).

Hermetic: tmp vault (path contains a space), HERMES_HOME monkeypatched to tmp,
an injected fixed clock, and no network. Mirrors the stdlib+pytest convention
used by other skill tests.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/note-taking/notes-extract/scripts"
)
SKILL_MD = SCRIPTS.parent / "SKILL.md"

FIXED1 = lambda: datetime(2026, 1, 2, tzinfo=timezone.utc)  # noqa: E731
FIXED2 = lambda: datetime(2026, 3, 4, tzinfo=timezone.utc)  # noqa: E731


@pytest.fixture
def mods(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.delenv("NOTES_EXTRACT_SOURCES", raising=False)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import _state
    import notes_scan
    import upsert_entry
    return _state, notes_scan, upsert_entry


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "my vault"  # space in path on purpose
    v.mkdir()
    return v


def person_entry(name, section, claim, text, aliases=None, op="assert"):
    return {
        "entity": {"kind": "person", "name": name, "aliases": aliases or []},
        "section": section, "claim": claim, "text": text, "op": op,
    }


def run(mods, vault, entries, source="note.md", link="note", sha="s1", clock=FIXED1):
    _state, _, upsert = mods
    state = _state.load_state(vault)
    rep = upsert.upsert_source(vault, str(vault / source), link, sha, entries, state, clock)
    _state.save_state(vault, state)
    return rep


def bullets(mods, path, region):
    _, _, upsert = mods
    return upsert.read_region_bullets(path.read_text(encoding="utf-8"), region)


# --------------------------------------------------------------------------- #
# upsert_entry
# --------------------------------------------------------------------------- #
def test_create_new_person_file(mods, vault):
    run(mods, vault, [person_entry("Jane Doe", "facts",
        {"subject": "jane", "predicate": "employer", "object": "acme"}, "VP Eng at Acme.")])
    f = vault / "People" / "jane-doe.md"
    assert f.exists()
    text = f.read_text(encoding="utf-8")
    assert "type: person" in text
    assert "updated: 2026-01-02" in text
    b = bullets(mods, f, "facts")
    assert len(b) == 1
    assert b[0][0].startswith("nx-")
    assert "VP Eng at Acme." in b[0][1]
    assert "[[note]]" in b[0][1]


def test_reworded_same_claim_keeps_one_bullet(mods, vault):
    claim = {"subject": "jane", "predicate": "employer", "object": "acme"}
    run(mods, vault, [person_entry("Jane Doe", "facts", claim, "VP Eng at Acme.")])
    run(mods, vault, [person_entry("Jane Doe", "facts", claim, "Jane is VP of Engineering at Acme.")])
    b = bullets(mods, vault / "People" / "jane-doe.md", "facts")
    assert len(b) == 1
    assert "VP of Engineering" in b[0][1]


def test_two_sources_same_claim_two_bullets(mods, vault):
    claim = {"subject": "jane", "predicate": "employer", "object": "acme"}
    run(mods, vault, [person_entry("Jane Doe", "facts", claim, "VP Eng.")], source="a.md", link="a")
    run(mods, vault, [person_entry("Jane Doe", "facts", claim, "VP Eng.")], source="b.md", link="b")
    b = bullets(mods, vault / "People" / "jane-doe.md", "facts")
    assert len(b) == 2  # distinct provenance preserved


def test_section_routing_person_and_project(mods, vault):
    run(mods, vault, [
        person_entry("Jane Doe", "facts", {"subject": "j", "predicate": "p", "object": "o"}, "f"),
        person_entry("Jane Doe", "topics", {"subject": "j", "predicate": "likes", "object": "tea"}, "tea"),
        {"entity": {"kind": "project", "name": "Flow Viz", "aliases": []},
         "section": "ideas", "claim": {"subject": "fv", "predicate": "idea", "object": "x"},
         "text": "Add a heatmap.", "op": "assert"},
    ])
    pf = vault / "People" / "jane-doe.md"
    assert len(bullets(mods, pf, "facts")) == 1
    assert len(bullets(mods, pf, "topics")) == 1
    prj = vault / "Ideas-Projects" / "flow-viz.md"
    assert prj.exists()
    assert "type: project" in prj.read_text(encoding="utf-8")
    assert len(bullets(mods, prj, "ideas")) == 1


def test_idempotent_rerun_no_diff(mods, vault):
    entries = [person_entry("Jane Doe", "facts",
               {"subject": "j", "predicate": "e", "object": "a"}, "Fact.")]
    run(mods, vault, entries)
    f = vault / "People" / "jane-doe.md"
    before = f.read_text(encoding="utf-8")
    rep = run(mods, vault, entries)  # identical, same clock
    after = f.read_text(encoding="utf-8")
    assert before == after
    assert all(e["action"] == "unchanged" for e in rep["entries"])


def test_hand_edit_outside_fences_preserved(mods, vault):
    run(mods, vault, [person_entry("Jane Doe", "facts",
        {"subject": "j", "predicate": "e", "object": "a"}, "Fact.")], source="n1.md")
    f = vault / "People" / "jane-doe.md"
    marker = "\n## My private notes\nHand-written. Do not touch.\n"
    f.write_text(f.read_text(encoding="utf-8") + marker, encoding="utf-8")
    # A different note adds another fact → region rewrite in the same section.
    run(mods, vault, [person_entry("Jane Doe", "facts",
        {"subject": "j", "predicate": "city", "object": "nyc"}, "Lives in NYC.")], source="n2.md")
    text = f.read_text(encoding="utf-8")
    assert marker in text  # byte-for-byte preserved
    assert len(bullets(mods, f, "facts")) == 2


def test_retract_moves_to_archive(mods, vault):
    claim = {"subject": "j", "predicate": "employer", "object": "acme"}
    run(mods, vault, [person_entry("Jane Doe", "facts", claim, "At Acme.")])
    run(mods, vault, [person_entry("Jane Doe", "facts", claim, "At Acme.", op="retract")])
    f = vault / "People" / "jane-doe.md"
    assert len(bullets(mods, f, "facts")) == 0
    assert len(bullets(mods, f, "facts-archive")) == 1


def test_updated_bumps_only_on_change(mods, vault):
    e1 = [person_entry("Jane Doe", "facts", {"subject": "j", "predicate": "e", "object": "a"}, "F1.")]
    run(mods, vault, e1, clock=FIXED1)
    f = vault / "People" / "jane-doe.md"
    run(mods, vault, e1, clock=FIXED2)  # unchanged content
    assert "updated: 2026-01-02" in f.read_text(encoding="utf-8")
    run(mods, vault, [person_entry("Jane Doe", "facts",
        {"subject": "j", "predicate": "city", "object": "nyc"}, "New.")], clock=FIXED2)
    assert "updated: 2026-03-04" in f.read_text(encoding="utf-8")


def test_nfc_slug_collision_suffixes_and_flags(mods, vault):
    rep = run(mods, vault, [
        person_entry("Cafe", "facts", {"subject": "c", "predicate": "p", "object": "o1"}, "one"),
        person_entry("Café", "facts", {"subject": "c", "predicate": "p", "object": "o2"}, "two"),
    ])
    assert (vault / "People" / "cafe.md").exists()
    assert (vault / "People" / "cafe-2.md").exists()
    assert any("cafe-2.md" in p for p in rep["needs_confirm"])


def test_per_source_reconciliation_removes_stale(mods, vault):
    a = person_entry("Jane Doe", "facts", {"subject": "j", "predicate": "e", "object": "acme"}, "A")
    b = person_entry("Jane Doe", "facts", {"subject": "j", "predicate": "city", "object": "nyc"}, "B")
    run(mods, vault, [a, b], source="n.md", sha="sha-v1")
    f = vault / "People" / "jane-doe.md"
    assert len(bullets(mods, f, "facts")) == 2
    run(mods, vault, [a], source="n.md", sha="sha-v2")  # same source, B dropped
    remaining = bullets(mods, f, "facts")
    assert len(remaining) == 1
    assert "A" in remaining[0][1] and "B" not in remaining[0][1]


def test_alias_routes_to_same_entity(mods, vault):
    run(mods, vault, [person_entry("Jane Doe", "facts",
        {"subject": "j", "predicate": "e", "object": "a"}, "F1", aliases=["Jane"])], source="n1.md")
    run(mods, vault, [person_entry("Jane", "facts",
        {"subject": "j", "predicate": "city", "object": "nyc"}, "F2")], source="n2.md")
    assert (vault / "People" / "jane-doe.md").exists()
    assert not (vault / "People" / "jane.md").exists()
    assert len(bullets(mods, vault / "People" / "jane-doe.md", "facts")) == 2


# --------------------------------------------------------------------------- #
# notes_scan
# --------------------------------------------------------------------------- #
def _write(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_scan_new_changed_unchanged(mods, vault):
    _state, notes_scan, _ = mods
    note = vault / "a note.md"
    _write(note, "# A\nhello")

    res = notes_scan.scan(vault, [], _state.load_state(vault), include_unchanged=False)
    assert len(res) == 1 and res[0]["status"] == "new"
    assert res[0]["link"] == "a note"

    # Record it (empty extraction still records the source sha).
    run(mods, vault, [], source="a note.md", link="a note", sha=res[0]["sha"])
    assert notes_scan.scan(vault, [], _state.load_state(vault), include_unchanged=False) == []

    _write(note, "# A\nhello world")  # change content
    res2 = notes_scan.scan(vault, [], _state.load_state(vault), include_unchanged=False)
    assert len(res2) == 1 and res2[0]["status"] == "changed"


def test_scan_skips_generated_and_dotdirs(mods, vault):
    _state, notes_scan, _ = mods
    _state.save_state(vault, _state.empty_state(vault))
    _write(vault / "real.md", "x")
    _write(vault / "People" / "jane.md", "generated")
    _write(vault / "Ideas-Projects" / "p.md", "generated")
    _write(vault / ".obsidian" / "config.md", "hidden")
    res = notes_scan.scan(vault, [], _state.load_state(vault), include_unchanged=True)
    links = {r["link"] for r in res}
    assert links == {"real"}


def test_scan_extra_sources(mods, vault, tmp_path):
    _state, notes_scan, _ = mods
    extra = tmp_path / "extra notes"
    _write(extra / "ext.txt", "external note")
    res = notes_scan.scan(vault, [extra], _state.load_state(vault), include_unchanged=True)
    assert any(r["link"] == "ext" for r in res)


# --------------------------------------------------------------------------- #
# helpers + frontmatter lint
# --------------------------------------------------------------------------- #
def test_split_frontmatter(mods):
    _, _, upsert = mods
    fm, body = upsert.split_frontmatter("---\ntype: person\nid: x\n---\n## Facts\n")
    assert fm["type"] == "person" and fm["id"] == "x"
    assert body == "## Facts\n"
    fm2, body2 = upsert.split_frontmatter("no frontmatter")
    assert fm2 == {} and body2 == "no frontmatter"


def test_slugify_unicode(mods):
    _state, _, _ = mods
    assert _state.slugify("José O'Brien") == "jose-o-brien"
    assert _state.slugify("Anne-Marie") == "anne-marie"
    assert _state.slugify("张伟").startswith("n-")  # pure CJK → hash fallback


@pytest.mark.parametrize("kind,section,folder", [
    ("person", "facts", "People"), ("project", "ideas", "Ideas-Projects"),
])
def test_unowned_frontmatter_preserved_on_region_change(mods, vault, kind, section, folder):
    entry = person_entry("Atlas", section, {}, "Old")
    entry["entity"]["kind"] = kind
    run(mods, vault, [entry])
    path = vault / folder / "atlas.md"
    unowned = ('# Human metadata: keep spacing and comments\n'
               'tags: [private, "keep: me"] # custom\n'
               'details:\n  priority: high\n  nested:\n    owner: human\n'
               'summary: |\n  First line.\n  updated: not a top-level key\n'
               '"custom:key": "quoted value"\n\n')
    text = path.read_text(encoding="utf-8").replace("id:", unowned + "id:", 1)
    text = text.replace("\nupdated:", "\n# Before updated\nupdated:", 1)
    path.write_text(text, encoding="utf-8")
    entry["text"] = "Changed"
    run(mods, vault, [entry], clock=FIXED2)
    after = path.read_text(encoding="utf-8")
    assert unowned in after
    assert "# Before updated\nupdated: 2026-03-04" in after
    assert "Changed" in after
    assert len(bullets(mods, path, section)) == 1
    run(mods, vault, [entry], clock=FIXED2)
    assert path.read_text(encoding="utf-8") == after


@pytest.mark.parametrize("loss", ["missing", "corrupt", "version", "not-object", "incomplete"])
def test_cache_loss_refuses_existing_outputs_without_modification(mods, vault, loss):
    import json
    state_mod, _, _ = mods
    run(mods, vault, [person_entry("Jane Doe", "facts", {}, "Original", aliases=["Jane"])])
    path = vault / "People" / "jane-doe.md"
    before = path.read_bytes()
    cache = state_mod.state_path(vault)
    if loss == "missing":
        cache.unlink()
    else:
        cache.write_text({"corrupt": "{bad", "version": json.dumps({"version": -1}),
                          "not-object": "[]", "incomplete": '{"version": 1}'}[loss], encoding="utf-8")
    with pytest.raises(ValueError, match="Restore.*state.*backup"):
        state_mod.load_state(vault)
    assert path.read_bytes() == before


def test_cache_loss_empty_vault_rescans_sources(mods, vault):
    state_mod, scan, _ = mods
    _write(vault / "source.md", "Source")
    run(mods, vault, [], source="source.md", sha="old-sha")
    state_mod.state_path(vault).unlink()
    state = state_mod.load_state(vault)
    assert state["sources"] == state["entities"] == state["alias_index"] == {}
    assert scan.scan(vault, [], state, include_unchanged=False)[0]["status"] == "new"


def test_cache_loss_cli_refuses_scan_and_upsert(mods, vault):
    import os
    import subprocess
    import sys
    state_mod, _, _ = mods
    run(mods, vault, [person_entry("Jane Doe", "facts", {}, "Original")])
    path = vault / "People" / "jane-doe.md"
    before = path.read_bytes()
    state_mod.state_path(vault).unlink()
    for script, args in [("notes_scan.py", []), ("upsert_entry.py", [
            "--source-path", str(vault / "note.md"), "--entries-json", "[]"])]:
        proc = subprocess.run([sys.executable, str(SCRIPTS / script), "--vault", str(vault), *args],
                              capture_output=True, text=True, env=os.environ.copy(), timeout=10)
        assert proc.returncode != 0
        assert "Restore the state file from backup" in proc.stderr
        assert path.read_bytes() == before
        assert not state_mod.state_path(vault).exists()


def test_project_status_and_owned_multiline_aliases(mods, vault):
    entry = {"entity": {"kind": "project", "name": "Atlas", "aliases": ["Shared"]},
             "section": "ideas", "claim": {}, "text": "Old"}
    run(mods, vault, [entry])
    path = vault / "Ideas-Projects" / "atlas.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("status: active", "status: paused # human choice")
    text = text.replace("aliases: [Shared]", "aliases:\n  - Shared\n# Keep comment")
    path.write_text(text, encoding="utf-8")
    entry["text"] = "Changed"
    run(mods, vault, [entry])
    after = path.read_text(encoding="utf-8")
    assert "status: paused # human choice" in after
    assert "aliases: [Shared]\n# Keep comment\n" in after
    assert "  - Shared" not in after


@pytest.mark.parametrize("first_kind", ["person", "project"])
def test_same_name_and_alias_are_scoped_by_kind(mods, vault, first_kind):
    state_mod, _, _ = mods
    kinds = [first_kind, "project" if first_kind == "person" else "person"]
    entries = []
    for kind in kinds:
        entry = person_entry("Atlas", "facts" if kind == "person" else "ideas",
                             {}, kind, aliases=["Shared"])
        entry["entity"]["kind"] = kind
        entries.append(entry)
    run(mods, vault, entries)
    for entry in entries:
        entry["entity"]["name"] = "Shared"
        entry["text"] += " via alias"
    run(mods, vault, entries)
    state = state_mod.load_state(vault)
    assert len(state["entities"]) == 2
    for kind, folder, section in [("person", "People", "facts"),
                                  ("project", "Ideas-Projects", "ideas")]:
        eid = state_mod.entity_id(kind, "Atlas")
        path = vault / folder / "atlas.md"
        assert f"id: {eid}" in path.read_text(encoding="utf-8")
        assert state["alias_index"][kind]["shared"] == eid
        assert len(bullets(mods, path, section)) == 1
        assert f"{kind} via alias" in bullets(mods, path, section)[0][1]
        assert not (vault / folder / "shared.md").exists()


def test_legacy_alias_index_migrates_without_losing_sources(mods, vault):
    state_mod, _, _ = mods
    run(mods, vault, [person_entry("Jane Doe", "facts", {}, "Fact", aliases=["Jane"])])
    state = state_mod.load_state(vault)
    sources = state["sources"].copy()
    eid = state_mod.entity_id("person", "Jane Doe")
    state["alias_index"] = {"jane doe": eid, "jane": eid}
    state_mod.save_state(vault, state)
    migrated = state_mod.load_state(vault)
    assert migrated["sources"] == sources
    assert migrated["alias_index"]["person"]["jane"] == eid
    run(mods, vault, [person_entry("Jane", "facts", {}, "Via alias")], source="other.md")
    assert not (vault / "People" / "jane.md").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_resolved_name_history_survives_reload(mods, vault, legacy):
    state_mod, _, _ = mods
    run(mods, vault, [person_entry("Jane Doe", "facts", {}, "Original", aliases=["Jane"])])
    run(mods, vault, [person_entry("J. Doe", "facts", {}, "Variant", aliases=["Jane"])])
    if legacy:
        state = state_mod.load_state(vault)
        state["alias_index"] = state["alias_index"]["person"]
        state_mod.save_state(vault, state)
    run(mods, vault, [person_entry("J. Doe", "facts", {}, "Reloaded")])
    assert not (vault / "People" / "j-doe.md").exists()
    assert len(state_mod.load_state(vault)["entities"]) == 1
    assert "Reloaded" in bullets(mods, vault / "People" / "jane-doe.md", "facts")[0][1]


def test_description_under_60_chars():
    text = SKILL_MD.read_text(encoding="utf-8")
    m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    assert m, "description missing"
    assert len(m.group(1).strip()) <= 60, len(m.group(1).strip())
