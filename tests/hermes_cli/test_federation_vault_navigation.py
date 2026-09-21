import json
import re

import pytest

from scripts.federation_vault_navigation import render_notes, write_notes


def test_navigation_links_resolve_and_unchanged_registry_does_not_rewrite(tmp_path):
    roles = [{"id": name, "display_name": name.title(), "description": "Bounded role",
              "authority": "advisory", "schedule": "daily", "handoffs": [other]}
             for name, other in [("librarian", "reviewer"), ("reviewer", "librarian")]]
    payload = json.dumps({"departments": [{"display_name": "Knowledge", "description": "Catalogue",
                                           "roles": roles}]}).encode()
    notes = render_notes(payload)
    assert write_notes(tmp_path, notes) == len(roles) + 1
    for relative, content in notes.items():
        for link in re.findall(r"\]\(([^)]+)\)", content):
            assert (tmp_path / relative).parent.joinpath(link).is_file()
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*.md")}
    assert write_notes(tmp_path, notes) == 0
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*.md")}


def test_navigation_preserves_authored_notes_and_rejects_symlink_destinations(tmp_path):
    target = tmp_path / "Index.md"
    target.write_text("Human-authored context")
    with pytest.raises(ValueError, match="authored note"):
        write_notes(tmp_path, {"Index.md": "generated"})
    assert target.read_text() == "Human-authored context"
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        write_notes(alias, {"Other.md": "generated"})
