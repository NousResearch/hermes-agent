"""Vault memory: wikilink graph and path safety, against a real temp vault."""

import pytest

from agent import vault_notes as vn


@pytest.fixture
def vault(tmp_path, monkeypatch):
    root = tmp_path / "vault"
    (root / "04_PROJEKTY").mkdir(parents=True)
    (root / ".obsidian").mkdir()
    (root / "A.md").write_text("# Alpha\nsee [[B]] and [[04_PROJEKTY/Proj|projekt]] #ludzie\n", encoding="utf-8")
    (root / "B.md").write_text("no links\n", encoding="utf-8")
    (root / "04_PROJEKTY" / "Proj.md").write_text("# Proj\n[[A]] [[Nope]]\n", encoding="utf-8")
    (root / ".obsidian" / "secret.md").write_text("hidden", encoding="utf-8")
    monkeypatch.setenv(vn.VAULT_ENV_KEY, str(root))
    return root


def test_graph_links_resolve_and_hidden_dirs_are_skipped(vault):
    graph = vn.build_vault_graph()
    ids = {n["id"] for n in graph["nodes"]}
    assert ids == {"A.md", "B.md", "04_PROJEKTY/Proj.md"}
    edges = {(e["source"], e["target"]) for e in graph["edges"]}
    assert ("A.md", "B.md") in edges and ("04_PROJEKTY/Proj.md", "A.md") in edges
    assert len(edges) == 2  # [[Nope]] dangles and A<->Proj dedupes
    assert next(n for n in graph["nodes"] if n["id"] == "A.md")["links"] == 2


@pytest.mark.parametrize("bad", ["../x.md", "/etc/passwd.md", ".obsidian/secret.md", "A.txt"])
def test_paths_outside_the_note_space_are_refused(vault, bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        vn.read_note(bad)


def test_create_write_delete_roundtrip_uses_trash(vault):
    made = vn.create_note("Nowa: notatka?", "04_PROJEKTY")
    assert made["id"] == "04_PROJEKTY/Nowa notatka.md"
    with pytest.raises(FileExistsError):
        vn.create_note("Nowa notatka", "04_PROJEKTY")
    vn.write_note(made["id"], "# Zmieniona\n")
    assert vn.read_note(made["id"])["label"] == "Zmieniona"
    vn.delete_note(made["id"])
    assert not (vault / made["id"]).exists()
    assert (vault / ".trash" / "Nowa notatka.md").exists()
    assert made["id"] not in {n["id"] for n in vn.build_vault_graph()["nodes"]}
