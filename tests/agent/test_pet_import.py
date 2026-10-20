"""Import contracts for the pet store (agent/pet/store.py).

``export_pet`` shipped without a counterpart, so the round trip is the invariant under
test: a pet you export must come back identical, and nothing an archive carries may land
outside the pets directory. No network — the sheet is synthetic.
"""

from __future__ import annotations

import io
import shutil
import zipfile

import pytest
from PIL import Image

from agent.pet import store
from agent.pet.constants import FRAME_H, FRAME_W


def _sheet_bytes(cols: int = 8, rows: int = 9) -> bytes:
    """A decodable spritesheet (the store checks geometry, never pixels)."""
    buf = io.BytesIO()
    Image.new("RGBA", (FRAME_W * cols, FRAME_H * rows), (30, 60, 120, 255)).save(buf, format="WEBP")
    return buf.getvalue()


def _installed(slug: str) -> store.InstalledPet:
    """The installed pet, asserted present — keeps Optionals out of the contract asserts."""
    pet = store.load_pet(slug)
    assert pet is not None and pet.exists, f"pet '{slug}' is not installed"
    return pet


@pytest.fixture
def pet_home(tmp_path, monkeypatch):
    """Temp HERMES_HOME holding one synthetic installed pet, so export/import run for real."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    pet_dir = store.pets_dir() / "boba"
    pet_dir.mkdir(parents=True, exist_ok=True)
    (pet_dir / "spritesheet.webp").write_bytes(_sheet_bytes())
    (pet_dir / "pet.json").write_text(
        '{"id":"boba","displayName":"Boba","description":"d","spritesheetPath":"spritesheet.webp",'
        '"createdBy":"generator"}'
    )
    return home


def test_export_then_import_restores_the_pet(pet_home, tmp_path):
    """The exported zip is a complete backup: same fields, same sheet bytes, adoptable again."""
    original = _installed("boba")
    filename, blob = store.export_pet("boba")
    archive = tmp_path / filename
    archive.write_bytes(blob)

    shutil.rmtree(store.pets_dir() / "boba")
    assert store.load_pet("boba") is None

    imported = store.import_pet(archive)

    assert (imported.slug, imported.display_name, imported.description) == ("boba", "Boba", "d")
    assert imported.created_by == "generator"  # the manifest survives, so the tag does too
    assert imported.spritesheet.read_bytes() == original.spritesheet.read_bytes()
    assert store.resolve_active_pet("boba").slug == "boba"


def test_import_copies_a_pet_folder_verbatim(pet_home, tmp_path):
    """A folder with the two files is enough — no manifest entry, no gallery round trip."""
    handout = tmp_path / "handout"
    handout.mkdir()
    (handout / "pet.json").write_text('{"id":"Nobu","displayName":"Nobu","description":"x"}')
    (handout / "sprite.webp").write_bytes(_sheet_bytes(cols=2, rows=1))

    pet = store.import_pet(handout)

    assert (pet.slug, pet.display_name) == ("nobu", "Nobu")  # slug normalized, id honored
    assert pet.spritesheet.name == "sprite.webp"  # a non-conventional name still resolves
    assert _installed("nobu").exists


def test_import_rejects_a_source_without_a_sheet(pet_home, tmp_path):
    """The likeliest user error is zipping the wrong folder — that must fail loudly."""
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (wrong / "pet.json").write_text('{"id":"boba","displayName":"Boba"}')

    with pytest.raises(store.PetStoreError, match="spritesheet"):
        store.import_pet(wrong)
    assert _installed("boba").display_name == "Boba"  # the installed pet is untouched


def test_import_refuses_escaping_archive_members(pet_home, tmp_path):
    """An archive member that climbs out of the pet dir aborts the import, writing nothing."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("pet.json", '{"id":"trap","displayName":"Trap","spritesheetPath":"spritesheet.webp"}')
        zipped.writestr("trap/spritesheet.webp", _sheet_bytes())
        zipped.writestr("../../escape.webp", b"pwned")

    with pytest.raises(store.PetStoreError, match="unsafe path"):
        store.import_pet(archive)

    assert not (tmp_path / "escape.webp").exists()
    assert not (tmp_path.parent / "escape.webp").exists()
    assert store.load_pet("trap") is None  # fail-closed: no half-imported pet left behind


def test_import_replaces_a_same_slug_pet_only_with_force(pet_home, tmp_path):
    """Re-importing must never silently clobber an installed pet; force is the explicit overwrite."""
    _, blob = store.export_pet("boba")
    archive = tmp_path / "boba.zip"
    archive.write_bytes(blob)

    with pytest.raises(store.PetStoreError, match="already installed"):
        store.import_pet(archive)

    replaced = store.import_pet(archive, display_name="Boba II", force=True)

    assert replaced.display_name == "Boba II"
    assert _installed("boba").display_name == "Boba II"


def test_import_drops_an_archive_member_with_no_path_segments(pet_home, tmp_path):
    """A member named ``.`` carries no segments: junk to skip, never a crash in the guard."""
    sheet = _sheet_bytes(cols=2, rows=1)
    archive = tmp_path / "dotted.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("pet.json", '{"id":"dotted","displayName":"Dotted","spritesheetPath":"spritesheet.webp"}')
        zipped.writestr("spritesheet.webp", sheet)
        zipped.writestr(".", "")
        zipped.writestr("./.", "")

    imported = store.import_pet(archive)

    assert imported.spritesheet.read_bytes() == sheet
    assert _installed("dotted").exists


def test_import_refuses_a_symlinked_folder_member(pet_home, tmp_path):
    """A symlink in the folder must not be read through — the zip path refuses those entries too."""
    outside = tmp_path / "outside.webp"
    outside.write_bytes(_sheet_bytes(cols=2, rows=1))
    handout = tmp_path / "handout"
    handout.mkdir()
    (handout / "pet.json").write_text('{"id":"linked","displayName":"Linked","spritesheetPath":"spritesheet.webp"}')
    (handout / "spritesheet.webp").symlink_to(outside)

    with pytest.raises(store.PetStoreError, match="symlink"):
        store.import_pet(handout)

    assert not (store.pets_dir() / "linked").exists()  # fail-closed: nothing copied out of the folder
