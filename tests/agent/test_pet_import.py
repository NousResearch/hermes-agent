"""Safe pet package import behavior."""

from __future__ import annotations

import io
import json
import struct
import zipfile

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from agent.pet import store
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def isolated_home(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _atlas_bytes(size=(1536, 1872), *, idle=True, image_format="PNG") -> bytes:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    if idle:
        ImageDraw.Draw(image).ellipse((32, 24, 160, 190), fill=(80, 120, 240, 255))
    output = io.BytesIO()
    image.save(output, format=image_format, lossless=True)
    return output.getvalue()


def _package(
    *,
    metadata=None,
    sheet=None,
    metadata_path="fox/pet.json",
    sheet_path="fox/spritesheet.png",
    extras=(),
) -> bytes:
    metadata = metadata or {"displayName": "Archive Fox", "spritesheetPath": "spritesheet.png"}
    sheet = sheet or _atlas_bytes()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr(metadata_path, json.dumps(metadata))
        zipped.writestr(sheet_path, sheet)
        for path, value in extras:
            zipped.writestr(path, value)
    return archive.getvalue()


def _set_encrypted_flag(data: bytes) -> bytes:
    patched = bytearray(data)
    local = patched.index(b"PK\x03\x04")
    central = patched.index(b"PK\x01\x02")
    struct.pack_into("<H", patched, local + 6, struct.unpack_from("<H", patched, local + 6)[0] | 1)
    struct.pack_into("<H", patched, central + 8, struct.unpack_from("<H", patched, central + 8)[0] | 1)
    return bytes(patched)


def test_import_raw_current_atlas_is_managed_local_pet():
    pet = store.import_pet_package(_atlas_bytes(), filename="Blue Fox.png", name="Blue Fox")

    assert pet.slug == "blue-fox"
    assert pet.created_by == "import"
    assert pet.generated is False
    assert pet.managed_local is True
    assert pet.spritesheet.is_file()


def test_import_accepts_raw_webp_and_legacy_geometry():
    pet = store.import_pet_package(
        _atlas_bytes((1728, 1664), image_format="WEBP"),
        filename="legacy.webp",
    )

    assert pet.display_name == "legacy"
    assert pet.exists


def test_import_exported_package_round_trip():
    pet = store.import_pet_package(_package(), filename="fox.zip")

    assert pet.display_name == "Archive Fox"
    assert pet.exists

    filename, exported = store.export_pet(pet.slug)
    imported_again = store.import_pet_package(exported, filename=filename)
    assert imported_again.slug == "archive-fox-2"
    assert imported_again.created_by == "import"


@pytest.mark.parametrize(
    ("metadata_path", "sheet_path", "metadata", "message"),
    [
        ("../pet.json", "fox/spritesheet.png", {}, "unsafe path"),
        ("fox/pet.json", "fox/../spritesheet.png", {}, "unsafe path"),
        (
            "fox/pet.json",
            "fox/spritesheet.png",
            {"spritesheetPath": "../spritesheet.png"},
            "unsafe spritesheet path",
        ),
    ],
)
def test_import_rejects_unsafe_paths(metadata_path, sheet_path, metadata, message):
    with pytest.raises(store.PetStoreError, match=message):
        store.import_pet_package(
            _package(metadata=metadata, metadata_path=metadata_path, sheet_path=sheet_path),
            filename="bad.zip",
        )


def test_import_rejects_duplicate_casefolded_paths():
    package = _package(extras=(("FOX/PET.JSON", "{}"),))

    with pytest.raises(store.PetStoreError, match="duplicate paths"):
        store.import_pet_package(package, filename="bad.zip")


def test_import_rejects_symlink_and_encrypted_entries():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        link = zipfile.ZipInfo("fox/spritesheet.png")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        zipped.writestr("fox/pet.json", json.dumps({"spritesheetPath": "spritesheet.png"}))
        zipped.writestr(link, "target")

    with pytest.raises(store.PetStoreError, match="linked or special"):
        store.import_pet_package(archive.getvalue(), filename="linked.zip")

    with pytest.raises(store.PetStoreError, match="encrypted"):
        store.import_pet_package(_set_encrypted_flag(_package()), filename="encrypted.zip")


def test_import_rejects_member_count_and_expanded_size():
    extras = tuple((f"fox/unused-{index}.txt", "x") for index in range(255))
    with pytest.raises(store.PetStoreError, match="more than 256"):
        store.import_pet_package(_package(extras=extras), filename="many.zip")

    archive = bytearray(_package())
    central = archive.index(b"PK\x01\x02")
    struct.pack_into("<I", archive, central + 24, 41 * 1024 * 1024)
    with pytest.raises(store.PetStoreError, match="expands beyond"):
        store.import_pet_package(bytes(archive), filename="large.zip")


def test_import_rejects_oversized_metadata():
    package = _package(metadata={"description": "x" * (1024 * 1024 + 1)})

    with pytest.raises(store.PetStoreError, match="pet.json exceeds"):
        store.import_pet_package(package, filename="metadata.zip")


def test_import_checks_geometry_before_decoding_pixels(monkeypatch):
    def unexpected_load(_self):
        raise AssertionError("pixel decode must not happen before geometry validation")

    monkeypatch.setattr("PIL.PngImagePlugin.PngImageFile.load", unexpected_load)
    with pytest.raises(store.PetStoreError, match="1536x1872"):
        store.import_pet_package(_atlas_bytes((512, 512)), filename="bad.png")


def test_import_rejects_empty_idle_and_disguised_format():
    with pytest.raises(store.PetStoreError, match="idle frame is empty"):
        store.import_pet_package(_atlas_bytes(idle=False), filename="empty.png")

    gif = io.BytesIO()
    Image.new("RGBA", (1536, 1872)).save(gif, format="GIF")
    with pytest.raises(store.PetStoreError, match="PNG or WebP"):
        store.import_pet_package(gif.getvalue(), filename="fake.png")


def test_import_cleans_staging_directory_after_write_failure(monkeypatch):
    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_spritesheet", fail_write)
    with pytest.raises(store.PetStoreError, match="could not install"):
        store.import_pet_package(_atlas_bytes(), filename="fox.png")

    assert not list(store.pets_dir().glob(".import-*"))
