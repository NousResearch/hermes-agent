"""Media claim-check spool: mint a ref to bytes, resolve it back, materialize + clean up."""

import pytest

from gateway.media_spool import MediaSpool, MediaRef, materialize_inbound


def test_mint_then_resolve_roundtrips(tmp_path):
    spool = MediaSpool(tmp_path)
    ref = spool.mint(b"hello-bytes", filename="a.png", mime="image/png", kind="image")
    assert isinstance(ref, MediaRef)
    assert ref.mime == "image/png"
    assert ref.kind == "image"
    assert ref.size == len(b"hello-bytes")
    assert spool.resolve(ref.ref) == b"hello-bytes"


def test_refs_are_unique(tmp_path):
    spool = MediaSpool(tmp_path)
    r1 = spool.mint(b"x", filename="a", mime="application/octet-stream", kind="document")
    r2 = spool.mint(b"x", filename="a", mime="application/octet-stream", kind="document")
    assert r1.ref != r2.ref


def test_materialize_writes_named_file(tmp_path):
    spool = MediaSpool(tmp_path)
    ref = spool.mint(b"data", filename="report.pdf", mime="application/pdf", kind="document")
    dest_dir = tmp_path / "cache"
    dest_dir.mkdir()
    path = spool.materialize(ref.ref, dest_dir)
    assert path.read_bytes() == b"data"
    assert path.suffix == ".pdf"


def test_unlink_removes_spooled_bytes(tmp_path):
    spool = MediaSpool(tmp_path)
    ref = spool.mint(b"data", filename="a.png", mime="image/png", kind="image")
    spool.unlink(ref.ref)
    with pytest.raises(KeyError):
        spool.resolve(ref.ref)


def test_resolve_unknown_ref_raises(tmp_path):
    with pytest.raises(KeyError):
        MediaSpool(tmp_path).resolve("nope")


def test_materialize_inbound_reconstructs_files_cross_process(tmp_path):
    # Front mints into the shared spool...
    front = MediaSpool(tmp_path / "spool")
    img = front.mint(b"PNGDATA", filename="pic.png", mime="image/png", kind="image")
    doc = front.mint(b"PDFDATA", filename="r.pdf", mime="application/pdf", kind="document")

    # ...worker resolves from a fresh MediaSpool (no shared in-memory state),
    # using only the wire dicts.
    worker = MediaSpool(tmp_path / "spool")
    cache = tmp_path / "cache"
    cache.mkdir()
    out = materialize_inbound(worker, [img.to_wire(), doc.to_wire()], cache)

    paths = {m.kind: p for p, m in out}
    assert paths["image"].read_bytes() == b"PNGDATA" and paths["image"].suffix == ".png"
    assert paths["document"].read_bytes() == b"PDFDATA" and paths["document"].suffix == ".pdf"


def test_ref_serializes_to_wire_dict_without_path(tmp_path):
    spool = MediaSpool(tmp_path)
    ref = spool.mint(b"d", filename="v.ogg", mime="audio/ogg", kind="voice", is_voice=True)
    wire = ref.to_wire()
    assert wire["is_voice"] is True
    assert "path" not in wire and "root" not in wire
    assert set(wire) == {"ref", "filename", "mime", "kind", "size", "as_document", "is_voice"}
