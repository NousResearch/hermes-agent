"""Tests for blog.reference_catalog — provider-free reference catalogue (P11).

Uses tmp_path fixture data only; never depends on the real external root.
"""

import hashlib
import json
import os

import pytest

from blog import reference_catalog as rc


def _write_file(root, rel, content=b"img"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _sha(content=b"img"):
    return hashlib.sha256(content).hexdigest()


def _manifest_row(rid, rel, content=None, cls="review-required"):
    content = content if content is not None else b"img"
    return {
        "record_schema_version": "2",
        "reference_id": rid,
        "path": rel,
        "sha256": _sha(content),
        "provenance_class": "sahil_curated",
        "ownership_or_usage_basis": "fixture-only explicit review required",
        "usage_classification": cls,
        "allowed_roles": ["layout", "style", "composition", "palette", "subject"],
        "parent_reference_id": None,
        "collection": "sahil_staging",
        "source": "Sahil staging reference pack",
        "provenance_state": "baseline-import-unreviewed",
        "bytes": len(content),
        "_content": content,
    }


def _core_row(rid, rel, content=None, cls="review-required"):
    content = content if content is not None else b"img"
    return {
        "record_schema_version": "2",
        "reference_id": rid,
        "path": rel,
        "sha256": _sha(content),
        "provenance_class": "sahil_curated",
        "ownership_or_usage_basis": "fixture-only explicit review required",
        "usage_classification": cls,
        "parent_reference_id": None,
        "collection": "sahil_staging",
        "core_role": "composition",
        "core_tag": "abstract-ink",
        "curation_status": "visually-reviewed-core-candidate-2026-07-23",
        "allowed_roles": ["composition"],
        "blocked_roles": ["generation", "publication"],
        "visual_rationale": "abstract systems/editorial texture",
        "_content": content,
    }


def _make_root(tmp_path, baseline_rows, core_rows=None, files=None):
    """Build a tmp root with manifest.jsonl, optional core-pack.jsonl, and files.

    ``files`` maps rel path -> bytes to write for that path. When a path is
    absent from ``files``, the row helper already encoded the intended content
    into the row's sha256; we write bytes that hash to that sha by using the
    same content the row helper used. To keep this trivial, the row helpers
    store their content under a private ``_content`` key so _make_root can
    write it back. Tests that bypass the helpers must pass ``files``.
    """
    root = tmp_path / "refs"
    root.mkdir()
    files = files or {}
    for row in baseline_rows:
        rel = row["path"]
        content = files.get(rel, row.get("_content", b"img"))
        _write_file(root, rel, content)
    with (root / "manifest.jsonl").open("w") as fh:
        for row in baseline_rows:
            clean = {k: v for k, v in row.items() if k != "_content"}
            fh.write(json.dumps(clean) + "\n")
    if core_rows:
        with (root / "core-pack.jsonl").open("w") as fh:
            for row in core_rows:
                clean = {k: v for k, v in row.items() if k != "_content"}
                fh.write(json.dumps(clean) + "\n")
    return root


# ---- valid baseline + core load ----------------------------------------

def test_valid_baseline_and_core_load(tmp_path):
    content = b"\x89PNGfake"
    root = _make_root(
        tmp_path,
        baseline_rows=[
            _manifest_row("ref-1", "Referencecontent/a.png", content),
            _manifest_row("ref-2", "Referencecontent/b.png", content),
        ],
        core_rows=[
            _core_row("ref-1", "Referencecontent/a.png", content),
        ],
    )
    cat = rc.ReferenceCatalog.load(root)
    assert len(cat.baseline) == 2
    assert len(cat.core) == 1
    rec = cat.get("ref-1")
    assert rec is not None
    assert rec.is_core is True
    assert rec.sha256 == _sha(content)
    assert rec.usage_classification == "review-required"


def test_no_core_pack_is_optional(tmp_path):
    root = _make_root(tmp_path, baseline_rows=[_manifest_row("ref-1", "a.png")])
    cat = rc.ReferenceCatalog.load(root)
    assert len(cat.core) == 0
    assert len(cat.baseline) == 1


# ---- hash/path/ID mismatch rejection ------------------------------------

def _write_jsonl(path, rows):
    """Write rows as JSONL, stripping the private _content key."""
    with path.open("w") as fh:
        for row in rows:
            clean = {k: v for k, v in row.items() if k != "_content"}
            fh.write(json.dumps(clean) + "\n")


def test_sha256_mismatch_rejected(tmp_path):
    row = _manifest_row("ref-1", "a.png")
    row["sha256"] = "0" * 64
    root = _make_root(tmp_path, baseline_rows=[row])
    with pytest.raises(rc.CatalogIntegrityError, match="sha256 mismatch"):
        rc.ReferenceCatalog.load(root)


def test_missing_file_rejected(tmp_path):
    root = tmp_path / "refs"
    root.mkdir()
    row = _manifest_row("ref-1", "missing.png")
    _write_jsonl(root / "manifest.jsonl", [row])
    with pytest.raises(rc.CatalogIntegrityError, match="missing file"):
        rc.ReferenceCatalog.load(root)


def test_absolute_path_rejected(tmp_path):
    root = tmp_path / "refs"
    root.mkdir()
    _write_file(root, "a.png")
    row = _manifest_row("ref-1", "a.png")
    row["path"] = "/etc/passwd"
    _write_jsonl(root / "manifest.jsonl", [row])
    with pytest.raises(rc.CatalogIntegrityError, match="must be relative"):
        rc.ReferenceCatalog.load(root)


def test_escaping_path_rejected(tmp_path):
    root = tmp_path / "refs"
    root.mkdir()
    _write_file(root, "a.png")
    row = _manifest_row("ref-1", "../escape.png")
    _write_jsonl(root / "manifest.jsonl", [row])
    with pytest.raises(rc.CatalogIntegrityError, match="escape"):
        rc.ReferenceCatalog.load(root)


def test_duplicate_id_rejected(tmp_path):
    root = tmp_path / "refs"
    root.mkdir()
    _write_file(root, "a.png")
    _write_file(root, "b.png")
    row1 = _manifest_row("dup", "a.png")
    row2 = _manifest_row("dup", "b.png")
    _write_jsonl(root / "manifest.jsonl", [row1, row2])
    with pytest.raises(rc.CatalogIntegrityError, match="duplicate reference_id"):
        rc.ReferenceCatalog.load(root)


def test_core_without_baseline_rejected(tmp_path):
    """A core row whose reference_id is not in baseline must be rejected."""
    root = _make_root(
        tmp_path,
        baseline_rows=[_manifest_row("ref-1", "a.png")],
        core_rows=[_core_row("ref-2", "a.png")],
    )
    with pytest.raises(rc.CatalogIntegrityError, match="no baseline row"):
        rc.ReferenceCatalog.load(root)


def test_core_hash_differs_from_baseline_rejected(tmp_path):
    """A core row with the same id as a baseline row but a different hash
    must be rejected (core must match the baseline hash for the same id)."""
    root = tmp_path / "refs"
    root.mkdir()
    _write_file(root, "a.png", b"img")
    _write_file(root, "b.png", b"different")
    base = _manifest_row("ref-1", "a.png", b"img")
    # Core shares the id but resolves to a different content-addressed baseline.
    core = _core_row("ref-1", "b.png", b"different")
    _write_jsonl(root / "manifest.jsonl", [base])
    _write_jsonl(root / "core-pack.jsonl", [core])
    with pytest.raises(rc.CatalogIntegrityError, match="path/hash must match baseline"):
        rc.ReferenceCatalog.load(root)


# ---- core-set only contract enumeration ---------------------------------

def test_records_for_contract_returns_core_only(tmp_path):
    content = b"img"
    root = _make_root(
        tmp_path,
        baseline_rows=[
            _manifest_row("ref-1", "a.png", content),
            _manifest_row("ref-2", "b.png", content),
        ],
        core_rows=[_core_row("ref-1", "a.png", content)],
    )
    cat = rc.ReferenceCatalog.load(root)
    contract = cat.records_for_contract()
    assert len(contract) == 1
    assert contract[0].reference_id == "ref-1"
    assert contract[0].is_core is True


# ---- fail-closed generation request for review-required -----------------

def test_generation_fails_closed_for_review_required(tmp_path):
    content = b"img"
    root = _make_root(
        tmp_path,
        baseline_rows=[
            _manifest_row("ref-1", "a.png", content),
        ],
        core_rows=[_core_row("ref-1", "a.png", content, cls="review-required")],
    )
    cat = rc.ReferenceCatalog.load(root)
    with pytest.raises(rc.GenerationEligibilityError, match="not permitted"):
        cat.references_for_generation(["ref-1"])


def test_generation_rejects_baseline_only_id(tmp_path):
    """A baseline-only id (not in core) must fail closed for generation."""
    content = b"img"
    root = _make_root(
        tmp_path,
        baseline_rows=[
            _manifest_row("ref-1", "a.png", content),
            _manifest_row("ref-2", "b.png", content),
        ],
        core_rows=[_core_row("ref-1", "a.png", content, cls="permitted")],
    )
    cat = rc.ReferenceCatalog.load(root)
    # ref-2 is baseline-only and thus not a core record.
    with pytest.raises(rc.GenerationEligibilityError, match="not a core record"):
        cat.references_for_generation(["ref-2"])


def test_generation_rejects_unknown_id(tmp_path):
    root = _make_root(tmp_path, baseline_rows=[_manifest_row("ref-1", "a.png")])
    cat = rc.ReferenceCatalog.load(root)
    with pytest.raises(rc.GenerationEligibilityError, match="unknown reference id"):
        cat.references_for_generation(["nope"])


# ---- explicit permitted record succeeds in generation eligibility only ---

def test_blocked_classification_loads_and_fails_closed(tmp_path):
    """A baseline row with usage_classification=blocked loads fine and fails
    closed for generation (blocked is more restrictive than review-required)."""
    content = b"img"
    root = _make_root(
        tmp_path,
        baseline_rows=[
            _manifest_row("ref-1", "a.png", content, cls="blocked"),
        ],
        core_rows=[_core_row("ref-1", "a.png", content, cls="blocked")],
    )
    cat = rc.ReferenceCatalog.load(root)
    record = cat.get("ref-1")
    assert record is not None
    assert record.usage_classification == "blocked"
    with pytest.raises(rc.GenerationEligibilityError, match="not permitted"):
        cat.references_for_generation(["ref-1"])


def test_invalid_classification_rejected(tmp_path):
    content = b"img"
    root = tmp_path / "refs"
    root.mkdir()
    _write_file(root, "a.png", content)
    row = _manifest_row("ref-1", "a.png", content)
    row["usage_classification"] = "maybe"
    _write_jsonl(root / "manifest.jsonl", [row])
    with pytest.raises(rc.CatalogIntegrityError, match="invalid usage_classification"):
        rc.ReferenceCatalog.load(root)


def test_permitted_record_succeeds_for_generation(tmp_path):
    content = b"img"
    root = _make_root(
        tmp_path,
        baseline_rows=[
            _manifest_row("ref-1", "a.png", content),
        ],
        core_rows=[_core_row("ref-1", "a.png", content, cls="permitted")],
    )
    cat = rc.ReferenceCatalog.load(root)
    recs = cat.references_for_generation(["ref-1"])
    assert len(recs) == 1
    assert recs[0].reference_id == "ref-1"
    assert recs[0].usage_classification == "permitted"
    # eligibility only — no generation happens here; this is just the contract.


def test_schema_v2_taxonomy_is_preserved_and_core_is_not_double_counted(tmp_path):
    root = _make_root(
        tmp_path,
        baseline_rows=[_manifest_row("ref-1", "a.png")],
        core_rows=[_core_row("ref-1", "a.png")],
    )
    cat = rc.ReferenceCatalog.load(root)
    record = cat.get("ref-1")
    assert record is not None
    assert record.provenance_class == "sahil_curated"
    assert record.ownership_or_usage_basis == "fixture-only explicit review required"
    assert record.allowed_roles == ("composition",)
    assert len(cat) == 1
    assert [r.reference_id for r in cat.all_records()] == ["ref-1"]


def test_manifest_row_missing_schema_v2_provenance_is_rejected(tmp_path):
    row = _manifest_row("ref-1", "a.png")
    del row["provenance_class"]
    root = _make_root(tmp_path, baseline_rows=[row])
    with pytest.raises(rc.CatalogIntegrityError, match="provenance_class"):
        rc.ReferenceCatalog.load(root)


def test_resolved_symlink_outside_root_is_rejected(tmp_path):
    root = tmp_path / "refs"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    os.symlink(outside, root / "linked.png")
    row = _manifest_row("ref-1", "linked.png", b"outside")
    _write_jsonl(root / "manifest.jsonl", [row])
    with pytest.raises(rc.CatalogIntegrityError, match="resolves outside"):
        rc.ReferenceCatalog.load(root)
