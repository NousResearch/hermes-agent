from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

from scripts.check_test_prune_resurrection import (
    DEFAULT_MANIFEST,
    DEFAULT_TESTS_ROOT,
    encode_fingerprint,
    find_resurrections,
    read_manifest,
)


def _literal_fingerprint(rel: str, qualname: str, source: str) -> bytes:
    node = ast.parse(source).body[0]
    payload = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(f"{rel}::{qualname}\0{payload}".encode()).digest()


def test_exact_pruned_function_is_rejected_but_a_changed_body_is_allowed(tmp_path: Path):
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    original = "def test_old():\n    assert 1 + 1 == 2\n"
    changed = "def test_old():\n    assert 2 + 2 == 4\n"
    banned = {_literal_fingerprint("test_sample.py", "test_old", original)}

    sample = tests_root / "test_sample.py"
    sample.write_text(original, encoding="utf-8")
    assert find_resurrections(tests_root, banned) == ["test_sample.py::test_old"]

    sample.write_text(changed, encoding="utf-8")
    assert find_resurrections(tests_root, banned) == []


def test_manifest_rejects_truncation_and_round_trips_sorted_records(tmp_path: Path):
    manifest = tmp_path / "pruned.bin"
    records = [b"b" * 32, b"a" * 32]
    manifest.write_bytes(b"".join(sorted(records)))
    assert read_manifest(manifest) == set(records)

    manifest.write_bytes(b"short")
    try:
        read_manifest(manifest)
    except ValueError as exc:
        assert "multiple of 32" in str(exc)
    else:  # pragma: no cover - the assertion above is the contract
        raise AssertionError("truncated manifest was accepted")


def test_encode_fingerprint_has_a_stable_literal_value():
    """Regression pin for encode_fingerprint()'s exact output, so a change to its hashing/AST-dump
    behavior is caught explicitly instead of silently invalidating every entry in the committed
    manifest. Two literals: ast.dump(annotate_fields=True) started omitting default-valued fields
    (e.g. arguments(posonlyargs=[], ...) -> arguments()) in 3.12, changing the normalized string
    encode_fingerprint hashes -- requires-python spans 3.11 through 3.13, straddling that cutover,
    so a single literal cannot be correct on the whole supported range."""
    source = "def test_old():\n    assert 1 + 1 == 2\n"
    node = ast.parse(source).body[0]
    expected = (
        "4a057261416941b2004f20f93c1e0b0be9f001c2b38614db8523ded12e4f4edc"
        if sys.version_info >= (3, 12) else
        "339b0b3d26ee7a97fe66ff83df677e67c480ada5bb6ed8c47ef4fa2763c93126"
    )
    assert encode_fingerprint("test_sample.py::test_old", node).hex() == expected


def test_no_dormant_pruned_test_has_returned():
    assert find_resurrections(DEFAULT_TESTS_ROOT, read_manifest(DEFAULT_MANIFEST)) == []
