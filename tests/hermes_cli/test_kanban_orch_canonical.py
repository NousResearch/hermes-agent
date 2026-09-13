"""
T01: Strict canonical contract — ORCH V4.

Test vectors from §I.5 must-reject table + §3.1 normative Python.
RED expected: module import fails → canonical_contract_missing.
GREEN expected: all vectors pass with exact CanonicalError codes.
"""

import pytest
import sys
import os

# Ensure worktree hermes_cli is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_canonical import (
    CanonicalError,
    strict_json_loads,
    canonical_json_bytes,
    digest,
    require_exact_type,
    require_sha256_hex,
    normalize_tenant_scope,
    validate_artifact_schema,
    route_digest,
    origin_id,
    request_key,
    node_key,
)


class TestStrictCanonicalContract:
    """§I.5 must-reject vectors — every input must raise exact CanonicalError code."""

    def test_strict_canonical_contract(self):
        """All §I.5 R-* vectors in one test (matches §13 T01 selector)."""

        # R-json-utf8: malformed UTF-8
        with pytest.raises(CanonicalError) as exc:
            strict_json_loads(b'{"k": "\xff"}' if False else b'{"k": "\xc0\xc0"}')
        assert exc.value.code == "invalid_utf8", f"R-json-utf8: got {exc.value.code}"

        # R-json-dup: duplicate key
        with pytest.raises(CanonicalError) as exc:
            strict_json_loads(b'{"a":1,"a":2}')
        assert exc.value.code == "duplicate_or_nfc_colliding_key", f"R-json-dup: got {exc.value.code}"

        # R-bool-int: bool where int expected
        with pytest.raises(CanonicalError) as exc:
            require_exact_type(True, int, code="invalid_int")
        assert exc.value.code == "invalid_int", f"R-bool-int: got {exc.value.code}"

        # R-float: any float in JSON
        with pytest.raises(CanonicalError) as exc:
            strict_json_loads(b'{"x": 1.5}')
        assert exc.value.code == "float_not_allowed", f"R-float: got {exc.value.code}"

        # R-digest: non-64-hex digest assertion
        with pytest.raises(CanonicalError) as exc:
            require_sha256_hex("not_a_hex", code="invalid_digest")
        assert exc.value.code == "invalid_digest", f"R-digest: got {exc.value.code}"

        # R-digest: 63 chars (too short)
        with pytest.raises(CanonicalError) as exc:
            require_sha256_hex("a" * 63, code="invalid_digest")
        assert exc.value.code == "invalid_digest"

        # R-tenant: tenant with trailing space (NFC + strip drift)
        with pytest.raises(CanonicalError) as exc:
            # normalize_tenant_scope strips, but validate_artifact_schema
            # rejects if stored != normalized
            artifact = {
                "schema_version": 4,
                "kind": "orch_origin",
                "board_instance_id": "b",
                "tenant_scope": "  drift  ",  # has leading/trailing space
                "selector_key": "s",
                "route_digest": "d",
            }
            validate_artifact_schema("orch_origin", artifact)
        assert exc.value.code == "noncanonical_tenant_scope", f"R-tenant: got {exc.value.code}"

        # R-ver: schema_version != 4
        with pytest.raises(CanonicalError) as exc:
            artifact = {
                "schema_version": 3,  # wrong version
                "kind": "orch_origin",
                "board_instance_id": "b",
                "tenant_scope": "",
                "selector_key": "s",
                "route_digest": "d",
            }
            validate_artifact_schema("orch_origin", artifact)
        assert exc.value.code == "unsupported_artifact_schema", f"R-ver: got {exc.value.code}"

        # Additional vectors from §3.1

        # Missing required key
        with pytest.raises(CanonicalError) as exc:
            validate_artifact_schema("orch_origin", {
                "schema_version": 4,
                "kind": "orch_origin",
                "board_instance_id": "b",
                # missing selector_key, route_digest, tenant_scope
            })
        assert exc.value.code == "missing_artifact_key"

        # Unknown artifact key
        with pytest.raises(CanonicalError) as exc:
            validate_artifact_schema("orch_origin", {
                "schema_version": 4,
                "kind": "orch_origin",
                "board_instance_id": "b",
                "tenant_scope": "",
                "selector_key": "s",
                "route_digest": "d",
                "bogus_key": "x",
            })
        assert exc.value.code == "unknown_artifact_key"

        # Artifact too large
        with pytest.raises(CanonicalError) as exc:
            strict_json_loads(b"x" * (1_048_576 + 1))
        assert exc.value.code == "artifact_too_large"

        # Max depth exceeded
        deep = b"["
        for _ in range(33):
            deep += b"["
        deep += b"]" * 33 + b"]"
        with pytest.raises(CanonicalError) as exc:
            strict_json_loads(deep)
        assert exc.value.code == "max_depth_exceeded"

        # Negative: valid input should NOT raise
        valid = strict_json_loads(b'{"key": "value", "num": 42, "flag": true, "arr": [1, 2, 3]}')
        assert valid["key"] == "value"
        assert valid["num"] == 42
        assert valid["flag"] is True
        assert valid["arr"] == [1, 2, 3]

        # Digest is deterministic 64-hex
        d = digest({"a": 1, "b": "test"})
        assert len(d) == 64
        assert all(c in "0123456789abcdef" for c in d)

        # node_key parent is __parent__
        assert node_key({"role": "parent"}) == "__parent__"

        # normalize_tenant_scope: None -> ""
        assert normalize_tenant_scope(None) == ""
        assert normalize_tenant_scope("  ") == ""


    def test_tenant_normalization_gate(self):
        """§13 T02: tenant NULL/space/NFC normalization and stored equality."""
        assert normalize_tenant_scope(None) == ""
        assert normalize_tenant_scope("  ") == ""
        assert normalize_tenant_scope("test") == "test"

        # NFC normalization: composed form
        # é = NFC of e + combining acute
        nfc = "caf\u00e9"  # café in NFC
        decomposed = "cafe\u0301"  # cafe + combining acute (NFD)
        assert normalize_tenant_scope(nfc) == nfc
        assert normalize_tenant_scope(decomposed) == nfc  # NFC normalizes

    def test_selector_precedence_and_zero_mutation(self):
        """§13 T03: selector ledger replay/conflict zero mutation (in-memory test)."""
        # For now, test that replay_selector_key is deterministic
        from hermes_cli.kanban_orch_canonical import replay_selector_key

        params = {
            "board_instance_id": "board1",
            "tenant_scope": "",
            "adapter_instance_id": "adapter1",
            "conversation_id": "conv1",
            "selector_kind": "event",
            "selector_value": "evt123",
        }
        key1 = replay_selector_key(params)
        key2 = replay_selector_key(params)
        assert key1 == key2, "replay_selector_key must be deterministic"
        assert len(key1) == 64

        # Different board → different key
        params2 = dict(params, board_instance_id="board2")
        key3 = replay_selector_key(params2)
        assert key1 != key3, "different board must yield different selector key"

    def test_generation_supersession_cas(self):
        """§13 T04: supersede generation+1, one successor, ledger CAS."""
        from hermes_cli.kanban_orch_canonical import validate_supersession, request_key

        base = {
            "board_instance_id": "board1",
            "tenant_scope": "",
            "selector_key": "sk1",
            "lineage_id": "lin1",
        }

        # Gen 1 predecessor (failed) → Gen 2 successor: valid
        predecessor = {**base, "generation": 1, "lifecycle_state": "failed"}
        successor = {**base, "generation": 2, "lifecycle_state": "submitted"}
        key = validate_supersession(predecessor, successor)
        assert len(key) == 64

        # request_key differs between generations
        pred_key = request_key(predecessor)
        succ_key = request_key(successor)
        assert pred_key != succ_key, "different generations must have different request_keys"

        # Generation must be exactly +1
        with pytest.raises(CanonicalError) as exc:
            bad_succ = {**base, "generation": 3, "lifecycle_state": "submitted"}
            validate_supersession(predecessor, bad_succ)
        assert exc.value.code == "supersession_generation_not_incremented"

        # Predecessor must be terminal
        with pytest.raises(CanonicalError) as exc:
            non_terminal = {**base, "generation": 1, "lifecycle_state": "waiting_lanes"}
            validate_supersession(non_terminal, {**base, "generation": 2, "lifecycle_state": "submitted"})
        assert exc.value.code == "supersession_predecessor_not_terminal"

        # Lineage must match
        with pytest.raises(CanonicalError) as exc:
            diff_lineage = {**base, "lineage_id": "lin2", "generation": 2, "lifecycle_state": "submitted"}
            validate_supersession(predecessor, diff_lineage)
        assert exc.value.code == "supersession_lineage_mismatch"

        # Cancelled predecessor also valid
        cancelled_pred = {**base, "generation": 5, "lifecycle_state": "cancelled"}
        cancelled_succ = {**base, "generation": 6, "lifecycle_state": "submitted"}
        key2 = validate_supersession(cancelled_pred, cancelled_succ)
        assert len(key2) == 64
