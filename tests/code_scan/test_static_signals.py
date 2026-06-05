"""Tests for scripts/code-scan/static_signals.py (TDD RED first).

Strictly follows bead: ua-tier1-001-static-signals-schema.
- Expect build_static_signals_artifact and signal record helpers.
- Stdlib only in impl.
- Core contract: heuristic_signal + not_validated unless Tier 0 fact.
- Empty artifact shape per spec.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "code-scan"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Import will fail in RED phase
from static_signals import (
    build_static_signals_artifact,
    make_signal_record,
    SignalRecord,
)


class TestSchemaConstants:
    """Ensure core constants / enums are exposed per schema contract."""

    def test_claim_type_heuristic(self):
        # The returned claim_type must be exactly this for non-Tier0 signals
        artifact = build_static_signals_artifact([])
        assert artifact["claim_type"] == "heuristic_signal"

    def test_semantic_status_not_validated(self):
        artifact = build_static_signals_artifact([])
        assert artifact["semantic_status"] == "not_validated"


class TestEmptyArtifact:
    """Exact empty artifact shape per bead spec."""

    def test_empty_artifact_shape(self):
        artifact = build_static_signals_artifact([])
        expected = {
            "schema_version": "1.0.0",
            "claim_type": "heuristic_signal",
            "semantic_status": "not_validated",
            "signals": [],
            "summary": {"total_signals": 0, "by_surface": {}, "by_marker_type": {}},
            "boundaries": [
                "Tier 1 static signals are content markers only; they do not prove security, RLS correctness, auth correctness, runtime behavior, deployment readiness, CI success, or policy semantics."
            ],
        }
        assert artifact == expected
        # Also roundtrip stable
        assert json.loads(json.dumps(artifact)) == expected


class TestSignalRecordHelper:
    """Tests for make_signal_record and SignalRecord (if dataclass or dict helper)."""

    def test_make_signal_record_minimal(self):
        rec = make_signal_record(
            surface="code",
            path="src/foo.py",
            line=42,
            marker_type="supabase_rls_comment",
            marker="row_level_security",
            claim_type="heuristic_signal",
            semantic_status="not_validated",
            boundary="Tier 1 static signals are content markers only.",
        )
        assert isinstance(rec, dict)
        assert rec["surface"] == "code"
        assert rec["path"] == "src/foo.py"
        assert rec["line"] == 42
        assert rec["marker_type"] == "supabase_rls_comment"
        assert rec["marker"] == "row_level_security"
        assert rec["claim_type"] == "heuristic_signal"
        assert rec["semantic_status"] == "not_validated"
        assert "boundary" in rec
        # SignalRecord alias should work too
        rec2 = SignalRecord(
            surface="sql",
            path="migrations/001.sql",
            line=10,
            marker_type="migration_file",
            marker="-- rls",
            claim_type="heuristic_signal",
            semantic_status="not_validated",
            boundary="content marker",
        )
        assert rec2["surface"] == "sql"

    def test_make_signal_record_defaults(self):
        # Helper should accept minimal kwargs and supply safe defaults
        rec = make_signal_record(
            surface="config",
            path="supabase/config.toml",
            line=1,
            marker_type="edge_function_config",
            marker="functions",
        )
        assert rec["claim_type"] == "heuristic_signal"
        assert rec["semantic_status"] == "not_validated"
        assert rec["boundary"] is not None and "content marker" in rec["boundary"].lower() or "does not prove" in rec["boundary"].lower()


class TestArtifactWithSignals:
    """Building artifact with one or more signals."""

    def test_artifact_includes_signals(self):
        sig1 = make_signal_record(
            surface="code",
            path="app/models.py",
            line=15,
            marker_type="supabase_client_init",
            marker="create_client",
        )
        artifact = build_static_signals_artifact([sig1])
        assert artifact["schema_version"] == "1.0.0"
        assert artifact["claim_type"] == "heuristic_signal"
        assert artifact["semantic_status"] == "not_validated"
        assert len(artifact["signals"]) == 1
        assert artifact["signals"][0]["path"] == "app/models.py"
        assert artifact["summary"]["total_signals"] == 1
        assert "code" in artifact["summary"]["by_surface"]
        assert artifact["summary"]["by_surface"]["code"] == 1
        assert "supabase_client_init" in artifact["summary"]["by_marker_type"]
        assert len(artifact["boundaries"]) >= 1
        assert any("does not prove security" in b or "RLS correctness" in b for b in artifact["boundaries"])

    def test_multiple_signals_summary(self):
        sigs = [
            make_signal_record(surface="code", path="a.py", line=1, marker_type="m1", marker="x"),
            make_signal_record(surface="code", path="b.py", line=2, marker_type="m1", marker="y"),
            make_signal_record(surface="sql", path="s.sql", line=3, marker_type="m2", marker="z"),
        ]
        artifact = build_static_signals_artifact(sigs)
        assert artifact["summary"]["total_signals"] == 3
        assert artifact["summary"]["by_surface"]["code"] == 2
        assert artifact["summary"]["by_surface"]["sql"] == 1
        assert artifact["summary"]["by_marker_type"]["m1"] == 2
        assert artifact["summary"]["by_marker_type"]["m2"] == 1


class TestBoundaryContract:
    """Explicit evidence-boundary enforcement tests per bead."""

    def test_every_non_tier0_claim_is_heuristic_and_not_validated(self):
        sig = make_signal_record(
            surface="docs",
            path="README.md",
            line=5,
            marker_type="supabase_url_comment",
            marker="https://...supabase.co",
        )
        artifact = build_static_signals_artifact([sig])
        for s in artifact["signals"]:
            assert s["claim_type"] == "heuristic_signal"
            assert s["semantic_status"] == "not_validated"
        assert artifact["claim_type"] == "heuristic_signal"
        assert artifact["semantic_status"] == "not_validated"

    def test_boundaries_contains_required_disclaimer_text(self):
        artifact = build_static_signals_artifact([])
        boundaries_text = " ".join(artifact["boundaries"])
        assert "Tier 1 static signals are content markers only" in boundaries_text
        assert "do not prove security" in boundaries_text
        assert "RLS correctness" in boundaries_text
        assert "auth correctness" in boundaries_text
        assert "runtime behavior" in boundaries_text
        assert "deployment readiness" in boundaries_text
        assert "CI success" in boundaries_text
        assert "policy semantics" in boundaries_text

    def test_signals_never_contain_validated_claims(self):
        # Even if caller tries to pass something, canonical builder forces heuristic/not_validated
        bad_sig = {
            "surface": "code",
            "path": "evil.py",
            "line": 1,
            "marker_type": "fake",
            "marker": "x",
            "claim_type": "concrete_proof",  # attempt to overclaim
            "semantic_status": "validated",
            "boundary": "ignore",
        }
        artifact = build_static_signals_artifact([bad_sig])
        for s in artifact["signals"]:
            assert s["claim_type"] == "heuristic_signal"
            assert s["semantic_status"] == "not_validated"
