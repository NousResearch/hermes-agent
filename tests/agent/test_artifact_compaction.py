import hashlib
import json
from pathlib import Path

from agent.artifact_compaction import (
    compact_artifacts_in_messages,
    detect_artifact_type,
    estimate_tokens_rough,
    retrieve_artifact_section,
)


def test_detects_openqp_log_and_stores_summary_card(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = (
        "Open Quantum Platform\n"
        "PyOQP Test Report\n"
        "MRSF-EKT ionization potentials (hartree)\n"
        "Z-vector converged\n"
        "state     energy           strength\n"
        + "SCF iteration line with a long OpenQP diagnostic payload " + "x" * 140 + "\n"
        + ("SCF converged alternative_scf z-vector tddft diagnostic " + "y" * 140 + "\n") * 900
        + "ERROR final diagnostic\n"
    )
    messages = [{"role": "user", "content": raw}]

    compacted, report = compact_artifacts_in_messages(messages, min_chars=1000)

    digest = hashlib.sha256(raw.encode()).hexdigest()
    meta = report["artifacts"][0]
    artifact = Path(meta["path"])
    metadata = Path(meta["metadata_path"])
    assert artifact == tmp_path / "artifacts" / f"{digest}.openqp.log"
    assert metadata == tmp_path / "artifacts" / f"{digest}.metadata.json"
    assert artifact.exists()
    assert metadata.exists()
    assert artifact.read_text() == raw
    saved_meta = json.loads(metadata.read_text(encoding="utf-8"))
    assert saved_meta["sha256"] == digest
    assert saved_meta["metadata_path"] == str(metadata)
    assert saved_meta["detected_sections"]
    card = compacted[0]["content"]
    assert "[HERMES ARTIFACT SUMMARY CARD]" in card
    assert digest in card
    assert "artifact_retrieve" in card
    assert report["artifact_count"] == 1
    assert report["after_total_tokens"] < report["before_total_tokens"]
    assert report["largest_after"][0]["token_estimate"] < report["largest_before"][0]["token_estimate"]


def test_retrieve_artifact_section_by_detected_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = "Open Quantum Platform\nMRSF-EKT ionization potentials\nZ-vector converged\nroot 1\nSCF section\ncycle 1\n" + "tail\n" * 300
    compacted, report = compact_artifacts_in_messages(
        [{"role": "tool", "content": raw}],
        min_chars=100,
        min_tokens=0,
    )
    meta = report["artifacts"][0]

    result = retrieve_artifact_section(meta["artifact_id"], section="mrsf_tddft")

    assert result["artifact"]["sha256"] == meta["sha256"]
    assert "MRSF-EKT ionization potentials" in result["content"]
    assert result["start_line"] <= result["end_line"]


def test_json_dump_compaction_records_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = json.dumps({"records": [{"i": i, "value": "x" * 50} for i in range(400)]}, indent=2)

    compacted, report = compact_artifacts_in_messages([{"role": "user", "content": raw}], min_chars=1000)

    assert detect_artifact_type(raw) == "json"
    meta = report["artifacts"][0]
    assert meta["artifact_type"] == "json"
    assert meta["parser_version"] == "artifact-aware-compaction-v1"
    assert meta["token_estimate"] > 1000
    assert "Detected sections:" in compacted[0]["content"]


def test_artifact_min_tokens_prevents_compacting_small_token_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = "Open Quantum Platform\nSCF\n" + ("x" * 30 + "\n") * 300

    compacted, report = compact_artifacts_in_messages(
        [{"role": "tool", "content": raw}],
        min_chars=1000,
        min_tokens=10_000,
    )

    assert compacted[0]["content"] == raw
    assert report["artifact_count"] == 0
    assert not (tmp_path / "artifacts").exists()


def test_artifact_summary_card_respects_token_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = (
        "Open Quantum Platform\nPyOQP Test Report\n"
        + ("SCF converged alternative_scf z-vector tddft diagnostic " + "x" * 180 + "\n") * 900
    )

    compacted, report = compact_artifacts_in_messages(
        [{"role": "tool", "content": raw}],
        min_chars=1000,
        min_tokens=100,
        max_summary_tokens=120,
    )

    assert report["artifact_count"] == 1
    card = compacted[0]["content"]
    assert estimate_tokens_rough(card) <= 120
    assert "artifact_retrieve" in card
    assert raw not in card


def test_retrieve_artifact_default_max_chars_is_12k(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = "Open Quantum Platform\nSCF section\n" + ("line " + "x" * 100 + "\n") * 500
    _compacted, report = compact_artifacts_in_messages(
        [{"role": "tool", "content": raw}],
        min_chars=100,
        min_tokens=0,
    )
    meta = report["artifacts"][0]

    result = retrieve_artifact_section(meta["artifact_id"], section="scf")

    assert result["truncated"] is True
    assert len(result["content"]) <= 12_100
    assert "[...artifact slice truncated...]" in result["content"]



def test_source_code_gets_symbol_summary_and_symbol_retrieval(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source = (
        "module m_test\n"
        "  use omp_lib\n"
        "  implicit none\n"
        "  real :: work(nroot, nmo), grad(3, natom)\n"
        "contains\n"
        "  subroutine build_sigma(vec)\n"
        "!$omp parallel do\n"
        "    real :: vec(:)\n"
        "  end subroutine build_sigma\n"
        "  function energy() result(e)\n"
        "    real :: e\n"
        "  end function energy\n"
        "end module m_test\n"
    ) * 120
    wrapped = json.dumps({
        "content": "\n".join(f"{i + 1:6}|{line}" for i, line in enumerate(source.splitlines())),
        "total_lines": len(source.splitlines()),
    })

    compacted, report = compact_artifacts_in_messages(
        [{"role": "tool", "content": wrapped}],
        min_chars=1000,
        min_tokens=0,
    )

    assert detect_artifact_type(wrapped, min_chars=1000) == "source_code"
    meta = report["artifacts"][0]
    assert meta["artifact_type"] == "source_code"
    assert meta["path"].endswith(".source.txt")
    assert Path(meta["path"]).read_text(encoding="utf-8").startswith("module m_test")
    card = compacted[0]["content"]
    assert "artifact_label: Source code summary" in card
    assert "modules: m_test" in card
    assert "build_sigma" in card
    assert "OpenMP directives" in card
    assert "work(nroot, nmo)" in card

    result = retrieve_artifact_section(meta["artifact_id"], section="build_sigma", max_chars=500)
    assert result["success"] is True
    assert result["section"] == "subroutine build_sigma"
    assert "subroutine build_sigma" in result["content"]


def test_scientific_artifact_metadata_omits_raw_coordinates(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = ("5\ncomment\nC 0.0 0.0 0.0\nH 0.0 0.0 1.0\nH 0.0 1.0 0.0\nH 1.0 0.0 0.0\nH 1.0 1.0 1.0\n" * 250)

    compacted, report = compact_artifacts_in_messages(
        [{"role": "tool", "content": raw}],
        min_chars=1000,
        min_tokens=0,
    )

    assert detect_artifact_type(raw, min_chars=1000) == "xyz"
    meta = report["artifacts"][0]
    assert meta["artifact_type"] == "xyz"
    card = compacted[0]["content"]
    assert "Scientific artifact metadata" in card
    assert "raw coordinates" in card
    assert "C 0.0 0.0 0.0" not in card
    assert report["context_budget_after"]["token_contribution_by_category"]["scientific_artifacts"] > 0
