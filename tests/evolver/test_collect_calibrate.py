"""Invariant tests for the factory collector and calibration math."""
import json

from evolver.archive import FAILURE_CLASSES
from evolver.calibrate import calibrate_credit, separation_summary
from evolver.collect import collect_fix_results, record_from_fix_result


def _fix_result(**over):
    data = {
        "issue": 110649,
        "branch": "fix/x",
        "commit": "abc123",
        "status": "parked_contested",
        "red_on_base": "test_x failed on base",
        "tests": {"test_a.py": "passed", "test_b.py": "failed"},
        "note": "some note",
    }
    data.update(over)
    return data


def test_record_from_fix_result_maps_fields():
    record = record_from_fix_result(_fix_result(), source_path="/tmp/1.json")
    assert record["task"]["task_id"] == "hermes-fix-110649"
    assert record["task"]["kind"] == "bugfix"
    assert record["failure_class"] == "regression"  # red_on_base present
    assert record["failure_class"] in FAILURE_CLASSES
    statuses = {s["status"]["code"] for s in record["trace"]["spans"]}
    assert statuses == {"ok", "error"}


def test_record_without_red_proof_is_test_failure():
    record = record_from_fix_result(_fix_result(red_on_base=""))
    assert record["failure_class"] == "test_failure"


def test_record_without_tests_is_unknown():
    record = record_from_fix_result(_fix_result(tests={}, red_on_base=""))
    assert record["failure_class"] == "unknown"


def test_collect_reads_dir_and_reports_bad_files(tmp_path):
    good = tmp_path / "1.json"
    good.write_text(json.dumps(_fix_result()), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{nope", encoding="utf-8")
    records, errors = collect_fix_results(tmp_path)
    assert len(records) == 1
    assert len(errors) == 1
    assert "bad.json" in errors[0]


def _case_report(**gates):
    return {"name": "x", "gates": gates}


def _g(passed):
    return {"passed": passed, "detail": {}, "duration_s": 0.0}


def test_separation_summary_counts_expectations():
    reports = [
        _case_report(validity_good=_g(True), validity_empty=_g(False),
                     validity_syntax_break=_g(False), validity_noop=_g(True),
                     activation_good=_g(True), activation_noop=_g(False)),
        _case_report(validity_good=_g(True), validity_empty=_g(False),
                     validity_syntax_break=_g(False), validity_noop=_g(True),
                     activation_good=_g(False), activation_noop=_g(False)),
    ]
    sep = separation_summary(reports)
    assert sep["validity_good"] == {"pass": 2, "total": 2,
                                   "expected_pass": True, "as_expected": 2}
    assert sep["activation_good"]["as_expected"] == 1  # one case missed


def test_calibrate_credit_scenarios_behave():
    results = calibrate_credit()
    assert results["clear_lift"]["correct"]
    assert results["no_lift"]["correct"]
    assert results["negative_lift"]["correct"]
    assert results["thin_data"]["correct"]
    assert results["unpaired"]["correct"]
