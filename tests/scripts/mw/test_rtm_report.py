"""Hermetic tests for scripts/mw/rtm_report.py (MW-P3-I2c).

All fixtures live under tmp_path — no network, no real project tree.
Covers L1/L2/L3 layers + fail-closed (no-result never verified).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# load module under test (path-stable; no package install required)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
RTM_PATH = REPO_ROOT / "scripts" / "mw" / "rtm_report.py"

_spec = importlib.util.spec_from_file_location("mw_rtm_report", RTM_PATH)
assert _spec and _spec.loader
rtm = importlib.util.module_from_spec(_spec)
sys.modules["mw_rtm_report"] = rtm
_spec.loader.exec_module(rtm)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _register(ids: List[str]) -> str:
    lines = ["# REQ register", "", "| ID | Title |", "|----|-------|"]
    for i, rid in enumerate(ids, 1):
        lines.append(f"| {rid} | requirement {i} |")
    lines.append("")
    return "\n".join(lines)


def _junit(*cases: Dict[str, Any]) -> str:
    """Build a minimal JUnit XML string.

    Each case: name, classname?, status in {pass, fail, error, skip}, message?
    """
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<testsuite>"]
    for c in cases:
        name = c["name"]
        cls = c.get("classname", "tests.sample")
        status = c.get("status", "pass")
        msg = c.get("message", "boom")
        open_tag = f'<testcase classname="{cls}" name="{name}" time="0.01">'
        if status == "fail":
            parts.append(open_tag)
            parts.append(f'  <failure message="{msg}">failed</failure>')
            parts.append("</testcase>")
        elif status == "error":
            parts.append(open_tag)
            parts.append(f'  <error message="{msg}">errored</error>')
            parts.append("</testcase>")
        elif status == "skip":
            parts.append(open_tag)
            parts.append(f'  <skipped message="{msg}"/>')
            parts.append("</testcase>")
        else:
            parts.append(f'<testcase classname="{cls}" name="{name}" time="0.01"/>')
    parts.append("</testsuite>")
    return "\n".join(parts)


def _config(
    *,
    mode: str = "mapfile",
    results_mode: str = "junit",
    results_path: str = "test-results.xml",
    min_pct: float = 100,
    map_file: str = ".work/rtm-map.yaml",
    test_glob: str = "tests/**/*.py",
    marker_pattern: str = r"REQ[:=\s]+((?:REQ|R)-\d+)",
    id_pattern: str = r"(?:REQ|R)-\d+",
) -> str:
    # Single-quoted YAML scalars so regex backslashes (\d, \s) stay literal.
    if mode == "markers":
        tmap = textwrap.dedent(
            f"""\
            test_map:
              mode: markers
              test_glob: '{test_glob}'
              marker_pattern: '{marker_pattern}'
            """
        )
    else:
        tmap = textwrap.dedent(
            f"""\
            test_map:
              mode: mapfile
              file: '{map_file}'
            """
        )
    return textwrap.dedent(
        f"""\
        req_register:
          file: '.work/req-register.md'
          id_pattern: '{id_pattern}'
        """
    ) + tmap + textwrap.dedent(
        f"""\
        results:
          mode: {results_mode}
          path: '{results_path}'
        thresholds:
          min_verified_pct: {min_pct}
        """
    )


def _mapfile(mapping: Dict[str, List[str]]) -> str:
    lines = []
    for rid, tests in mapping.items():
        if not tests:
            lines.append(f"{rid}: []")
            continue
        lines.append(f"{rid}:")
        for t in tests:
            lines.append(f'  - "{t}"')
    return "\n".join(lines) + "\n"


def _setup_project(
    tmp: Path,
    *,
    reqs: List[str],
    mapping: Optional[Dict[str, List[str]]] = None,
    junit_cases: Optional[List[Dict[str, Any]]] = None,
    simple_lines: Optional[str] = None,
    mode: str = "mapfile",
    min_pct: float = 100,
    results_mode: str = "junit",
    extra_test_files: Optional[Dict[str, str]] = None,
) -> Path:
    """Create a minimal project tree under tmp and return root."""
    _write(tmp / ".work" / "req-register.md", _register(reqs))
    if mode == "mapfile":
        assert mapping is not None
        _write(tmp / ".work" / "rtm-map.yaml", _mapfile(mapping))
    if extra_test_files:
        for rel, body in extra_test_files.items():
            _write(tmp / rel, body)
    if simple_lines is not None:
        _write(tmp / "results.txt", simple_lines)
        results_path = "results.txt"
        results_mode = "simple"
    else:
        cases = junit_cases or []
        _write(tmp / "test-results.xml", _junit(*cases))
        results_path = "test-results.xml"
    cfg = _config(
        mode=mode,
        results_mode=results_mode,
        results_path=results_path,
        min_pct=min_pct,
    )
    _write(tmp / ".work" / "rtm.yaml", cfg)
    return tmp


def _run(root: Path, *extra: str, config: Optional[str] = None) -> int:
    argv = ["--root", str(root)]
    if config is not None:
        argv.extend(["--config", config])
    else:
        argv.extend(["--config", str(root / ".work" / "rtm.yaml")])
    argv.extend(extra)
    return rtm.run(argv)


def _run_json(root: Path, *extra: str) -> Dict[str, Any]:
    code = _run(root, "--json", *extra)
    # caller checks code; we re-run build for structured data via API
    cfg = rtm.load_config(root / ".work" / "rtm.yaml")
    # parse min override if present
    thr = None
    if "--min-verified-pct" in extra:
        i = list(extra).index("--min-verified-pct")
        thr = float(extra[i + 1])
    report = rtm.build_report_from_config(root, cfg, threshold_override=thr)
    return code, report.to_json_dict()


# ---------------------------------------------------------------------------
# full trace / layers
# ---------------------------------------------------------------------------


def test_full_trace_three_reqs_all_pass_exit_0(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2", "R-3"],
        mapping={
            "R-1": ["test_one"],
            "R-2": ["test_two"],
            "R-3": ["test_three"],
        },
        junit_cases=[
            {"name": "test_one", "status": "pass"},
            {"name": "test_two", "status": "pass"},
            {"name": "test_three", "status": "pass"},
        ],
    )
    code, data = _run_json(root)
    assert code == rtm.EXIT_OK
    assert data["total"] == 3
    assert data["l2_count"] == 3
    assert data["l3_count"] == 3
    assert data["verified_pct"] == 100.0
    assert data["complete"] is True
    for row in data["reqs"]:
        assert row["l1"] is True
        assert row["l2"] is True
        assert row["l3"] is True


def test_l1_only_req_without_test_in_no_test(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2"],
        mapping={"R-1": ["test_one"]},  # R-2 has no mapping
        junit_cases=[{"name": "test_one", "status": "pass"}],
    )
    code, data = _run_json(root)
    assert code == rtm.EXIT_BELOW  # 50% < 100
    assert data["total"] == 2
    assert data["l2_count"] == 1
    assert data["l3_count"] == 1
    assert data["verified_pct"] == 50.0
    assert "R-2" in data["gaps"]["no_test"]
    r2 = next(r for r in data["reqs"] if r["id"] == "R-2")
    assert r2["l1"] is True
    assert r2["l2"] is False
    assert r2["l3"] is False


def test_l2_not_l3_failing_test(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2"],
        mapping={"R-1": ["test_ok"], "R-2": ["test_bad"]},
        junit_cases=[
            {"name": "test_ok", "status": "pass"},
            {"name": "test_bad", "status": "fail"},
        ],
    )
    code, data = _run_json(root)
    assert code == rtm.EXIT_BELOW
    r2 = next(r for r in data["reqs"] if r["id"] == "R-2")
    assert r2["l2"] is True
    assert r2["l3"] is False
    assert r2["reason"] == "test failed"
    nv = {g["id"]: g["reason"] for g in data["gaps"]["not_verified"]}
    assert nv["R-2"] == "test failed"
    assert data["l3_count"] == 1
    assert data["verified_pct"] == 50.0


def test_l2_not_l3_no_result_fail_closed(tmp_path: Path) -> None:
    """Mapped test present in map but absent from results -> NOT verified."""
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2"],
        mapping={"R-1": ["test_ok"], "R-2": ["test_ghost"]},
        junit_cases=[
            {"name": "test_ok", "status": "pass"},
            # test_ghost intentionally missing
        ],
    )
    code, data = _run_json(root)
    assert code == rtm.EXIT_BELOW
    r2 = next(r for r in data["reqs"] if r["id"] == "R-2")
    assert r2["l2"] is True
    assert r2["l3"] is False
    assert r2["reason"] == "no result"
    # fail closed: must NOT count as verified
    assert data["l3_count"] == 1
    assert "R-2" not in [
        r["id"] for r in data["reqs"] if r["l3"]
    ]


# ---------------------------------------------------------------------------
# markers / mapfile modes
# ---------------------------------------------------------------------------


def test_markers_mode_maps_req_to_enclosing_test(tmp_path: Path) -> None:
    # Markers must sit after (inside) the def so the preceding-test bind works.
    test_body = textwrap.dedent(
        """\
        # sample test module

        def test_x():
            # REQ: R-1
            assert True

        def test_y():
            # REQ: R-2
            assert True
        """
    )
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2"],
        mode="markers",
        extra_test_files={"tests/test_sample.py": test_body},
        junit_cases=[
            {"name": "test_x", "status": "pass"},
            {"name": "test_y", "status": "pass"},
        ],
    )
    code, data = _run_json(root)
    assert code == rtm.EXIT_OK
    r1 = next(r for r in data["reqs"] if r["id"] == "R-1")
    r2 = next(r for r in data["reqs"] if r["id"] == "R-2")
    assert "test_x" in r1["tests"]
    assert "test_y" in r2["tests"]
    assert r1["l3"] is True and r2["l3"] is True


def test_mapfile_mode_explicit_mapping_loads(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["REQ-10", "REQ-20"],
        mapping={
            "REQ-10": ["test_alpha", "test_beta"],
            "REQ-20": ["test_gamma"],
        },
        junit_cases=[
            {"name": "test_alpha", "status": "pass"},
            {"name": "test_beta", "status": "pass"},
            {"name": "test_gamma", "status": "pass"},
        ],
    )
    code, data = _run_json(root)
    assert code == rtm.EXIT_OK
    r10 = next(r for r in data["reqs"] if r["id"] == "REQ-10")
    assert r10["tests"] == ["test_alpha", "test_beta"]
    assert data["total"] == 2
    assert data["l3_count"] == 2


def test_orphan_marker_not_in_register(tmp_path: Path) -> None:
    test_body = textwrap.dedent(
        """\
        def test_known():
            # REQ: R-1
            pass

        def test_orphan():
            # REQ: R-99
            pass
        """
    )
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mode="markers",
        extra_test_files={"tests/test_orph.py": test_body},
        junit_cases=[{"name": "test_known", "status": "pass"}],
    )
    code, data = _run_json(root)
    assert code == rtm.EXIT_OK
    assert data["total"] == 1  # no phantom R-99
    assert all(r["id"] != "R-99" for r in data["reqs"])
    orphans = data["gaps"]["orphan_mappings"]
    assert any(o.get("req") == "R-99" for o in orphans)


# ---------------------------------------------------------------------------
# results parsers
# ---------------------------------------------------------------------------


def test_junit_classifies_failure_error_skipped_pass(tmp_path: Path) -> None:
    xml = _junit(
        {"name": "t_pass", "status": "pass"},
        {"name": "t_fail", "status": "fail"},
        {"name": "t_error", "status": "error"},
        {"name": "t_skip", "status": "skip"},
    )
    p = _write(tmp_path / "j.xml", xml)
    results = rtm.parse_junit(p)
    assert results["t_pass"] == "pass"
    assert results["t_fail"] == "fail"
    assert results["t_error"] == "fail"  # error -> fail
    assert results["t_skip"] == "skip"
    # composite keys also present
    assert results.get("tests.sample.t_pass") == "pass"


def test_simple_results_mode(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2", "R-3"],
        mapping={
            "R-1": ["test_a"],
            "R-2": ["test_b"],
            "R-3": ["test_c"],
        },
        simple_lines="test_a PASS\ntest_b FAIL\ntest_c SKIP\n",
    )
    code, data = _run_json(root)
    assert data["l3_count"] == 1
    r1 = next(r for r in data["reqs"] if r["id"] == "R-1")
    r2 = next(r for r in data["reqs"] if r["id"] == "R-2")
    r3 = next(r for r in data["reqs"] if r["id"] == "R-3")
    assert r1["l3"] is True
    assert r2["l3"] is False and r2["reason"] == "test failed"
    assert r3["l3"] is False and r3["reason"] == "no result"
    assert code == rtm.EXIT_BELOW


def test_parse_simple_unit(tmp_path: Path) -> None:
    p = _write(tmp_path / "r.txt", "foo PASS\nbar FAIL\nbaz SKIP\n")
    r = rtm.parse_simple_results(p)
    assert r == {"foo": "pass", "bar": "fail", "baz": "skip"}


# ---------------------------------------------------------------------------
# errors / exit 2
# ---------------------------------------------------------------------------


def test_empty_register_exit_2(tmp_path: Path) -> None:
    _write(tmp_path / ".work" / "req-register.md", "# empty\nno ids here\n")
    _write(tmp_path / ".work" / "rtm-map.yaml", "R-1:\n  - t\n")
    _write(tmp_path / "test-results.xml", _junit({"name": "t", "status": "pass"}))
    _write(
        tmp_path / ".work" / "rtm.yaml",
        _config(mode="mapfile"),
    )
    code = _run(tmp_path)
    assert code == rtm.EXIT_ERR


def test_missing_config_exit_2(tmp_path: Path) -> None:
    # no .work/rtm.yaml
    code = rtm.run(["--root", str(tmp_path)])
    assert code == rtm.EXIT_ERR


def test_missing_config_explicit_path_exit_2(tmp_path: Path) -> None:
    code = rtm.run(
        ["--root", str(tmp_path), "--config", str(tmp_path / "nope.yaml")]
    )
    assert code == rtm.EXIT_ERR


def test_unparseable_junit_exit_2(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mapping={"R-1": ["test_a"]},
        junit_cases=[{"name": "test_a", "status": "pass"}],
    )
    # overwrite with garbage
    _write(root / "test-results.xml", "<not-valid-junit>>>")
    code = _run(root)
    assert code == rtm.EXIT_ERR


def test_unparseable_simple_results_exit_2(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mapping={"R-1": ["test_a"]},
        simple_lines="this is not a valid results line at all\n",
    )
    code = _run(root)
    assert code == rtm.EXIT_ERR


# ---------------------------------------------------------------------------
# threshold overrides
# ---------------------------------------------------------------------------


def test_threshold_100_with_75_exit_1(tmp_path: Path) -> None:
    # 3 reqs, 3 mapped, only 2 pass -> ~66.7%; use 4 reqs for clean 75%
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2", "R-3", "R-4"],
        mapping={
            "R-1": ["t1"],
            "R-2": ["t2"],
            "R-3": ["t3"],
            "R-4": ["t4"],
        },
        junit_cases=[
            {"name": "t1", "status": "pass"},
            {"name": "t2", "status": "pass"},
            {"name": "t3", "status": "pass"},
            {"name": "t4", "status": "fail"},
        ],
        min_pct=100,
    )
    code, data = _run_json(root)
    assert data["verified_pct"] == 75.0
    assert data["complete"] is False
    assert code == rtm.EXIT_BELOW


def test_threshold_cli_override_50_with_75_exit_0(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2", "R-3", "R-4"],
        mapping={
            "R-1": ["t1"],
            "R-2": ["t2"],
            "R-3": ["t3"],
            "R-4": ["t4"],
        },
        junit_cases=[
            {"name": "t1", "status": "pass"},
            {"name": "t2", "status": "pass"},
            {"name": "t3", "status": "pass"},
            {"name": "t4", "status": "fail"},
        ],
        min_pct=100,  # config says 100; CLI overrides to 50
    )
    code, data = _run_json(root, "--min-verified-pct", "50")
    assert data["verified_pct"] == 75.0
    assert data["threshold"] == 50.0
    assert data["complete"] is True
    assert code == rtm.EXIT_OK


# ---------------------------------------------------------------------------
# json shape + live counts
# ---------------------------------------------------------------------------


def test_json_shape_and_live_counts_recompute(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2", "R-3"],
        mapping={
            "R-1": ["t1"],
            "R-2": ["t2"],
            # R-3 no map
        },
        junit_cases=[
            {"name": "t1", "status": "pass"},
            {"name": "t2", "status": "fail"},
        ],
        min_pct=50,
    )
    code, data = _run_json(root)
    # required top-level keys
    for key in (
        "total",
        "l2_count",
        "l3_count",
        "verified_pct",
        "threshold",
        "complete",
        "reqs",
        "gaps",
    ):
        assert key in data
    for key in ("no_test", "not_verified", "orphan_mappings"):
        assert key in data["gaps"]

    # live recompute from reqs must equal reported counts
    reqs = data["reqs"]
    assert data["total"] == len(reqs)
    assert data["l2_count"] == sum(1 for r in reqs if r["l2"])
    assert data["l3_count"] == sum(1 for r in reqs if r["l3"])
    expected_pct = round(100.0 * data["l3_count"] / data["total"], 1)
    assert data["verified_pct"] == expected_pct
    assert data["verified_pct"] == round(100.0 * 1 / 3, 1)  # 33.3
    assert code == rtm.EXIT_BELOW  # 33.3 < 50


def test_human_output_contains_summary_and_gaps(tmp_path: Path, capsys) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-5", "R-12"],
        mapping={
            "R-1": ["t1"],
            "R-12": ["t12"],
            # R-5 no test
        },
        junit_cases=[
            {"name": "t1", "status": "pass"},
            {"name": "t12", "status": "fail"},
        ],
    )
    code = _run(root)
    out = capsys.readouterr().out
    assert "RTM:" in out
    assert "requirements" in out
    assert "L2" in out and "L3" in out
    assert "no test: R-5" in out
    assert "R-12 (test failed)" in out
    assert "INCOMPLETE" in out
    assert code == rtm.EXIT_BELOW


def test_error_counts_as_fail_not_l3(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mapping={"R-1": ["t_err"]},
        junit_cases=[{"name": "t_err", "status": "error"}],
    )
    code, data = _run_json(root)
    assert data["l3_count"] == 0
    assert data["reqs"][0]["reason"] == "test failed"
    assert code == rtm.EXIT_BELOW


def test_skip_does_not_grant_l3(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mapping={"R-1": ["t_skip"]},
        junit_cases=[{"name": "t_skip", "status": "skip"}],
    )
    _, data = _run_json(root)
    assert data["reqs"][0]["l2"] is True
    assert data["reqs"][0]["l3"] is False
    assert data["reqs"][0]["reason"] == "no result"


def test_mixed_pass_and_fail_does_not_grant_l3(tmp_path: Path) -> None:
    """Pass+fail on mapped tests must fail closed (not hide the failure)."""
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mapping={"R-1": ["t_fail", "t_pass"]},
        junit_cases=[
            {"name": "t_fail", "status": "fail"},
            {"name": "t_pass", "status": "pass"},
        ],
    )
    code, data = _run_json(root)
    row = data["reqs"][0]
    assert row["l2"] is True
    assert row["l3"] is False
    assert "fail" in (row["reason"] or "").lower() or "conflict" in (
        row["reason"] or ""
    ).lower()
    assert row["reason"] == "test failed (conflict)"
    nv = {g["id"]: g["reason"] for g in data["gaps"]["not_verified"]}
    assert "R-1" in nv
    assert code == rtm.EXIT_BELOW
    assert data["l3_count"] == 0


def test_flexible_junit_classname_match(tmp_path: Path) -> None:
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mapping={"R-1": ["test_widget"]},
        junit_cases=[
            {
                "name": "test_widget",
                "classname": "tests.test_mod",
                "status": "pass",
            }
        ],
    )
    code, data = _run_json(root)
    assert data["l3_count"] == 1
    assert code == rtm.EXIT_OK


def test_default_config_path_under_root(tmp_path: Path) -> None:
    """Without --config, resolve <root>/.work/rtm.yaml."""
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mapping={"R-1": ["t1"]},
        junit_cases=[{"name": "t1", "status": "pass"}],
    )
    code = rtm.run(["--root", str(root)])
    assert code == rtm.EXIT_OK


def test_parse_req_register_ordered_unique(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "reg.md",
        "R-2 first\nR-1 second\nR-2 again\nREQ-3 third\n",
    )
    ids = rtm.parse_req_register(p, r"(?:REQ|R)-\d+")
    assert ids == ["R-2", "R-1", "REQ-3"]


def test_mini_yaml_mapfile_without_pyyaml(tmp_path: Path, monkeypatch) -> None:
    """Mapfile loads via mini-YAML fallback when forced."""
    monkeypatch.setattr(rtm, "_FORCE_MINI_YAML", True)
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mapping={"R-1": ["t1"]},
        junit_cases=[{"name": "t1", "status": "pass"}],
    )
    code, data = _run_json(root)
    assert code == rtm.EXIT_OK
    assert data["l3_count"] == 1


def test_counts_never_hardcoded_mixed_gaps(tmp_path: Path) -> None:
    """Live counts: 40 would not appear — derive from this fixture's 5 reqs."""
    reqs = [f"R-{i}" for i in range(1, 6)]
    mapping = {
        "R-1": ["t1"],
        "R-2": ["t2"],
        "R-3": ["t3"],
        # R-4, R-5 no tests
    }
    root = _setup_project(
        tmp_path,
        reqs=reqs,
        mapping=mapping,
        junit_cases=[
            {"name": "t1", "status": "pass"},
            {"name": "t2", "status": "pass"},
            {"name": "t3", "status": "fail"},
        ],
    )
    _, data = _run_json(root)
    assert data["total"] == 5
    assert data["l2_count"] == 3
    assert data["l3_count"] == 2
    assert data["verified_pct"] == 40.0
    # recompute
    assert data["l2_count"] == sum(1 for r in data["reqs"] if r["l2"])
    assert data["l3_count"] == sum(1 for r in data["reqs"] if r["l3"])
    assert set(data["gaps"]["no_test"]) == {"R-4", "R-5"}


# ---------------------------------------------------------------------------
# FIX ROUND 1 regressions (false-verified paths)
# ---------------------------------------------------------------------------


def test_reg_mixed_pass_fail_conflict_not_verified(tmp_path: Path) -> None:
    """FIX 1: REQ mapped to [pass, fail] is not L3 (conflict)."""
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mapping={"R-1": ["pass_test", "fail_test"]},
        junit_cases=[
            {"name": "pass_test", "status": "pass"},
            {"name": "fail_test", "status": "fail"},
        ],
    )
    code, data = _run_json(root)
    row = data["reqs"][0]
    assert row["l3"] is False
    assert "fail" in (row["reason"] or "").lower() or "conflict" in (
        row["reason"] or ""
    ).lower()
    assert any(g["id"] == "R-1" for g in data["gaps"]["not_verified"])
    assert code == rtm.EXIT_BELOW


def test_reg_short_name_collision_ambiguous_not_borrow_pass(tmp_path: Path) -> None:
    """FIX 2: bare test_x with pkgA pass + pkgB fail is ambiguous, not pass."""
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2"],
        mapping={
            "R-1": ["test_x"],
            "R-2": ["pkgB::test_x"],
        },
        junit_cases=[
            {"name": "test_x", "classname": "pkgA", "status": "pass"},
            {"name": "test_x", "classname": "pkgB", "status": "fail"},
        ],
    )
    code, data = _run_json(root)
    bare = next(r for r in data["reqs"] if r["id"] == "R-1")
    exact = next(r for r in data["reqs"] if r["id"] == "R-2")
    assert bare["l3"] is False
    assert bare["reason"] == "ambiguous test id"
    assert exact["l3"] is False
    assert exact["reason"] == "test failed"
    # Direct lookup: bare must not resolve to pass
    results = rtm.parse_junit(root / "test-results.xml")
    assert rtm.lookup_outcome("test_x", results) == rtm.OUTCOME_AMBIGUOUS
    assert rtm.lookup_outcome("pkgB::test_x", results) == rtm.OUTCOME_FAIL
    assert code == rtm.EXIT_BELOW


def test_reg_marker_binds_preceding_only_not_following(tmp_path: Path) -> None:
    """FIX 4: marker between test_a and test_b binds to test_a; pre-file unbound."""
    test_body = textwrap.dedent(
        """\
        def test_a():
            assert True

        # REQ: R-1

        def test_b():
            assert True
        """
    )
    unbound_body = textwrap.dedent(
        """\
        # REQ: R-2
        def test_only():
            assert True
        """
    )
    root = _setup_project(
        tmp_path,
        reqs=["R-1", "R-2"],
        mode="markers",
        extra_test_files={
            "tests/test_between.py": test_body,
            "tests/test_unbound.py": unbound_body,
        },
        junit_cases=[
            {"name": "test_a", "status": "pass"},
            {"name": "test_b", "status": "pass"},
            {"name": "test_only", "status": "pass"},
        ],
    )
    code, data = _run_json(root)
    r1 = next(r for r in data["reqs"] if r["id"] == "R-1")
    r2 = next(r for r in data["reqs"] if r["id"] == "R-2")
    assert r1["tests"] == ["test_a"]
    assert "test_b" not in r1["tests"]
    # Unbound marker before any def: no mapping to test_only
    assert r2["tests"] == []
    assert r2["l2"] is False
    assert r2["l3"] is False
    assert "R-2" in data["gaps"]["no_test"]
    orphans = data["gaps"]["orphan_mappings"]
    assert any(
        o.get("req") == "R-2" and o.get("test") == "(unbound)" for o in orphans
    )
    # R-1 should still verify via preceding test_a
    assert r1["l3"] is True
    assert code == rtm.EXIT_BELOW  # R-2 incomplete


def test_reg_rounded_pct_does_not_control_complete() -> None:
    """FIX 5: 2000/2001 rounds to 100.0% display but complete is False at thr 100."""
    req_ids = [f"R-{i}" for i in range(1, 2002)]
    mapping = {rid: [f"t{i}"] for i, rid in enumerate(req_ids, 1)}
    # 2000 pass, 1 fail
    results = {f"t{i}": rtm.OUTCOME_PASS for i in range(1, 2001)}
    results["t2001"] = rtm.OUTCOME_FAIL
    report = rtm.compute_report(
        req_ids, mapping, results, orphans=[], threshold=100.0
    )
    assert report.total == 2001
    assert report.l3_count == 2000
    assert report.verified_pct == 100.0  # display rounds
    assert report.complete is False
    # CLI exit path mirrors complete flag
    assert report.complete is False


def test_reg_junit_suite_errors_not_attributable_exit_2(tmp_path: Path) -> None:
    """FIX 6: suite errors=N with no per-case error/failure nodes -> exit 2."""
    root = _setup_project(
        tmp_path,
        reqs=["R-1"],
        mapping={"R-1": ["test_ok"]},
        junit_cases=[{"name": "test_ok", "status": "pass"}],
    )
    # Overwrite with suite-level error not visible on any testcase
    _write(
        root / "test-results.xml",
        textwrap.dedent(
            """\
            <?xml version="1.0" encoding="UTF-8"?>
            <testsuite name="suite" errors="1" failures="0" tests="1">
              <testcase classname="pkg" name="test_ok" time="0.01"/>
            </testsuite>
            """
        ),
    )
    code = _run(root)
    assert code == rtm.EXIT_ERR
    # parse_junit itself must refuse
    with pytest.raises(ValueError, match="unreliable"):
        rtm.parse_junit(root / "test-results.xml")
