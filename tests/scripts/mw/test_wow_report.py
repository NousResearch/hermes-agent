"""Hermetic tests for scripts/mw/wow_report.py (MW-P3-I2g).

All fixtures live under tmp_path — no network, no real project tree.
Values must come from files only (no inline metric numbers in config).
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
WOW_PATH = REPO_ROOT / "scripts" / "mw" / "wow_report.py"

_spec = importlib.util.spec_from_file_location("mw_wow_report", WOW_PATH)
assert _spec and _spec.loader
wow = importlib.util.module_from_spec(_spec)
sys.modules["mw_wow_report"] = wow
_spec.loader.exec_module(wow)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _write_json(path: Path, obj: Any) -> Path:
    return _write(path, json.dumps(obj, indent=2) + "\n")


def _config(metrics: List[Dict[str, Any]], tolerance: float = 0) -> str:
    """Build a block-style wow.yaml from metric dicts (mini-YAML friendly)."""
    lines = [f"tolerance: {tolerance}", "metrics:"]
    for m in metrics:
        lines.append(f"  - name: {m['name']}")
        lines.append(f"    direction: {m['direction']}")
        if m.get("unit") is not None:
            lines.append(f"    unit: \"{m['unit']}\"")
        if "tolerance" in m:
            lines.append(f"    tolerance: {m['tolerance']}")
        for side in ("before", "after"):
            sc = m[side]
            lines.append(f"    {side}:")
            if isinstance(sc, dict) and "file" in sc:
                lines.append(f"      file: \"{sc['file']}\"")
                ext = sc.get("extract") or {}
                lines.append("      extract:")
                for ek, ev in ext.items():
                    if isinstance(ev, bool):
                        lines.append(f"        {ek}: {'true' if ev else 'false'}")
                    elif isinstance(ev, (int, float)):
                        lines.append(f"        {ek}: {ev}")
                    else:
                        # quote regex / paths so backslashes stay literal
                        lines.append(f"        {ek}: '{ev}'")
            else:
                # inline value on purpose (for config-error tests)
                lines.append(f"      value: {sc}")
    return "\n".join(lines) + "\n"


def _run(root: Path, cfg: Path, *extra: str) -> int:
    argv = ["--root", str(root), "--config", str(cfg), *extra]
    return wow.run(argv)


def _run_json(root: Path, cfg: Path, *extra: str) -> Dict[str, Any]:
    argv = ["--root", str(root), "--config", str(cfg), "--json", *extra]
    # Capture stdout via building report API when possible; CLI path for exit codes.
    report = wow.build_report(root, wow.load_config(cfg))
    return report.to_json_dict()


# ---------------------------------------------------------------------------
# fixtures: full improved suite
# ---------------------------------------------------------------------------


def _setup_all_improved(tmp: Path) -> Path:
    """Write baseline + current files where every metric improves."""
    # Lighthouse performance 0.5 -> 0.9 (scaled *100 => 50 -> 90)
    _write_json(
        tmp / "baseline" / "lh.json",
        {"categories": {"performance": {"score": 0.5}}},
    )
    _write_json(
        tmp / "current" / "lh.json",
        {"categories": {"performance": {"score": 0.9}}},
    )
    # axe violations 10 -> 3
    _write_json(tmp / "baseline" / "axe.json", {"violations": list(range(10))})
    _write_json(tmp / "current" / "axe.json", {"violations": list(range(3))})
    # gate 2/4 -> 4/4
    _write(tmp / "baseline" / "gate.txt", "gate result: 2 / 4 items\n")
    _write(tmp / "current" / "gate.txt", "gate result: 4 / 4 items\n")
    # rtm 50 -> 100
    _write_json(tmp / "baseline" / "rtm.json", {"verified_pct": 50})
    _write_json(tmp / "current" / "rtm.json", {"verified_pct": 100})
    # hero size: 2048 bytes (~2.0 KB) -> 1024 bytes (1.0 KB)
    _write_bytes(tmp / "baseline" / "hero.jpg", b"x" * 2048)
    _write_bytes(tmp / "current" / "hero.jpg", b"x" * 1024)

    metrics = [
        {
            "name": "lighthouse_perf",
            "direction": "higher_better",
            "unit": "score",
            "before": {
                "file": "baseline/lh.json",
                "extract": {
                    "json_path": "categories.performance.score",
                    "scale": 100,
                },
            },
            "after": {
                "file": "current/lh.json",
                "extract": {
                    "json_path": "categories.performance.score",
                    "scale": 100,
                },
            },
        },
        {
            "name": "axe_violations",
            "direction": "lower_better",
            "before": {
                "file": "baseline/axe.json",
                "extract": {"json_len": "violations"},
            },
            "after": {
                "file": "current/axe.json",
                "extract": {"json_len": "violations"},
            },
        },
        {
            "name": "gate_pass",
            "direction": "higher_better",
            "unit": "ratio",
            "before": {
                "file": "baseline/gate.txt",
                "extract": {"regex_ratio": r"(\d+)\s*/\s*(\d+)"},
            },
            "after": {
                "file": "current/gate.txt",
                "extract": {"regex_ratio": r"(\d+)\s*/\s*(\d+)"},
            },
        },
        {
            "name": "rtm_verified_pct",
            "direction": "higher_better",
            "before": {
                "file": "baseline/rtm.json",
                "extract": {"json_path": "verified_pct"},
            },
            "after": {
                "file": "current/rtm.json",
                "extract": {"json_path": "verified_pct"},
            },
        },
        {
            "name": "hero_size_kb",
            "direction": "lower_better",
            "before": {
                "file": "baseline/hero.jpg",
                "extract": {"file_size_kb": True},
            },
            "after": {
                "file": "current/hero.jpg",
                "extract": {"file_size_kb": True},
            },
        },
    ]
    cfg_path = _write(tmp / ".work" / "wow.yaml", _config(metrics))
    return cfg_path


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_all_metrics_improved_exit_0(tmp_path: Path) -> None:
    cfg = _setup_all_improved(tmp_path)
    code = _run(tmp_path, cfg)
    assert code == wow.EXIT_OK

    data = _run_json(tmp_path, cfg)
    assert data["complete"] is True
    assert data["any_regression"] is False
    assert data["summary"]["improved"] == 5
    assert data["summary"]["regressed"] == 0
    assert data["summary"]["no_data"] == 0
    for m in data["metrics"]:
        assert m["status"] == "IMPROVED", m

    # Live summary counts equal recomputed statuses
    recomputed = {
        "improved": sum(1 for m in data["metrics"] if m["status"] == "IMPROVED"),
        "regressed": sum(1 for m in data["metrics"] if m["status"] == "REGRESSED"),
        "unchanged": sum(1 for m in data["metrics"] if m["status"] == "UNCHANGED"),
        "no_data": sum(1 for m in data["metrics"] if m["status"] == "NO_DATA"),
    }
    assert data["summary"] == recomputed


def test_regression_fail_on_regression_exit_1(tmp_path: Path) -> None:
    # higher_better: 90 -> 50 is a regression
    _write_json(tmp_path / "baseline" / "lh.json", {"score": 90})
    _write_json(tmp_path / "current" / "lh.json", {"score": 50})
    metrics = [
        {
            "name": "lighthouse_perf",
            "direction": "higher_better",
            "before": {
                "file": "baseline/lh.json",
                "extract": {"json_path": "score"},
            },
            "after": {
                "file": "current/lh.json",
                "extract": {"json_path": "score"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    code = _run(tmp_path, cfg, "--fail-on-regression")
    assert code == wow.EXIT_FAIL

    data = _run_json(tmp_path, cfg)
    assert data["metrics"][0]["status"] == "REGRESSED"
    assert data["any_regression"] is True
    # without flag still exit 0
    assert _run(tmp_path, cfg) == wow.EXIT_OK


def test_lower_better_axe_improved_and_regressed(tmp_path: Path) -> None:
    # 10 -> 3 improved
    _write_json(tmp_path / "baseline" / "axe.json", {"violations": list(range(10))})
    _write_json(tmp_path / "current" / "axe.json", {"violations": list(range(3))})
    metrics = [
        {
            "name": "axe_violations",
            "direction": "lower_better",
            "before": {
                "file": "baseline/axe.json",
                "extract": {"json_len": "violations"},
            },
            "after": {
                "file": "current/axe.json",
                "extract": {"json_len": "violations"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    assert data["metrics"][0]["before"] == 10
    assert data["metrics"][0]["after"] == 3
    assert data["metrics"][0]["status"] == "IMPROVED"

    # 3 -> 10 regressed
    _write_json(tmp_path / "baseline" / "axe.json", {"violations": list(range(3))})
    _write_json(tmp_path / "current" / "axe.json", {"violations": list(range(10))})
    data2 = _run_json(tmp_path, cfg)
    assert data2["metrics"][0]["status"] == "REGRESSED"


def test_extractors_json_path_scale_json_len_regex_ratio_file_size(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "baseline" / "lh.json",
        {"categories": {"performance": {"score": 0.4}}},
    )
    _write_json(
        tmp_path / "current" / "lh.json",
        {"categories": {"performance": {"score": 0.8}}},
    )
    _write_json(tmp_path / "baseline" / "axe.json", {"violations": [1, 2, 3, 4]})
    _write_json(tmp_path / "current" / "axe.json", {"violations": [1]})
    _write(tmp_path / "baseline" / "gate.txt", "pass 1/2\n")
    _write(tmp_path / "current" / "gate.txt", "pass 2/2\n")
    _write_bytes(tmp_path / "baseline" / "hero.jpg", b"a" * 3072)  # 3.0 KB
    _write_bytes(tmp_path / "current" / "hero.jpg", b"b" * 1024)  # 1.0 KB

    metrics = [
        {
            "name": "lh",
            "direction": "higher_better",
            "before": {
                "file": "baseline/lh.json",
                "extract": {
                    "json_path": "categories.performance.score",
                    "scale": 100,
                },
            },
            "after": {
                "file": "current/lh.json",
                "extract": {
                    "json_path": "categories.performance.score",
                    "scale": 100,
                },
            },
        },
        {
            "name": "axe",
            "direction": "lower_better",
            "before": {
                "file": "baseline/axe.json",
                "extract": {"json_len": "violations"},
            },
            "after": {
                "file": "current/axe.json",
                "extract": {"json_len": "violations"},
            },
        },
        {
            "name": "gate",
            "direction": "higher_better",
            "before": {
                "file": "baseline/gate.txt",
                "extract": {"regex_ratio": r"(\d+)/(\d+)"},
            },
            "after": {
                "file": "current/gate.txt",
                "extract": {"regex_ratio": r"(\d+)/(\d+)"},
            },
        },
        {
            "name": "size",
            "direction": "lower_better",
            "before": {
                "file": "baseline/hero.jpg",
                "extract": {"file_size_kb": True},
            },
            "after": {
                "file": "current/hero.jpg",
                "extract": {"file_size_kb": True},
            },
        },
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    by_name = {m["name"]: m for m in data["metrics"]}

    assert by_name["lh"]["before"] == 40.0
    assert by_name["lh"]["after"] == 80.0
    assert by_name["axe"]["before"] == 4
    assert by_name["axe"]["after"] == 1
    assert by_name["gate"]["before"] == 0.5
    assert by_name["gate"]["after"] == 1.0
    assert by_name["gate"]["n"] == 2
    assert by_name["gate"]["m"] == 2
    assert by_name["size"]["before"] == 3.0
    assert by_name["size"]["after"] == 1.0


def test_regex_ratio_m_zero_is_no_data(tmp_path: Path) -> None:
    _write(tmp_path / "baseline" / "gate.txt", "0/0 done\n")
    _write(tmp_path / "current" / "gate.txt", "1/2 done\n")
    metrics = [
        {
            "name": "gate_pass",
            "direction": "higher_better",
            "before": {
                "file": "baseline/gate.txt",
                "extract": {"regex_ratio": r"(\d+)/(\d+)"},
            },
            "after": {
                "file": "current/gate.txt",
                "extract": {"regex_ratio": r"(\d+)/(\d+)"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    m = data["metrics"][0]
    assert m["status"] == "NO_DATA"
    assert m["before"] is None
    assert m["delta"] is None
    assert "M==0" in (m["reason"] or "")


def test_missing_source_no_data_not_fabricated(tmp_path: Path) -> None:
    _write_json(tmp_path / "baseline" / "rtm.json", {"verified_pct": 80})
    # current missing on purpose
    metrics = [
        {
            "name": "rtm_verified_pct",
            "direction": "higher_better",
            "before": {
                "file": "baseline/rtm.json",
                "extract": {"json_path": "verified_pct"},
            },
            "after": {
                "file": "current/rtm.json",
                "extract": {"json_path": "verified_pct"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    m = data["metrics"][0]
    assert m["status"] == "NO_DATA"
    assert m["after"] is None
    assert m["delta"] is None
    assert m["pct_change"] is None
    assert data["summary"]["improved"] == 0
    assert data["summary"]["no_data"] == 1
    assert data["complete"] is False

    # without --require-complete: exit 0
    assert _run(tmp_path, cfg) == wow.EXIT_OK
    # with --require-complete: exit 1
    assert _run(tmp_path, cfg, "--require-complete") == wow.EXIT_FAIL


def test_tolerance_exactly_unchanged_beyond_regressed(tmp_path: Path) -> None:
    # higher_better, before=100, after=99, tolerance=1 -> UNCHANGED
    _write_json(tmp_path / "baseline" / "s.json", {"v": 100})
    _write_json(tmp_path / "current" / "s.json", {"v": 99})
    metrics = [
        {
            "name": "score",
            "direction": "higher_better",
            "tolerance": 1,
            "before": {
                "file": "baseline/s.json",
                "extract": {"json_path": "v"},
            },
            "after": {
                "file": "current/s.json",
                "extract": {"json_path": "v"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics, tolerance=0))
    data = _run_json(tmp_path, cfg)
    assert data["metrics"][0]["status"] == "UNCHANGED"

    # beyond tolerance: after=98 -> REGRESSED
    _write_json(tmp_path / "current" / "s.json", {"v": 98})
    data2 = _run_json(tmp_path, cfg)
    assert data2["metrics"][0]["status"] == "REGRESSED"

    # lower_better: before=10, after=11, tol=1 -> UNCHANGED; after=12 -> REGRESSED
    _write_json(tmp_path / "baseline" / "a.json", {"violations": list(range(10))})
    _write_json(tmp_path / "current" / "a.json", {"violations": list(range(11))})
    metrics_lb = [
        {
            "name": "axe",
            "direction": "lower_better",
            "tolerance": 1,
            "before": {
                "file": "baseline/a.json",
                "extract": {"json_len": "violations"},
            },
            "after": {
                "file": "current/a.json",
                "extract": {"json_len": "violations"},
            },
        }
    ]
    cfg2 = _write(tmp_path / "wow2.yaml", _config(metrics_lb))
    assert _run_json(tmp_path, cfg2)["metrics"][0]["status"] == "UNCHANGED"
    _write_json(tmp_path / "current" / "a.json", {"violations": list(range(12))})
    assert _run_json(tmp_path, cfg2)["metrics"][0]["status"] == "REGRESSED"


def test_inline_value_no_file_exit_2(tmp_path: Path) -> None:
    # Craft config with inline value (no file) for before
    bad = textwrap.dedent(
        """\
        tolerance: 0
        metrics:
          - name: fake_score
            direction: higher_better
            before:
              value: 95
            after:
              file: "current/s.json"
              extract:
                json_path: 'v'
        """
    )
    _write_json(tmp_path / "current" / "s.json", {"v": 99})
    cfg = _write(tmp_path / "wow.yaml", bad)
    code = _run(tmp_path, cfg)
    assert code == wow.EXIT_ERR

    # Also bare number as before via loader
    with pytest.raises(ValueError, match="must read from a file"):
        wow.evaluate_metric(
            tmp_path,
            {
                "name": "x",
                "direction": "higher_better",
                "before": 42,
                "after": {
                    "file": "current/s.json",
                    "extract": {"json_path": "v"},
                },
            },
            0.0,
        )


def test_path_escape_dotdot_is_no_data(tmp_path: Path) -> None:
    _write_json(tmp_path / "baseline" / "ok.json", {"v": 1})
    _write_json(tmp_path / "current" / "ok.json", {"v": 2})
    # Escape attempt on after
    metrics = [
        {
            "name": "escape_me",
            "direction": "higher_better",
            "before": {
                "file": "baseline/ok.json",
                "extract": {"json_path": "v"},
            },
            "after": {
                "file": "../outside.json",
                "extract": {"json_path": "v"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    m = data["metrics"][0]
    assert m["status"] == "NO_DATA"
    assert m["after"] is None
    assert "escapes" in (m["reason"] or "").lower() or "path" in (
        m["reason"] or ""
    ).lower()


def test_json_shape_and_live_summary_counts(tmp_path: Path) -> None:
    cfg = _setup_all_improved(tmp_path)
    # flip one metric to regression for mixed summary
    _write_json(
        tmp_path / "current" / "rtm.json",
        {"verified_pct": 10},  # worse than baseline 50
    )
    # remove hero after for one no_data
    (tmp_path / "current" / "hero.jpg").unlink()

    data = _run_json(tmp_path, cfg)
    assert set(data.keys()) == {
        "metrics",
        "summary",
        "complete",
        "any_regression",
    }
    assert set(data["summary"].keys()) == {
        "improved",
        "regressed",
        "unchanged",
        "no_data",
    }
    for m in data["metrics"]:
        for key in (
            "name",
            "direction",
            "unit",
            "before",
            "after",
            "delta",
            "pct_change",
            "status",
            "n",
            "reason",
        ):
            assert key in m

    # Live recompute equals reported
    s = data["summary"]
    assert s["improved"] == sum(
        1 for m in data["metrics"] if m["status"] == "IMPROVED"
    )
    assert s["regressed"] == sum(
        1 for m in data["metrics"] if m["status"] == "REGRESSED"
    )
    assert s["unchanged"] == sum(
        1 for m in data["metrics"] if m["status"] == "UNCHANGED"
    )
    assert s["no_data"] == sum(
        1 for m in data["metrics"] if m["status"] == "NO_DATA"
    )
    assert data["complete"] is (s["no_data"] == 0)
    assert data["any_regression"] is (s["regressed"] > 0)
    assert s["regressed"] >= 1
    assert s["no_data"] >= 1
    assert s["improved"] >= 1


def test_config_missing_exit_2(tmp_path: Path) -> None:
    # no --config and no .work/wow.yaml
    code = wow.run(["--root", str(tmp_path)])
    assert code == wow.EXIT_ERR

    # explicit missing path
    code2 = wow.run(
        ["--root", str(tmp_path), "--config", str(tmp_path / "nope.yaml")]
    )
    assert code2 == wow.EXIT_ERR


def test_default_config_path_under_root(tmp_path: Path) -> None:
    _setup_all_improved(tmp_path)
    # resolve without --config should find .work/wow.yaml
    code = wow.run(["--root", str(tmp_path)])
    assert code == wow.EXIT_OK


def test_human_output_table_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    cfg = _setup_all_improved(tmp_path)
    code = _run(tmp_path, cfg)
    assert code == wow.EXIT_OK
    out = capsys.readouterr().out
    assert "metric | before | after" in out
    assert "IMPROVED" in out
    assert "wow:" in out
    assert "improved" in out


def test_regex_single_capture(tmp_path: Path) -> None:
    _write(tmp_path / "baseline" / "score.txt", "score=40\n")
    _write(tmp_path / "current" / "score.txt", "score=80\n")
    metrics = [
        {
            "name": "score_txt",
            "direction": "higher_better",
            "before": {
                "file": "baseline/score.txt",
                "extract": {"regex": r"score=(\d+)"},
            },
            "after": {
                "file": "current/score.txt",
                "extract": {"regex": r"score=(\d+)"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    assert data["metrics"][0]["before"] == 40
    assert data["metrics"][0]["after"] == 80
    assert data["metrics"][0]["status"] == "IMPROVED"
    assert data["metrics"][0]["pct_change"] == 100.0


def test_json_len_not_array_is_no_data(tmp_path: Path) -> None:
    _write_json(tmp_path / "baseline" / "axe.json", {"violations": "not-a-list"})
    _write_json(tmp_path / "current" / "axe.json", {"violations": [1]})
    metrics = [
        {
            "name": "axe",
            "direction": "lower_better",
            "before": {
                "file": "baseline/axe.json",
                "extract": {"json_len": "violations"},
            },
            "after": {
                "file": "current/axe.json",
                "extract": {"json_len": "violations"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    assert data["metrics"][0]["status"] == "NO_DATA"
    assert data["metrics"][0]["before"] is None


def test_pct_change_null_when_before_zero(tmp_path: Path) -> None:
    _write_json(tmp_path / "baseline" / "s.json", {"v": 0})
    _write_json(tmp_path / "current" / "s.json", {"v": 5})
    metrics = [
        {
            "name": "from_zero",
            "direction": "higher_better",
            "before": {
                "file": "baseline/s.json",
                "extract": {"json_path": "v"},
            },
            "after": {
                "file": "current/s.json",
                "extract": {"json_path": "v"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    assert data["metrics"][0]["status"] == "IMPROVED"
    assert data["metrics"][0]["pct_change"] is None
    assert data["metrics"][0]["delta"] == 5


def test_equal_values_are_unchanged(tmp_path: Path) -> None:
    _write_json(tmp_path / "baseline" / "s.json", {"v": 50})
    _write_json(tmp_path / "current" / "s.json", {"v": 50})
    metrics = [
        {
            "name": "same",
            "direction": "higher_better",
            "before": {
                "file": "baseline/s.json",
                "extract": {"json_path": "v"},
            },
            "after": {
                "file": "current/s.json",
                "extract": {"json_path": "v"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    assert data["metrics"][0]["status"] == "UNCHANGED"
    assert data["metrics"][0]["delta"] == 0
    assert data["summary"]["unchanged"] == 1


def test_mini_yaml_loader_works(tmp_path: Path) -> None:
    """Force mini-YAML path so tests pass without PyYAML."""
    old = wow._FORCE_MINI_YAML
    wow._FORCE_MINI_YAML = True
    try:
        cfg = _setup_all_improved(tmp_path)
        data = wow.load_config(cfg)
        assert isinstance(data, dict)
        assert isinstance(data["metrics"], list)
        assert len(data["metrics"]) == 5
        report = wow.build_report(tmp_path, data)
        assert report.improved == 5
    finally:
        wow._FORCE_MINI_YAML = old


def test_contained_path_rejects_dotdot() -> None:
    root = Path("/tmp/fake-root-for-wow").resolve()
    path, err = wow.contained_path(root, "../etc/passwd")
    assert path is None
    assert err is not None
    assert "escapes" in err


# ---------------------------------------------------------------------------
# FIX ROUND 1 regressions (MW-P3-I2g) — fail-closed numbers
# ---------------------------------------------------------------------------


def test_non_finite_json_values_are_no_data_not_status(tmp_path: Path) -> None:
    """FIX A: Infinity/NaN from a file must never become IMPROVED/UNCHANGED."""
    # higher_better: finite 10 → Infinity must NOT look like IMPROVED
    _write(tmp_path / "baseline" / "s.json", '{"v": 10}\n')
    _write(tmp_path / "current" / "s.json", '{"v": Infinity}\n')
    metrics = [
        {
            "name": "score",
            "direction": "higher_better",
            "before": {
                "file": "baseline/s.json",
                "extract": {"json_path": "v"},
            },
            "after": {
                "file": "current/s.json",
                "extract": {"json_path": "v"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    m = data["metrics"][0]
    assert m["status"] == "NO_DATA"
    assert m["after"] is None
    assert m["delta"] is None
    assert m["status"] != "IMPROVED"

    # NaN after equal before must NOT look like UNCHANGED
    _write(tmp_path / "current" / "s.json", '{"v": NaN}\n')
    data2 = _run_json(tmp_path, cfg)
    m2 = data2["metrics"][0]
    assert m2["status"] == "NO_DATA"
    assert m2["after"] is None
    assert m2["status"] != "UNCHANGED"

    # scale: Infinity in config is a config error (exit 2), not a bogus value
    bad_scale = textwrap.dedent(
        """\
        tolerance: 0
        metrics:
          - name: scaled
            direction: higher_better
            before:
              file: "baseline/s.json"
              extract:
                json_path: 'v'
                scale: Infinity
            after:
              file: "current/s.json"
              extract:
                json_path: 'v'
                scale: Infinity
        """
    )
    _write(tmp_path / "current" / "s.json", '{"v": 20}\n')
    cfg_bad = _write(tmp_path / "wow_scale.yaml", bad_scale)
    assert _run(tmp_path, cfg_bad) == wow.EXIT_ERR


def test_regex_ratio_ambiguous_multi_match_is_no_data(tmp_path: Path) -> None:
    """FIX B: first-match is a false number — require exactly one match."""
    # Two ratios in one file: date 12/31 and gate 19/19 — must not pick 12/31
    _write(
        tmp_path / "baseline" / "gate.txt",
        "date 12/31 ... gate 19/19\n",
    )
    _write(
        tmp_path / "current" / "gate.txt",
        "date 12/31 ... gate 19/19\n",
    )
    metrics = [
        {
            "name": "gate_pass",
            "direction": "higher_better",
            "before": {
                "file": "baseline/gate.txt",
                "extract": {"regex_ratio": r"(\d+)\s*/\s*(\d+)"},
            },
            "after": {
                "file": "current/gate.txt",
                "extract": {"regex_ratio": r"(\d+)\s*/\s*(\d+)"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics))
    data = _run_json(tmp_path, cfg)
    m = data["metrics"][0]
    assert m["status"] == "NO_DATA"
    assert m["before"] is None
    assert m["after"] is None
    assert "ambiguous" in (m["reason"] or "").lower()
    # Must NOT have fabricated first-match 12/31 → 12/31 = 1.0 or 0.387...
    assert m["before"] != 12 / 31
    assert m["after"] != 12 / 31

    # Exactly one ratio → value ok
    _write(tmp_path / "baseline" / "gate.txt", "gate result: 2 / 4 items\n")
    _write(tmp_path / "current" / "gate.txt", "gate result: 4 / 4 items\n")
    data2 = _run_json(tmp_path, cfg)
    m2 = data2["metrics"][0]
    assert m2["before"] == 0.5
    assert m2["after"] == 1.0
    assert m2["status"] == "IMPROVED"
    assert m2["n"] == 4
    assert m2["m"] == 4


def test_tolerance_symmetric_improvement_band(tmp_path: Path) -> None:
    """FIX C: noise within tolerance is UNCHANGED, not IMPROVED."""
    # higher_better, before=90 after=90.4 tolerance=1 → UNCHANGED (not IMPROVED)
    _write_json(tmp_path / "baseline" / "s.json", {"v": 90})
    _write_json(tmp_path / "current" / "s.json", {"v": 90.4})
    metrics = [
        {
            "name": "score",
            "direction": "higher_better",
            "tolerance": 1,
            "before": {
                "file": "baseline/s.json",
                "extract": {"json_path": "v"},
            },
            "after": {
                "file": "current/s.json",
                "extract": {"json_path": "v"},
            },
        }
    ]
    cfg = _write(tmp_path / "wow.yaml", _config(metrics, tolerance=0))
    data = _run_json(tmp_path, cfg)
    assert data["metrics"][0]["status"] == "UNCHANGED"
    assert data["metrics"][0]["status"] != "IMPROVED"

    # after=92 clears the band → IMPROVED
    _write_json(tmp_path / "current" / "s.json", {"v": 92})
    data2 = _run_json(tmp_path, cfg)
    assert data2["metrics"][0]["status"] == "IMPROVED"


def test_inline_value_alongside_file_exit_2(tmp_path: Path) -> None:
    """FIX D: file + extract + inline value must be rejected (exit 2)."""
    _write_json(tmp_path / "baseline" / "s.json", {"v": 50})
    _write_json(tmp_path / "current" / "s.json", {"v": 80})
    bad = textwrap.dedent(
        """\
        tolerance: 0
        metrics:
          - name: smuggled
            direction: higher_better
            before:
              file: "baseline/s.json"
              extract:
                json_path: 'v'
              value: 99
            after:
              file: "current/s.json"
              extract:
                json_path: 'v'
        """
    )
    cfg = _write(tmp_path / "wow.yaml", bad)
    code = _run(tmp_path, cfg)
    assert code == wow.EXIT_ERR

    with pytest.raises(ValueError, match="inline value not allowed"):
        wow.evaluate_metric(
            tmp_path,
            {
                "name": "smuggled",
                "direction": "higher_better",
                "before": {
                    "file": "baseline/s.json",
                    "extract": {"json_path": "v"},
                    "value": 99,
                },
                "after": {
                    "file": "current/s.json",
                    "extract": {"json_path": "v"},
                },
            },
            0.0,
        )
