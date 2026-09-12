"""Contract tests for the hillclimb skill: the two helper CLIs and the skill's metadata.

The scripts are exercised as subprocesses through their real command-line
interface, which is the artifact users actually run. Every test pins behaviour
(exit codes, computed values, emitted files) rather than source text.

Run: scripts/run_tests.sh tests/skills/test_hillclimb_skill.py -q
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "software-development" / "hillclimb"
DECISION_LOG = SKILL_DIR / "scripts" / "decision_log.py"
SAMPLE_METRIC = SKILL_DIR / "scripts" / "sample_metric.py"
PY = sys.executable
COLUMNS = [
    "id",
    "timestamp",
    "hypothesis",
    "change",
    "before",
    "after",
    "delta",
    "tests",
    "verdict",
    "harness_id",
    "note",
]
HEADER = "\t".join(COLUMNS)


def run(script, *args, cwd=None):
    return subprocess.run(
        [PY, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )


def append(log_dir, before, after, verdict="kept", tests="pass", hypothesis="h", note=""):
    return run(
        DECISION_LOG,
        "append",
        "--dir",
        str(log_dir),
        "--hypothesis",
        hypothesis,
        "--change",
        "c",
        "--before",
        str(before),
        "--after",
        str(after),
        "--tests",
        tests,
        "--verdict",
        verdict,
        "--note",
        note,
    )


def tsv_rows(log_dir):
    text = (log_dir / "decision.tsv").read_text(encoding="utf-8")
    lines = [line for line in text.split("\n") if line]
    assert lines[0] == HEADER
    return [line.split("\t") for line in lines[1:]]


def write_harness(path, body):
    path.write_text(body, encoding="utf-8")
    return f"{PY} {path}"


# --------------------------------------------------------------------------- #
# decision_log.py
# --------------------------------------------------------------------------- #


def test_append_creates_dir_and_roundtrips(tmp_path):
    """No --dir: the log directory is created lazily, with a valid header."""
    result = append(tmp_path / ".hillclimb", 5.8, 5.2)
    assert result.returncode == 0, result.stderr
    rows = tsv_rows(tmp_path / ".hillclimb")
    assert len(rows) == 1
    assert len(rows[0]) == 11
    assert rows[0][0] == "1"
    assert rows[0][4] == "5.8"
    assert rows[0][5] == "5.2"


def test_append_computes_delta_and_ignores_any_caller_value(tmp_path):
    log_dir = tmp_path / "log"
    assert append(log_dir, 5.8, 5.2).returncode == 0
    assert tsv_rows(log_dir)[0][6] == "-0.6"


def test_append_accepts_na_and_records_na_delta(tmp_path):
    log_dir = tmp_path / "log"
    result = run(
        DECISION_LOG,
        "append",
        "--dir",
        str(log_dir),
        "--hypothesis",
        "pivot",
        "--change",
        "none",
        "--before",
        "na",
        "--after",
        "na",
        "--tests",
        "none",
        "--verdict",
        "reverted",
    )
    assert result.returncode == 0, result.stderr
    row = tsv_rows(log_dir)[0]
    assert row[6] == "na"
    assert run(DECISION_LOG, "verify", "--dir", str(log_dir)).returncode == 0


def test_append_sanitizes_tabs_and_newlines_in_free_text(tmp_path):
    """A tab inside the note must not shift the row into more columns."""
    log_dir = tmp_path / "log"
    result = append(log_dir, 1.0, 0.5, note="tab\there\nand newline")
    assert result.returncode == 0, result.stderr
    rows = tsv_rows(log_dir)
    assert len(rows) == 1
    assert len(rows[0]) == 11
    assert "\t" not in rows[0][10]
    assert "\n" not in rows[0][10]


def test_append_rejects_invalid_enum_and_writes_nothing(tmp_path):
    log_dir = tmp_path / "log"
    result = append(log_dir, 1.0, 0.5, tests="bogus")
    assert result.returncode != 0
    assert not (log_dir / "decision.tsv").exists()


def test_append_rejects_non_numeric_metric(tmp_path):
    log_dir = tmp_path / "log"
    result = append(log_dir, "fast", 0.5)
    assert result.returncode == 2
    assert not (log_dir / "decision.tsv").exists()


def test_ids_increment_monotonically(tmp_path):
    log_dir = tmp_path / "log"
    for index in range(3):
        assert append(log_dir, 5.0, 5.0 - index).returncode == 0
    assert [row[0] for row in tsv_rows(log_dir)] == ["1", "2", "3"]
    assert run(DECISION_LOG, "verify", "--dir", str(log_dir)).returncode == 0


def test_verify_reports_delta_mismatch(tmp_path):
    log_dir = tmp_path / "log"
    assert append(log_dir, 5.0, 4.0).returncode == 0
    tsv = log_dir / "decision.tsv"
    rows = tsv_rows(log_dir)
    rows[0][6] = "-99"
    tsv.write_text(HEADER + "\n" + "\t".join(rows[0]) + "\n", encoding="utf-8")
    result = run(DECISION_LOG, "verify", "--dir", str(log_dir))
    assert result.returncode == 1
    assert "delta-mismatch" in result.stdout


def test_verify_reports_stale_harness(tmp_path):
    log_dir = tmp_path / "log"
    assert append(log_dir, 5.0, 4.0).returncode == 0
    (log_dir / "baseline.json").write_text(
        json.dumps({"harness_id": "ffffffffffff", "median": 5.0}), encoding="utf-8"
    )
    result = run(DECISION_LOG, "verify", "--dir", str(log_dir))
    assert result.returncode == 1
    assert "stale-harness" in result.stdout


def test_verify_missing_log_reports_no_log_and_exits_0(tmp_path):
    result = run(DECISION_LOG, "verify", "--dir", str(tmp_path / "absent"))
    assert result.returncode == 0
    assert "no-log" in result.stdout


def test_read_paths_on_absent_state_exit_0(tmp_path):
    absent = tmp_path / "absent"
    assert run(DECISION_LOG, "list", "--dir", str(absent)).returncode == 0
    stats = run(DECISION_LOG, "stats", "--dir", str(absent), "--json")
    assert stats.returncode == 0
    payload = json.loads(stats.stdout)
    assert payload["attempts"] == 0
    assert payload["plateau"] is False


def test_dir_pointing_at_a_file_exits_2(tmp_path):
    target = tmp_path / "afile"
    target.write_text("not a directory", encoding="utf-8")
    result = append(target, 1.0, 2.0)
    assert result.returncode == 2
    assert "not a directory" in result.stderr


def test_stats_flags_a_plateau_of_sub_threshold_attempts(tmp_path):
    log_dir = tmp_path / "log"
    assert append(log_dir, 10.0, 9.0).returncode == 0  # a real win, 10%
    for before, after in ((9.0, 8.95), (8.95, 8.91), (8.91, 8.88)):
        assert append(log_dir, before, after).returncode == 0
    payload = json.loads(run(DECISION_LOG, "stats", "--dir", str(log_dir), "--json").stdout)
    assert payload["plateau"] is True
    assert len(payload["plateau_reasons"]) == 3


def test_stats_does_not_flag_plateau_while_a_recent_attempt_still_wins(tmp_path):
    log_dir = tmp_path / "log"
    for before, after in ((9.0, 8.95), (8.95, 8.90), (8.90, 8.0)):
        assert append(log_dir, before, after).returncode == 0
    payload = json.loads(run(DECISION_LOG, "stats", "--dir", str(log_dir), "--json").stdout)
    assert payload["plateau"] is False


def test_stats_needs_a_full_window_before_claiming_a_plateau(tmp_path):
    log_dir = tmp_path / "log"
    assert append(log_dir, 9.0, 8.99).returncode == 0
    payload = json.loads(run(DECISION_LOG, "stats", "--dir", str(log_dir), "--json").stdout)
    assert payload["plateau"] is False


def test_stats_is_direction_aware(tmp_path):
    log_dir = tmp_path / "log"
    (log_dir).mkdir(parents=True)
    (log_dir / "baseline.json").write_text(
        json.dumps({"harness_id": "abc123abc123", "median": 50.0, "direction": "maximize"}),
        encoding="utf-8",
    )
    assert append(log_dir, 50.0, 60.0).returncode == 0  # a win when maximizing
    payload = json.loads(run(DECISION_LOG, "stats", "--dir", str(log_dir), "--json").stdout)
    assert payload["direction"] == "maximize"
    assert payload["best_improvement"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# sample_metric.py
# --------------------------------------------------------------------------- #


def test_run_reports_median_min_max_spread(tmp_path):
    harness = write_harness(
        tmp_path / "varying.py",
        "import pathlib, sys\n"
        "state = pathlib.Path(__file__).with_suffix('.state')\n"
        "n = int(state.read_text()) if state.exists() else 0\n"
        "state.write_text(str(n + 1))\n"
        "print([1.0, 2.0, 6.0][n % 3])\n",
    )
    result = run(SAMPLE_METRIC, "run", "--harness", harness, "--samples", "3", "--dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["samples"] == 3
    assert payload["median"] == 2.0
    assert payload["min"] == 1.0
    assert payload["max"] == 6.0
    assert payload["spread"] == 5.0


def test_run_emits_no_metric_when_a_sample_fails(tmp_path):
    harness = write_harness(tmp_path / "bad.py", "import sys\nsys.exit(3)\n")
    result = run(SAMPLE_METRIC, "run", "--harness", harness, "--samples", "2", "--dir", str(tmp_path))
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["error"] == "sample failed; no metric emitted"
    assert payload["failed_samples"] == 2
    assert "median" not in payload


def test_run_emits_no_metric_when_no_number_is_printed(tmp_path):
    harness = write_harness(tmp_path / "wordy.py", "print('no metric here')\n")
    result = run(SAMPLE_METRIC, "run", "--harness", harness, "--samples", "1", "--dir", str(tmp_path))
    assert result.returncode == 2
    assert "extract_error" in result.stdout


def test_baseline_freezes_and_compare_scores_both_ways(tmp_path):
    baseline_harness = write_harness(tmp_path / "slow.py", "print('runtime 8.0 s')\n")
    slow = run(
        SAMPLE_METRIC,
        "baseline",
        "--harness",
        baseline_harness,
        "--samples",
        "2",
        "--name",
        "runtime",
        "--unit",
        "s",
        "--extract",
        "regex:runtime (\\d+\\.\\d+)",
        "--dir",
        str(tmp_path),
    )
    assert slow.returncode == 0, slow.stderr
    frozen = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert frozen["frozen"] is True
    assert frozen["median"] == 8.0
    assert frozen["samples"] == 2

    same = run(
        SAMPLE_METRIC,
        "compare",
        "--harness",
        baseline_harness,
        "--samples",
        "2",
        "--extract",
        "regex:runtime (\\d+\\.\\d+)",
        "--dir",
        str(tmp_path),
    )
    assert same.returncode == 1  # no improvement
    assert json.loads(same.stdout)["improved"] is False


def test_compare_refuses_a_changed_harness(tmp_path):
    original = write_harness(tmp_path / "a.py", "print('runtime 8.0 s')\n")
    changed = write_harness(tmp_path / "b.py", "print('runtime 4.0 s')\n")
    spec = "regex:runtime (\\d+\\.\\d+)"
    assert (
        run(
            SAMPLE_METRIC,
            "baseline",
            "--harness",
            original,
            "--samples",
            "1",
            "--extract",
            spec,
            "--dir",
            str(tmp_path),
        ).returncode
        == 0
    )
    result = run(
        SAMPLE_METRIC,
        "compare",
        "--harness",
        changed,
        "--samples",
        "1",
        "--extract",
        spec,
        "--dir",
        str(tmp_path),
    )
    assert result.returncode == 3
    assert "harness-id mismatch" in result.stdout


def test_compare_accepts_an_explicit_value_and_scores_improvement(tmp_path):
    harness = write_harness(tmp_path / "a.py", "print('runtime 8.0 s')\n")
    assert (
        run(
            SAMPLE_METRIC,
            "baseline",
            "--harness",
            harness,
            "--samples",
            "1",
            "--extract",
            "regex:runtime (\\d+\\.\\d+)",
            "--dir",
            str(tmp_path),
        ).returncode
        == 0
    )
    result = run(SAMPLE_METRIC, "compare", "--value", "4.0", "--dir", str(tmp_path))
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["improved"] is True
    assert payload["delta"] == pytest.approx(-4.0)


def test_compare_without_a_baseline_exits_2(tmp_path):
    harness = write_harness(tmp_path / "a.py", "print(1)\n")
    result = run(SAMPLE_METRIC, "compare", "--harness", harness, "--dir", str(tmp_path / "empty"))
    assert result.returncode == 2
    assert "baseline" in result.stderr


@pytest.mark.parametrize(
    "spec,body,expected",
    [
        ("auto", "print('plain 99 tokens')\n", 99.0),
        ("regex:value=(\\d+)", "print('value=42')\n", 42.0),
        ("json:results.median", "import json\nprint(json.dumps({'results': {'median': 7.5}}))\n", 7.5),
        ("line:Total runtime:", "print('Total runtime: 3.25 s')\n", 3.25),
    ],
)
def test_extract_modes(tmp_path, spec, body, expected):
    harness = write_harness(tmp_path / "h.py", body)
    result = run(
        SAMPLE_METRIC,
        "run",
        "--harness",
        harness,
        "--samples",
        "1",
        "--extract",
        spec,
        "--dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["median"] == expected


# --------------------------------------------------------------------------- #
# The skill's own metadata contract
# --------------------------------------------------------------------------- #


def skill_frontmatter():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---")
    closing = re.search(r"\n---\s*\n", text[3:])
    assert closing, "SKILL.md frontmatter is not closed"
    return yaml.safe_load(text[3:][: closing.start()])


def test_skill_metadata_satisfies_the_authoring_standards():
    frontmatter = skill_frontmatter()
    assert frontmatter["name"] == SKILL_DIR.name
    description = frontmatter["description"]
    assert len(description) <= 60, f"description is {len(description)} chars"
    assert description.rstrip().endswith(".")
    assert not re.search(
        r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
        description,
        re.I,
    )
    for field in ("version", "author", "license", "platforms", "metadata"):
        assert field in frontmatter
    assert frontmatter["metadata"]["hermes"]["tags"]


def test_skill_related_skills_all_exist():
    frontmatter = skill_frontmatter()
    known = {path.parent.name for path in REPO.glob("skills/**/SKILL.md")}
    known |= {path.parent.name for path in REPO.glob("optional-skills/**/SKILL.md")}
    dangling = [name for name in frontmatter["metadata"]["hermes"]["related_skills"] if name not in known]
    assert not dangling, f"related_skills do not resolve: {dangling}"


def test_skill_documents_only_files_it_ships():
    for relative in ("scripts/decision_log.py", "scripts/sample_metric.py"):
        assert (SKILL_DIR / relative).is_file(), f"missing {relative}"
    references = {path.name for path in (SKILL_DIR / "references").glob("*.md")}
    assert references == {
        "decision-log.md",
        "harness-lifecycle.md",
        "plateau-playbook.md",
        "unattended-mode.md",
    }


# --------------------------------------------------------------------------- #
# Regression tests. Every case below was found by cross-vendor review of the
# scripts rather than by the original implementation.
# --------------------------------------------------------------------------- #


def test_extract_regex_uses_the_first_capture_group(tmp_path):
    harness = write_harness(tmp_path / "two.py", "print('a=1 b=2')\n")
    result = run(
        SAMPLE_METRIC,
        "run",
        "--harness",
        harness,
        "--samples",
        "1",
        "--extract",
        r"regex:a=(\d+) b=(\d+)",
        "--dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["median"] == 1.0


def test_compare_reuses_the_extraction_spec_recorded_in_the_baseline(tmp_path):
    """`compare` used to silently fall back to `auto` and measure a different number."""
    harness = write_harness(
        tmp_path / "two.py", "print('Total runtime: 5.0 s')\nprint('checksum 999')\n"
    )
    spec = "line:Total runtime:"
    baseline = run(
        SAMPLE_METRIC,
        "baseline",
        "--harness",
        harness,
        "--samples",
        "1",
        "--extract",
        spec,
        "--dir",
        str(tmp_path),
    )
    assert baseline.returncode == 0, baseline.stderr
    frozen = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert frozen["extract"] == spec
    assert frozen["median"] == 5.0

    # No --extract here: `auto` would read 999 and report a huge win.
    result = run(
        SAMPLE_METRIC, "compare", "--harness", harness, "--samples", "1", "--dir", str(tmp_path)
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["before"] == 5.0
    assert payload["after"] == 5.0


def test_samples_must_be_at_least_one(tmp_path):
    harness = write_harness(tmp_path / "a.py", "print(1)\n")
    result = run(
        SAMPLE_METRIC, "run", "--harness", harness, "--samples", "0", "--dir", str(tmp_path)
    )
    assert result.returncode == 2
    assert "at least 1" in result.stderr


def test_list_does_not_crash_on_a_malformed_row(tmp_path):
    log_dir = tmp_path / "log"
    assert append(log_dir, 5.0, 4.0).returncode == 0
    good = tsv_rows(log_dir)[0]
    (log_dir / "decision.tsv").write_text(
        HEADER + "\n" + "\t".join(good) + "\n" + "7\tbroken\n", encoding="utf-8"
    )
    result = run(DECISION_LOG, "list", "--dir", str(log_dir))
    assert result.returncode == 0, result.stderr
    assert good[0] in result.stdout
    # the malformed row is skipped from the table but never hidden silently
    assert "malformed" in result.stderr
    assert run(DECISION_LOG, "verify", "--dir", str(log_dir)).returncode == 1


def test_stats_rejects_a_window_below_one(tmp_path):
    log_dir = tmp_path / "log"
    assert append(log_dir, 5.0, 4.0).returncode == 0
    assert run(DECISION_LOG, "stats", "--dir", str(log_dir), "--window", "0").returncode == 2


def test_plateau_reasons_are_empty_when_there_is_no_plateau(tmp_path):
    log_dir = tmp_path / "log"
    for before, after in ((10.0, 9.0), (9.0, 8.99), (8.99, 8.98)):
        assert append(log_dir, before, after).returncode == 0
    payload = json.loads(run(DECISION_LOG, "stats", "--dir", str(log_dir), "--json").stdout)
    assert payload["plateau"] is False
    assert payload["plateau_reasons"] == []


def test_stats_window_is_taken_from_the_window_rows(tmp_path):
    """Regression: the window average used to be sliced from a filtered list."""
    log_dir = tmp_path / "log"
    assert append(log_dir, 10.0, 9.0).returncode == 0  # a big win, outside the window
    assert (
        run(
            DECISION_LOG,
            "append",
            "--dir",
            str(log_dir),
            "--hypothesis",
            "pivot",
            "--change",
            "none",
            "--before",
            "na",
            "--after",
            "na",
            "--tests",
            "none",
            "--verdict",
            "reverted",
        ).returncode
        == 0
    )
    assert append(log_dir, 9.0, 8.95).returncode == 0
    assert append(log_dir, 8.95, 8.91).returncode == 0
    payload = json.loads(run(DECISION_LOG, "stats", "--dir", str(log_dir), "--json").stdout)
    # mean of the two measurable improvements inside the window: (0.05 + 0.04) / 2
    assert payload["mean_improvement_last_window"] == pytest.approx(0.045)
    # an unmeasurable attempt is not evidence of movement, so it counts as a stall
    assert payload["plateau"] is True


def test_worst_regression_is_none_when_nothing_regressed(tmp_path):
    log_dir = tmp_path / "log"
    assert append(log_dir, 10.0, 9.0).returncode == 0
    assert append(log_dir, 9.0, 8.0).returncode == 0
    payload = json.loads(run(DECISION_LOG, "stats", "--dir", str(log_dir), "--json").stdout)
    assert payload["best_improvement"] == pytest.approx(1.0)
    assert payload["worst_regression"] is None


def test_worst_regression_is_negative_when_a_metric_regressed(tmp_path):
    log_dir = tmp_path / "log"
    assert append(log_dir, 10.0, 11.5).returncode == 0
    payload = json.loads(run(DECISION_LOG, "stats", "--dir", str(log_dir), "--json").stdout)
    assert payload["best_improvement"] is None
    assert payload["worst_regression"] == pytest.approx(-1.5)


def test_verify_flags_an_unparseable_metric_field(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True)
    row = ["1", "2026-09-10T00:00:00Z", "h", "c", "fast", "na", "na", "none", "reverted", "", "note"]
    assert len(row) == 11
    (log_dir / "decision.tsv").write_text(HEADER + "\n" + "\t".join(row) + "\n", encoding="utf-8")
    result = run(DECISION_LOG, "verify", "--dir", str(log_dir))
    assert result.returncode == 1
    assert "invalid-metric" in result.stdout


def test_verify_accepts_a_crlf_log(tmp_path):
    """A Windows-written log uses CRLF; the header must still match."""
    log_dir = tmp_path / "log"
    assert append(log_dir, 5.0, 4.0).returncode == 0
    text = (log_dir / "decision.tsv").read_text(encoding="utf-8")
    (log_dir / "decision.tsv").write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    verify = run(DECISION_LOG, "verify", "--dir", str(log_dir))
    assert verify.returncode == 0
    assert json.loads(verify.stdout)["problems"] == []
    listed = run(DECISION_LOG, "list", "--dir", str(log_dir), "--json")
    assert json.loads(listed.stdout)[0]["id"] == "1"


@pytest.mark.parametrize("command", ["list", "stats", "verify"])
def test_read_paths_reject_a_file_used_as_dir(tmp_path, command):
    target = tmp_path / "afile"
    target.write_text("not a directory", encoding="utf-8")
    result = run(DECISION_LOG, command, "--dir", str(target))
    assert result.returncode == 2
    assert "not a directory" in result.stderr
