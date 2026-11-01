"""Skill evaluation windows and report output contracts (review follow-up for #122544)."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals.skills.runner import calculate_metrics, main


AS_OF = datetime(2026, 9, 25, tzinfo=timezone.utc)


def _write_skill(root: Path, name: str, category: str) -> None:
    path = root / category / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use {name}.\ncategory: {category}\n---\n\n# Skill\n",
        encoding="utf-8",
    )


def test_arm_a_reports_maturity_trigger_precision_duplicates_and_sessions(tmp_path):
    skills = tmp_path / "skills"
    _write_skill(skills, "browser-one", "browser")
    _write_skill(skills, "browser-two", "browser")
    _write_skill(skills, "database", "data")
    _write_skill(skills, "bundled", "browser")
    (skills / ".bundled_manifest").write_text("bundled:abc\n", encoding="utf-8")
    (skills / ".usage.json").write_text(
        json.dumps(
            {
                "browser-one": {
                    "created_by": "agent",
                    "created_at": "2026-08-01T00:00:00+00:00",
                    "use_count": 1,
                    "last_used_at": "2026-09-10T00:00:00+00:00",
                },
                "browser-two": {
                    "created_by": "agent",
                    "created_at": "2026-08-01T00:00:00+00:00",
                    "use_count": 0,
                },
                "database": {
                    "created_by": "agent",
                    "created_at": "2026-09-01T00:00:00+00:00",
                    "use_count": 1,
                    "last_used_at": "2026-09-10T00:00:00+00:00",
                },
                "bundled": {
                    "created_by": "agent",
                    "created_at": "2026-08-01T00:00:00+00:00",
                    "use_count": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    (skills / ".curator_ledger.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "action": "create",
                    "ts": "2026-09-01T00:00:00Z",
                    "evidence": {"session_id": session},
                }
            )
            for session in ("s-1", "s-1", "s-2")
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = calculate_metrics(skills, window_days=30, as_of=AS_OF)

    assert metrics.agent_created_skills == 3
    assert metrics.mature_skills == 2
    assert metrics.observed_triggered_skills == 1
    assert metrics.trigger_precision == 0.5
    assert metrics.duplicate_class_groups == 1
    assert metrics.duplicate_class_skills == 2
    assert metrics.duplicate_class_rate == 2 / 3
    assert metrics.creation_events == 3
    assert metrics.observed_sessions == 2
    assert metrics.creation_rate_per_session == 1.5


def test_trigger_precision_is_conservative_when_latest_use_is_after_window(tmp_path):
    skills = tmp_path / "skills"
    _write_skill(skills, "late", "data")
    (skills / ".usage.json").write_text(
        json.dumps(
            {
                "late": {
                    "created_by": "agent",
                    "created_at": "2026-08-01T00:00:00+00:00",
                    "use_count": 2,
                    "last_used_at": "2026-09-30T00:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = calculate_metrics(skills, window_days=30, as_of=AS_OF)

    assert metrics.mature_skills == 1
    assert metrics.observed_triggered_skills == 0
    assert any("latest-use telemetry" in warning for warning in metrics.warnings)


@pytest.mark.parametrize("created,last_used,use_count,expected", [
    ("2026-06-01", "2026-06-10", 1, 0),
    ("2026-08-01", "2026-08-26", 1, 1),
    ("2026-08-01", "2026-09-10", 1, 1),
    ("2026-08-01", "2026-09-25", 1, 1),
    ("2026-08-01", "2026-09-26", 1, 0),
    ("2026-08-01", "2026-09-10", 0, 0),
    ("2026-09-01", "2026-09-10", 1, 0),
])
def test_review_metrics_share_the_report_window(tmp_path, created, last_used, use_count, expected):
    skills = tmp_path / "skills"
    _write_skill(skills, "observed", "browser")
    (skills / ".usage.json").write_text(json.dumps({"observed": {
        "created_by": "agent", "created_at": created,
        "last_used_at": last_used, "use_count": use_count,
    }}))
    rows = [
        {"action": "create", "ts": stamp, "evidence": {"session_id": session}}
        for stamp, session in [
            ("2026-06-10", "old"), ("2026-08-26", "boundary"),
            ("2026-09-01", "one"), ("2026-09-25", "two"),
            ("2026-09-26", "future"), ("invalid", "undated"),
        ]
    ]
    (skills / ".curator_ledger.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\nnot-json\n")
    metrics = calculate_metrics(skills, window_days=30, as_of=AS_OF)
    assert metrics.observed_triggered_skills == expected
    assert metrics.skills[0]["observed_trigger_within_window"] is bool(expected)
    assert metrics.creation_events == 3
    assert metrics.observed_sessions == 3
    assert metrics.creation_rate_per_session == 1
    assert any("timestamp" in warning for warning in metrics.warnings)


@pytest.mark.parametrize("stamp,valid", [
    ("2026-09-25T00:00:00Z", True), (None, True),
    ("2026-13-45", False), ("tomorrow", False),
])
def test_review_cli_reports_are_parseable_and_invalid_windows_fail(tmp_path, capsys, stamp, valid):
    output = tmp_path / "report.json"
    argv = ["--home", str(tmp_path), "--output", str(output)]
    if stamp is not None:
        argv += ["--as-of", stamp]
    if not valid:
        with pytest.raises(SystemExit) as error:
            main(argv)
        assert error.value.code == 2
        assert not output.exists()
        assert "--as-of" in capsys.readouterr().err
        return
    assert main(argv) == 0
    payload = output.read_text()
    report = json.loads(payload)
    assert payload.endswith("\n")
    assert report["window_days"] == 30
    if stamp:
        assert report["as_of"] == AS_OF.isoformat()
    assert main(["--home", str(tmp_path), "--as-of", AS_OF.isoformat()]) == 0
    assert json.loads(capsys.readouterr().out)["as_of"] == AS_OF.isoformat()

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    default = tmp_path / "default"
    named = tmp_path / "profiles" / "work"
    for home in (default, named, default):
        token = set_hermes_home_override(home)
        try:
            assert main(["--as-of", AS_OF.isoformat()]) == 0
            assert json.loads(capsys.readouterr().out)["skills_root"] == str(home / "skills")
        finally:
            reset_hermes_home_override(token)
