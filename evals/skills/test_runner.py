import json
from datetime import datetime, timezone
from pathlib import Path

from evals.skills.runner import calculate_metrics


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
                    "last_used_at": "2026-08-20T00:00:00+00:00",
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
                    "last_used_at": "2026-09-20T00:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = calculate_metrics(skills, window_days=30, as_of=AS_OF)

    assert metrics.mature_skills == 1
    assert metrics.observed_triggered_skills == 0
    assert any("latest-use telemetry" in warning for warning in metrics.warnings)
