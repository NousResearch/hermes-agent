"""Model-sovereign curator classification and reporting contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import importlib
    import hermes_constants

    importlib.reload(hermes_constants)
    from agent import curator

    importlib.reload(curator)
    yield curator


def _delete(name: str, absorbed_into: str) -> dict:
    return {
        "name": "skill_manage",
        "arguments": json.dumps(
            {
                "action": "delete",
                "name": name,
                "absorbed_into": absorbed_into,
            }
        ),
    }


def test_removed_skill_projection_uses_explicit_delete_declarations(curator_env):
    result = curator_env._classify_removed_skills(
        removed=["narrow", "stale"],
        added=["umbrella"],
        after_names={"umbrella"},
        tool_calls=[_delete("narrow", "umbrella"), _delete("stale", "")],
    )

    assert result["consolidated"] == [
        {
            "name": "narrow",
            "into": "umbrella",
            "source": "absorbed_into (model-declared at delete)",
        }
    ]
    assert result["pruned"][0]["name"] == "stale"
    assert result["unclassified"] == []


@pytest.mark.parametrize(
    "authored_text",
    [
        "narrow",
        "references/narrow.md",
        "Absorbed narrow into umbrella",
        "error: narrow failed then recovered",
        "running latest tests with pytest",
    ],
)
def test_arbitrary_skill_text_never_classifies_removal(curator_env, authored_text):
    result = curator_env._classify_removed_skills(
        removed=["narrow"],
        added=["umbrella"],
        after_names={"umbrella"},
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps(
                    {
                        "action": "write_file",
                        "name": "umbrella",
                        "file_path": authored_text,
                        "file_content": authored_text,
                        "content": authored_text,
                    }
                ),
            }
        ],
    )

    assert result["consolidated"] == []
    assert result["pruned"] == []
    assert result["unclassified"] == [
        {"name": "narrow", "source": "missing_model_delete_declaration"}
    ]


def test_invalid_declared_destination_is_unclassified(curator_env):
    result = curator_env._classify_removed_skills(
        removed=["narrow"],
        added=[],
        after_names={"real-umbrella"},
        tool_calls=[_delete("narrow", "ghost-umbrella")],
    )

    assert result["consolidated"] == []
    assert result["pruned"] == []
    assert result["unclassified"][0]["model_claimed_into"] == "ghost-umbrella"


def test_parse_structured_summary_happy_path(curator_env):
    parsed = curator_env._parse_structured_summary(
        """Done.\n```yaml
consolidations:
  - from: narrow
    into: umbrella
    reason: duplicate
prunings:
  - name: stale
    reason: obsolete
```"""
    )

    assert parsed["consolidations"] == [
        {"from": "narrow", "into": "umbrella", "reason": "duplicate"}
    ]
    assert parsed["prunings"] == [{"name": "stale", "reason": "obsolete"}]


@pytest.mark.parametrize("text", ["", "no block", "```yaml\nnot: [valid\n```"])
def test_parse_structured_summary_malformed_or_absent_is_empty(curator_env, text):
    assert curator_env._parse_structured_summary(text) == {
        "consolidations": [],
        "prunings": [],
    }


def test_extract_delete_declarations_is_schema_driven(curator_env):
    declarations = curator_env._extract_absorbed_into_declarations(
        [
            _delete("narrow", "umbrella"),
            _delete("stale", ""),
            {
                "name": "skill_manage",
                "arguments": json.dumps(
                    {
                        "action": "patch",
                        "name": "other",
                        "absorbed_into": "must-not-count",
                    }
                ),
            },
            {"name": "skill_manage", "arguments": "{bad json"},
        ]
    )

    assert declarations == {
        "narrow": {"into": "umbrella", "declared": True},
        "stale": {"into": "", "declared": True},
    }


def test_reconcile_uses_model_authored_structured_block(curator_env):
    result = curator_env._reconcile_classification(
        removed=["narrow", "stale", "unknown"],
        model_block={
            "consolidations": [
                {"from": "narrow", "into": "umbrella", "reason": "duplicate"}
            ],
            "prunings": [{"name": "stale", "reason": "obsolete"}],
        },
        destinations={"umbrella"},
    )

    assert [item["name"] for item in result["consolidated"]] == ["narrow"]
    assert [item["name"] for item in result["pruned"]] == ["stale"]
    assert result["unclassified"] == [
        {"name": "unknown", "source": "missing_model_authored_classification"}
    ]


def test_delete_declaration_is_authoritative_and_preserves_model_reason(curator_env):
    result = curator_env._reconcile_classification(
        removed=["narrow", "stale"],
        model_block={
            "consolidations": [
                {"from": "narrow", "into": "umbrella", "reason": "duplicate"}
            ],
            "prunings": [{"name": "stale", "reason": "obsolete"}],
        },
        destinations={"umbrella"},
        absorbed_declarations={
            "narrow": {"into": "umbrella", "declared": True},
            "stale": {"into": "", "declared": True},
        },
    )

    assert result["consolidated"][0]["reason"] == "duplicate"
    assert result["pruned"][0]["reason"] == "obsolete"
    assert result["unclassified"] == []


def test_invalid_model_destination_remains_unclassified(curator_env):
    result = curator_env._reconcile_classification(
        removed=["narrow"],
        model_block={"consolidations": [], "prunings": []},
        destinations={"real-umbrella"},
        absorbed_declarations={
            "narrow": {"into": "ghost-umbrella", "declared": True}
        },
    )

    assert result["consolidated"] == []
    assert result["pruned"] == []
    assert result["unclassified"][0]["source"] == "invalid_model_delete_destination"


def test_report_splits_explicit_model_decisions(curator_env):
    before = [
        {"name": "narrow", "state": "active", "pinned": False},
        {"name": "stale", "state": "stale", "pinned": False},
    ]
    after = [{"name": "umbrella", "state": "active", "pinned": False}]
    final = """Done.\n```yaml
consolidations:
  - from: narrow
    into: umbrella
    reason: duplicate
prunings:
  - name: stale
    reason: obsolete
```"""

    run_dir = curator_env._write_run_report(
        started_at=datetime.now(timezone.utc),
        elapsed_seconds=1,
        auto_counts={"checked": 2, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="none",
        before_report=before,
        before_names={"narrow", "stale"},
        after_report=after,
        llm_meta={
            "final": final,
            "summary": "done",
            "model": "m",
            "provider": "p",
            "error": None,
            "tool_calls": [_delete("narrow", "umbrella"), _delete("stale", "")],
        },
    )

    payload = json.loads((run_dir / "run.json").read_text())
    assert [item["name"] for item in payload["consolidated"]] == ["narrow"]
    assert [item["name"] for item in payload["pruned"]] == ["stale"]
    assert payload["unclassified"] == []
    report = (run_dir / "REPORT.md").read_text()
    assert "Consolidated into umbrella skills" in report
    assert "Pruned — archived for staleness" in report


def test_report_exposes_missing_model_classification_without_guessing(curator_env):
    run_dir = curator_env._write_run_report(
        started_at=datetime.now(timezone.utc),
        elapsed_seconds=1,
        auto_counts={"checked": 1, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="none",
        before_report=[{"name": "narrow", "state": "active", "pinned": False}],
        before_names={"narrow"},
        after_report=[{"name": "umbrella", "state": "active", "pinned": False}],
        llm_meta={
            "final": "The authored prose says narrow was absorbed into umbrella.",
            "summary": "done",
            "model": "m",
            "provider": "p",
            "error": None,
            "tool_calls": [
                {
                    "name": "skill_manage",
                    "arguments": json.dumps(
                        {
                            "action": "write_file",
                            "name": "umbrella",
                            "content": "narrow was absorbed into umbrella",
                        }
                    ),
                }
            ],
        },
    )

    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["consolidated"] == []
    assert payload["pruned"] == []
    assert payload["unclassified"][0]["name"] == "narrow"
    report = (run_dir / "REPORT.md").read_text()
    assert "model classification unavailable" in report
    assert "No keyword or content heuristic was used" in report


def test_rename_summary_uses_only_explicit_model_decisions(curator_env):
    summary = curator_env._build_rename_summary(
        before_names={"narrow", "stale"},
        after_report=[{"name": "umbrella", "state": "active"}],
        tool_calls=[_delete("narrow", "umbrella"), _delete("stale", "")],
        model_final="",
    )

    assert "narrow → umbrella" in summary
    assert "stale — pruned" in summary


def test_rename_summary_marks_opaque_prose_unclassified(curator_env):
    summary = curator_env._build_rename_summary(
        before_names={"narrow"},
        after_report=[{"name": "umbrella", "state": "active"}],
        tool_calls=[
            {
                "name": "skill_manage",
                "arguments": json.dumps(
                    {
                        "action": "patch",
                        "name": "umbrella",
                        "new_string": "narrow was consolidated here",
                    }
                ),
            }
        ],
        model_final="narrow was consolidated into umbrella",
    )

    assert "archived (model classification unavailable)" in summary
    assert "narrow → umbrella" not in summary
