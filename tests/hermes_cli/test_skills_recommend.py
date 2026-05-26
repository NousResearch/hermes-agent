import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console


def test_recommend_skills_invokes_manual_wrapper_without_loading_or_logging(tmp_path):
    from hermes_cli.skill_recommend import recommend_skills

    calls = []

    def fake_run(cmd, text, capture_output, check):
        calls.append(cmd)
        payload = {
            "phase": "2H",
            "query": "review this pull request",
            "scope": {
                "manual_only": True,
                "loads_skills": False,
                "runtime_coupling": False,
                "db_writes": False,
            },
            "recommendations": [
                {
                    "rank": 1,
                    "skill_name": "github-code-review",
                    "relative_path": "github/github-code-review/SKILL.md",
                    "score": 0.188875,
                }
            ],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    wrapper = tmp_path / "pgvector_recommended_skills.py"
    wrapper.write_text("#!/usr/bin/env python3\n")

    result = recommend_skills(
        "review this pull request",
        top_k=3,
        min_score=0.04,
        wrapper_path=wrapper,
        runner=fake_run,
    )

    assert result["recommendations"][0]["skill_name"] == "github-code-review"
    assert result["scope"]["loads_skills"] is False
    assert result["scope"]["runtime_coupling"] is False
    assert "--log-event" not in calls[0]
    assert calls[0] == [str(wrapper), "review this pull request", "--top-k", "3", "--min-score", "0.04", "--json"]


def test_render_recommendations_is_advisory_and_human_readable():
    from hermes_cli.skill_recommend import render_recommendations

    output = render_recommendations(
        {
            "query": "review this pull request",
            "recommendations": [
                {
                    "rank": 1,
                    "skill_name": "github-code-review",
                    "relative_path": "github/github-code-review/SKILL.md",
                    "score": 0.188875,
                }
            ],
            "diagnostics": {"min_score": 0.04},
        },
        show_scores=True,
    )

    assert "Recommended skills:" in output
    assert "github-code-review" in output
    assert "github/github-code-review/SKILL.md" in output
    assert "score 0.188875" in output
    assert "Advisory only" in output
    assert "No skills were loaded" in output


def test_skills_command_recommend_routes_to_read_only_advisory_output(monkeypatch):
    import hermes_cli.skills_hub as skills_hub

    def fake_recommend_skills(query, top_k=3, min_score=0.04, wrapper_path=None):
        assert query == "restart hermes gateway safely"
        assert top_k == 3
        assert min_score == 0.04
        assert wrapper_path is None
        return {
            "query": query,
            "recommendations": [
                {
                    "rank": 1,
                    "skill_name": "hermes-agent",
                    "relative_path": "autonomous-ai-agents/hermes-agent/SKILL.md",
                    "score": 0.2,
                }
            ],
            "diagnostics": {"min_score": min_score},
        }

    monkeypatch.setattr(skills_hub, "recommend_skills", fake_recommend_skills, raising=False)

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    args = SimpleNamespace(
        skills_action="recommend",
        query="restart hermes gateway safely",
        top_k=3,
        min_score=0.04,
        wrapper=None,
        show_scores=True,
    )

    skills_hub.skills_command(args, console=console)
    output = sink.getvalue()

    assert "hermes-agent" in output
    assert "Advisory only" in output
    assert "No skills were loaded" in output


def test_skills_slash_help_mentions_recommend_advisory_command():
    from hermes_cli.skills_hub import handle_skills_slash

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)

    handle_skills_slash("/skills help", console=console)
    output = sink.getvalue()

    assert "recommend" in output
    assert "advisory" in output.lower()
    assert "does not load" in output.lower()


def test_handle_skills_slash_recommend_is_advisory_only(monkeypatch):
    import hermes_cli.skill_recommend as skill_recommend
    from hermes_cli.skills_hub import handle_skills_slash

    calls = []

    def fake_recommend_skills(query, top_k=3, min_score=0.04, wrapper_path=None):
        calls.append({
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
            "wrapper_path": wrapper_path,
        })
        return {
            "query": query,
            "scope": {
                "manual_only": True,
                "loads_skills": False,
                "runtime_coupling": False,
                "db_writes": False,
            },
            "recommendations": [
                {
                    "rank": 1,
                    "skill_name": "github-code-review",
                    "relative_path": "github/github-code-review/SKILL.md",
                    "score": 0.188875,
                }
            ],
            "diagnostics": {"min_score": min_score},
        }

    monkeypatch.setattr(skill_recommend, "recommend_skills", fake_recommend_skills)

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)

    handle_skills_slash(
        "/skills recommend review this pull request --top-k 2 --min-score 0.05 --show-scores",
        console=console,
    )
    output = sink.getvalue()

    assert calls == [{
        "query": "review this pull request",
        "top_k": 2,
        "min_score": 0.05,
        "wrapper_path": None,
    }]
    assert "github-code-review" in output
    assert "score 0.188875" in output
    assert "Advisory only" in output
    assert "No skills were loaded" in output


def test_handle_skills_slash_recommend_requires_query():
    from hermes_cli.skills_hub import handle_skills_slash

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)

    handle_skills_slash("/skills recommend", console=console)
    output = sink.getvalue()

    assert "Usage:" in output
    assert "/skills recommend <query>" in output
