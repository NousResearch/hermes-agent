from pathlib import Path

from scripts.check_elixir_analytics_release_packaging import (
    classify_release_files,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_elixir_analytics_release_packaging_classifies_current_lanes():
    result = classify_release_files(
        [
            "profile-distributions/elixir-analytics/ROADMAP.md",
            "profile-distributions/elixir-analytics/profile_plugins/elixir-analytics-runner/__init__.py",
            "hermes_cli/plugins.py",
            "toolsets.py",
            "gateway/platforms/slack.py",
            "tests/gateway/test_slack.py",
            ".hermes-bootstrap-complete",
        ]
    )

    assert [package["id"] for package in result["packages"]] == [
        "profile-distribution",
        "hermes-runtime",
        "slack-gateway",
    ]
    assert result["local_artifacts"] == [".hermes-bootstrap-complete"]
    assert result["unclassified"] == []
    assert result["packages"][0]["files"] == [
        "profile-distributions/elixir-analytics/ROADMAP.md",
        "profile-distributions/elixir-analytics/profile_plugins/elixir-analytics-runner/__init__.py",
    ]


def test_elixir_analytics_release_packaging_flags_unknown_files():
    result = classify_release_files(
        [
            "profile-distributions/elixir-analytics/ROADMAP.md",
            "random/new-file.txt",
        ]
    )

    assert result["unclassified"] == ["random/new-file.txt"]


def test_elixir_analytics_release_packaging_doc_describes_the_split():
    body = (
        REPO_ROOT
        / "profile-distributions"
        / "elixir-analytics"
        / "RELEASE_PACKAGING.md"
    ).read_text(encoding="utf-8")

    assert "profile-distribution" in body
    assert "hermes-runtime" in body
    assert "slack-gateway" in body
    assert "Do not stage the full Hermes dirty worktree as one commit" in body
    assert "check_elixir_analytics_release_packaging.py --strict" in body
