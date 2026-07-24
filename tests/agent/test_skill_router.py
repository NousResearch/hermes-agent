"""Tests for deterministic local skill-router proof output."""

from __future__ import annotations

from agent.skill_router import route_skills, visible_skill_records


def _write_skill(root, rel, *, name, description, body="", metadata=""):
    skill_dir = root / "skills" / rel
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"{metadata}"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )


def test_route_skills_selects_release_sre_memory_sandbox_skills(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_skill(
        tmp_path,
        "software-development/gateway-heartbeat-investigation",
        name="gateway-heartbeat-investigation",
        description="Proof-based investigation of Hermes gateway heartbeat status.",
        body="systemd heartbeat live gateway watchdog stale health proof",
    )
    _write_skill(
        tmp_path,
        "software-development/hermes-cron-state-authority-repair",
        name="hermes-cron-state-authority-repair",
        description="Diagnose and repair Hermes cron jobs that appear missing or stale.",
        body="cron state authority repair heartbeat self improvement self annealing",
    )
    _write_skill(
        tmp_path,
        "software-development/systematic-debugging",
        name="systematic-debugging",
        description="4-phase root cause debugging before fixing.",
        body="diagnose root cause evidence tests",
    )
    _write_skill(
        tmp_path,
        "creative/ascii-art",
        name="ascii-art",
        description="Make terminal art.",
        body="figlet cowsay banners",
    )

    result = route_skills(
        "Diagnose Hermes heartbeat, cron, memory, sandbox and self-improvement quality gates",
        top_k=3,
    )

    selected = [item["name"] for item in result["selected_skills"]]
    assert result["router_available"] is True
    assert "gateway-heartbeat-investigation" in selected
    assert "hermes-cron-state-authority-repair" in selected
    assert "ascii-art" not in selected
    assert result["candidate_count"] >= 3
    assert result["selection_source"].startswith("deterministic local lexical router")


def test_route_skills_reports_skipped_better_looking_skills(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for idx in range(4):
        _write_skill(
            tmp_path,
            f"software-development/release-gate-{idx}",
            name=f"release-gate-{idx}",
            description="release gate sandbox proof test mutation rollback",
            body="release sandbox test mutation rollback",
        )

    result = route_skills("release sandbox proof mutation rollback", top_k=2)

    assert len(result["selected_skills"]) == 2
    assert len(result["skipped_better_looking_skills"]) >= 1
    assert {"name", "score", "why"}.issubset(result["selected_skills"][0])


def test_visible_skill_records_respects_toolset_requirements(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_skill(
        tmp_path,
        "smart-home/openhue",
        name="openhue",
        description="Hue lights",
        metadata="metadata:\n  hermes:\n    requires_toolsets: [terminal]\n",
    )

    assert visible_skill_records(available_toolsets=set()) == []
    records = visible_skill_records(available_toolsets={"terminal"})
    assert [record["name"] for record in records] == ["openhue"]


def test_route_skills_abstains_when_only_stopwords_overlap(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_skill(
        tmp_path,
        "software-development/generic-release",
        name="generic-release",
        description="Use when operating release workflows for teams.",
        body="release deploy rollback production",
    )
    _write_skill(
        tmp_path,
        "creative/generic-writing",
        name="generic-writing",
        description="Use when writing prose for people.",
        body="draft edit publish",
    )

    result = route_skills("use when for the and", top_k=3)

    assert result["selected_skills"] == []
    assert result["candidate_count"] == 0
    assert result["warning"] == "SKILL_ROUTER_ABSTAINED_LOW_CONFIDENCE_USE_MANUAL_SELECTION"


def test_route_skills_searches_full_skill_body_not_first_4000_chars(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    deep_body = ("neutral filler " * 450) + " modal direct terminal backend canary "
    _write_skill(
        tmp_path,
        "software-development/deep-modal-backend",
        name="deep-modal-backend",
        description="General backend notes.",
        body=deep_body,
    )

    result = route_skills("modal direct terminal backend canary", top_k=1)

    assert [item["name"] for item in result["selected_skills"]] == ["deep-modal-backend"]
    assert {"modal", "direct", "terminal", "backend", "canary"}.issubset(
        set(result["selected_skills"][0]["matched_terms"])
    )


def test_route_skills_exact_name_champion_beats_body_spam(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_skill(
        tmp_path,
        "software-development/gateway-heartbeat-investigation",
        name="gateway-heartbeat-investigation",
        description="Proof-based heartbeat investigation.",
        body="brief gateway heartbeat proof",
    )
    _write_skill(
        tmp_path,
        "software-development/generic-gateway-notes",
        name="generic-gateway-notes",
        description="Generic notes.",
        body="gateway heartbeat investigation " * 300,
    )

    result = route_skills(
        "Use gateway-heartbeat-investigation for gateway heartbeat investigation",
        top_k=1,
    )

    assert [item["name"] for item in result["selected_skills"]] == [
        "gateway-heartbeat-investigation"
    ]
    assert any(
        "champion" in reason.lower() for reason in result["selected_skills"][0]["why"]
    )


def test_route_skills_reports_setup_needed_readiness(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_ROUTER_TEST_MISSING_VALUE", raising=False)
    _write_skill(
        tmp_path,
        "mlops/cloud-widget",
        name="cloud-widget",
        description="Cloud widget operations.",
        body="cloud widget deploy",
        metadata=(
            "required_environment_variables:\n"
            "  - name: HERMES_ROUTER_TEST_MISSING_VALUE\n"
        ),
    )

    result = route_skills("cloud-widget cloud widget operations", top_k=1)

    assert result["selected_skills"][0]["name"] == "cloud-widget"
    assert result["selected_skills"][0]["readiness_status"] == "setup_needed"
    assert result["selected_skills"][0]["setup_needed"] is True
