import json
import os
from pathlib import Path

import pytest

from hermes_cli import agents_os
from hermes_cli.agents_os import AgentsOSService, connect, log_event, resolve_paths, utc_now
from hermes_cli.agents_os_web import (
    agents_registry_payload,
    artifacts_payload,
    create_idea_action,
    jarvis_briefing_payload,
    knowledge_index_payload,
    media_assets_payload,
    mission_control_html,
    redacted_manage_status_payload,
    voice_status_payload,
)


@pytest.fixture()
def agents_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("AGENTS_OS_HOME", str(home / "agents_os"))
    monkeypatch.setenv("AGENTS_OS_VAULT_ROOT", str(tmp_path / "vault"))
    agents_os.main(["init", "--no-vault"])
    return home


def test_root_html_contains_operator_tabs_and_bootstrap_payload(agents_home):
    service = AgentsOSService(resolve_paths(None))
    html = mission_control_html(service)

    assert "Agents OS Mission Control" in html
    for label in [
        "Idea Factory",
        "Agent Registry",
        "Knowledge Galaxy",
        "Artifact Library",
        "Operator Loop",
        "Media Studio",
        "Manage / Status",
        "Voice / Jarvis",
    ]:
        assert label in html
    assert "/api/idea-factory/draft" in html
    assert "vault/reference graph, not runtime memory merge" in html


def test_idea_factory_schema_and_draft_payloads_are_local_only(agents_home):
    service = AgentsOSService(resolve_paths(None))

    schema = service.idea_factory_schema_payload()
    draft = service.idea_factory_draft_payload({"idea_text": "Pošalji email klijentu"})

    assert schema["local_only"] is True
    assert draft["approval_required"] is True
    assert draft["risk_class"] == "public_gated"
    assert draft["execution_created"] is False


def test_create_idea_action_creates_safe_task_but_gates_public_action(agents_home):
    service = AgentsOSService(resolve_paths(None))

    safe = create_idea_action(service, {"idea_text": "Obradi YouTube video"})
    gated = create_idea_action(service, {"idea_text": "Pošalji email klijentu"})

    assert safe["mode"] == "safe_local_task"
    assert safe["task_id"].startswith("task-")
    assert safe["approval_id"] is None
    assert gated["mode"] == "approval_draft"
    assert gated["task_id"].startswith("task-")
    assert gated["approval_id"].startswith("approval-")
    assert gated["execution_created"] is False
    with connect(resolve_paths(None)) as conn:
        gated_task = conn.execute("SELECT * FROM tasks WHERE id=?", (gated["task_id"],)).fetchone()
        approval = conn.execute("SELECT * FROM approvals WHERE id=?", (gated["approval_id"],)).fetchone()
    assert gated_task["status"] == "needs_approval"
    assert approval["status"] == "pending"


def test_agent_registry_payload_includes_boundaries(agents_home):
    payload = agents_registry_payload(resolve_paths(None))
    ids = {agent["id"] for agent in payload["agents"]}

    assert {"doni-local", "kodi-codex", "marija-profile", "ero-openclaw"}.issubset(ids)
    doni = next(agent for agent in payload["agents"] if agent["id"] == "doni-local")
    assert doni["memory_boundary"] == "Doni Hermes home only"
    assert "gateway restart" in " ".join(doni["approval_gates"])


def test_knowledge_index_is_non_empty_and_links_video_sources(agents_home, tmp_path, monkeypatch):
    transcript = tmp_path / "q13OqknCh-c_transcript.txt"
    transcript.write_text("Mission Control transcript", encoding="utf-8")
    plan = tmp_path / "2026-06-08-agent-os-full-product-plan.md"
    plan.write_text("# Full product plan", encoding="utf-8")
    monkeypatch.setenv("AGENTS_OS_SOURCE_TRANSCRIPT", str(transcript))
    monkeypatch.setenv("AGENTS_OS_SOURCE_FULL_PLAN", str(plan))

    payload = knowledge_index_payload(resolve_paths(None))

    assert payload["local_only"] is True
    assert payload["runtime_memory_merge"] is False
    assert payload["nodes"]
    assert any(node["id"] == "video:q13OqknCh-c" for node in payload["nodes"])
    assert any(edge["from"] == "video:q13OqknCh-c" for edge in payload["edges"])


def test_artifacts_media_operator_manage_voice_payloads_are_redacted_and_read_only(agents_home, tmp_path):
    paths = resolve_paths(None)
    note = paths.artifacts / "smoke" / "note.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# smoke", encoding="utf-8")
    image = paths.artifacts / "screenshots" / "shot.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    with connect(paths) as conn:
        conn.execute(
            "INSERT INTO artifacts(id,kind,title,path,task_id,workflow,created_at) VALUES(?,?,?,?,?,?,?)",
            ("artifact-test", "smoke_report", "Smoke", str(note), None, "qa-report", utc_now()),
        )
        log_event(conn, "judge_pending", payload={"reason": "not implemented"})
        conn.commit()

    artifacts = artifacts_payload(paths)
    media = media_assets_payload(paths)
    manage = redacted_manage_status_payload(paths)
    voice = voice_status_payload(paths)
    jarvis = jarvis_briefing_payload(paths)
    operator = AgentsOSService(paths).operator_loop_payload()

    assert artifacts["items"]
    assert media["assets"]
    assert manage["credentials_visible"] is False
    assert "api_key" not in json.dumps(manage).lower()
    assert voice["computer_control"] == "approval_gated_unexecuted"
    assert jarvis["local_only"] is True
    assert jarvis["execution_created"] is False
    assert jarvis["always_on_microphone"] is False
    assert jarvis["wake_word_enabled"] is False
    assert jarvis["computer_control"] == "approval_gated_unexecuted"
    assert {command["name"] for command in jarvis["commands"]} == {"wake", "show", "build", "act"}
    assert jarvis["briefing"]["artifact_count"] >= 1
    assert "cross_agent_memory_merge" in jarvis["approval_gates"]
    assert operator["judge_status"] in {"pending", "ready"}


def test_root_html_contains_jarvis_oracle_briefing_panel(agents_home):
    service = AgentsOSService(resolve_paths(None))
    html = mission_control_html(service)

    assert "Jarvis / Oracle Briefing" in html
    assert "/api/jarvis/briefing" in html
    assert "wake/show/build/act" in html
