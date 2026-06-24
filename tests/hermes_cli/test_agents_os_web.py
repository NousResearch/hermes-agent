import json
import os
from pathlib import Path

import pytest

from hermes_cli import agents_os, agents_os_web
from hermes_cli.agents_os import AgentsOSService, connect, log_event, resolve_paths, utc_now
from hermes_cli.agents_os_web import (
    agents_registry_payload,
    approval_detail_payload,
    artifact_detail_payload,
    artifacts_payload,
    create_idea_action,
    jarvis_briefing_payload,
    jarvis_model_advisor_payload,
    jarvis_preview_payload,
    jarvis_reply_payload,
    approvals_payload,
    cron_readiness_payload,
    events_payload,
    jarvis_transcribe_payload,
    knowledge_index_payload,
    media_assets_payload,
    mission_control_html,
    redacted_manage_status_payload,
    run_detail_payload,
    runs_payload,
    sessions_visibility_payload,
    skills_visibility_payload,
    task_detail_payload,
    tasks_payload,
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
    assert "demo=task-detail" in html
    assert "demo=approval-detail" in html
    assert "showTaskDetail(tasks.items[0].id)" in html
    assert "showApprovalDetail(approvals.items[0].id)" in html
    assert "vault/reference graph, not runtime memory merge" in html
    assert "Local-only operator cockpit" in html
    assert "Doni" not in html


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

    assert {"local-agent", "coding-delegate", "separate-profile", "external-reference-runtime"}.issubset(ids)
    local_agent = next(agent for agent in payload["agents"] if agent["id"] == "local-agent")
    assert local_agent["memory_boundary"] == "profile-local Hermes home only"
    assert "gateway restart" in " ".join(local_agent["approval_gates"])
    dumped = json.dumps(payload, ensure_ascii=False)
    assert "Doni" not in dumped
    assert "Goran" not in dumped
    assert "Marija" not in dumped
    assert "ERO" not in dumped
    assert "OpenClaw" not in dumped
    assert "/home/goran" not in dumped
    assert "/mnt/d" not in dumped


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


def test_task_approval_run_event_and_cron_payloads_are_read_only_and_redacted(agents_home):
    paths = resolve_paths(None)
    with connect(paths) as conn:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,route,approval_required) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("task-visible", "Visible task", "ready", "code-task", 1, utc_now(), utc_now(), "notes", "local:direct", 0),
        )
        conn.execute(
            "INSERT INTO approvals(id,title,status,risk,task_id,payload,created_at) VALUES(?,?,?,?,?,?,?)",
            ("approval-visible", "Visible approval", "pending", "external-action", "task-visible", '{"token":"secret-value"}', utc_now()),
        )
        conn.execute(
            "INSERT INTO runs(id,task_id,workflow,status,input,created_at,completed_at) VALUES(?,?,?,?,?,?,?)",
            ("run-visible", "task-visible", "code-task", "created", "safe input", utc_now(), None),
        )
        log_event(conn, "visible_event", task_id="task-visible", run_id="run-visible", payload={"cookie": "secret"})
        conn.commit()

    tasks = tasks_payload(paths)
    approvals = approvals_payload(paths)
    runs = runs_payload(paths)
    events = events_payload(paths)
    cron = cron_readiness_payload(paths)

    assert tasks["read_only"] is True
    assert any(item["id"] == "task-visible" for item in tasks["items"])
    approval = next(item for item in approvals["items"] if item["id"] == "approval-visible")
    assert approvals["resolution_enabled"] is False
    assert approval["payload_preview"] == "[redacted-sensitive-preview]"
    assert runs["read_only"] is True
    assert any(item["id"] == "run-visible" for item in runs["items"])
    event = next(item for item in events["items"] if item["event_type"] == "visible_event")
    assert event["payload_preview"] == "[redacted-sensitive-preview]"
    assert cron["read_only"] is True
    assert cron["cron_mutation_enabled"] is False



def test_detail_visibility_payloads_are_read_only_and_bounded(agents_home):
    paths = resolve_paths(None)
    artifact_file = paths.artifacts / "detail" / "artifact.md"
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text("# Evidence\n\nSafe local evidence.", encoding="utf-8")
    with connect(paths) as conn:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,route,approval_required) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("task-detail", "Detail task", "ready", "code-task", 1, utc_now(), utc_now(), "notes", "local:direct", 1),
        )
        conn.execute(
            "INSERT INTO approvals(id,title,status,risk,task_id,payload,created_at) VALUES(?,?,?,?,?,?,?)",
            ("approval-detail", "Detail approval", "pending", "external-action", "task-detail", '{"api_key":"hidden"}', utc_now()),
        )
        conn.execute(
            "INSERT INTO runs(id,task_id,workflow,status,input,created_at,completed_at) VALUES(?,?,?,?,?,?,?)",
            ("run-detail", "task-detail", "code-task", "created", '{"secret":"hidden"}', utc_now(), None),
        )
        conn.execute(
            "INSERT INTO artifacts(id,kind,title,path,task_id,workflow,created_at,run_id) VALUES(?,?,?,?,?,?,?,?)",
            ("artifact-detail", "verification", "Detail artifact", str(artifact_file), "task-detail", "qa-report", utc_now(), "run-detail"),
        )
        log_event(conn, "detail_event", task_id="task-detail", run_id="run-detail", payload={"token": "hidden"})
        conn.commit()

    task = task_detail_payload(paths, "task-detail")
    approval = approval_detail_payload(paths, "approval-detail")
    run = run_detail_payload(paths, "run-detail")
    artifact = artifact_detail_payload(paths, "artifact-detail")
    missing = approval_detail_payload(paths, "missing")

    assert task["read_only"] is True
    assert task["mutation_actions_enabled"] is False
    assert task["approvals"][0]["payload_preview"] == "[redacted-sensitive-preview]"
    assert task["runs"][0]["input_preview"] == "[redacted-sensitive-preview]"
    assert task["events"][0]["payload_preview"] == "[redacted-sensitive-preview]"
    assert approval["resolution_enabled"] is False
    assert approval["approval"]["payload_preview"] == "[redacted-sensitive-preview]"
    assert "approve" in approval["blocked_actions"]
    assert approval["risk_taxonomy"]["deterministic"] is True
    assert run["read_only"] is True
    assert run["run"]["input_preview"] == "[redacted-sensitive-preview]"
    assert artifact["preview_status"] == "ok"
    assert "Evidence" in artifact["preview"]
    assert missing["status"] == "not_found"


def test_skills_and_sessions_visibility_are_metadata_only(agents_home, tmp_path):
    paths = resolve_paths(None)
    skill = paths.home / "skills" / "demo" / "sample" / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text('---\nname: sample-skill\ndescription: Sample description\n---\nbody', encoding="utf-8")
    session = paths.home / "sessions" / "demo.json"
    session.parent.mkdir(parents=True, exist_ok=True)
    session.write_text('{"private":"content"}', encoding="utf-8")

    skills = skills_visibility_payload(paths)
    sessions = sessions_visibility_payload(paths)

    assert skills["read_only"] is True
    assert skills["content_visible"] is False
    assert any(item["name"] == "sample-skill" for item in skills["items"])
    assert sessions["metadata_only"] is True
    assert sessions["raw_transcript_visible"] is False
    assert any(item["file"] == "demo.json" for item in sessions["items"])

def test_jarvis_transcribe_writes_local_artifacts_without_execution(agents_home):
    paths = resolve_paths(None)

    payload = jarvis_transcribe_payload(
        paths,
        {
            "audio_base64": "UklGRg==",
            "audio_mime": "audio/webm",
            "transcript_text": "Prikaži zadnje BP24 stanje",
        },
    )

    assert payload["local_only"] is True
    assert payload["execution_created"] is False
    assert payload["status"] == "transcribed"
    assert payload["stt"]["provider"] == "provided_transcript"
    assert payload["advisor"]["provider"] == "deterministic"
    assert payload["transcript"]["text"] == "Prikaži zadnje BP24 stanje"
    assert payload["intent_preview"]["risk_class"] == "safe_local"
    assert payload["intent_preview"]["approval_required"] is False
    assert Path(payload["audio_artifact_path"]).exists()
    assert Path(payload["transcript_artifact_path"]).exists()
    assert "Prikaži zadnje BP24 stanje" in Path(payload["transcript_artifact_path"]).read_text(encoding="utf-8")


def test_jarvis_transcribe_uses_stt_adapter_when_transcript_missing(agents_home):
    paths = resolve_paths(None)

    payload = jarvis_transcribe_payload(
        paths,
        {
            "audio_base64": "UklGRg==",
            "audio_mime": "audio/webm",
            "stt_result": {"text": "Deployaj BP24", "provider": "local-faster-whisper", "confidence": 0.91},
        },
    )

    assert payload["execution_created"] is False
    assert payload["stt"]["provider"] == "local-faster-whisper"
    assert payload["stt"]["confidence"] == 0.91
    assert payload["transcript"]["text"] == "Deployaj BP24"
    assert payload["command_card"]["risk_class"] == "public_gated"
    assert payload["command_card"]["approval_required"] is True


def test_jarvis_transcribe_can_call_local_faster_whisper_adapter(agents_home, monkeypatch):
    paths = resolve_paths(None)
    seen = {}

    def fake_transcribe(audio_path, *, model="base", language="hr"):
        seen["audio_path"] = audio_path
        seen["model"] = model
        seen["language"] = language
        return {"text": "Prikaži zadnje BP24 stanje", "provider": "local-faster-whisper", "confidence": 0.83, "language": "hr"}

    monkeypatch.setattr(agents_os_web, "_transcribe_with_local_faster_whisper", fake_transcribe)

    payload = jarvis_transcribe_payload(
        paths,
        {
            "audio_base64": "UklGRg==",
            "audio_mime": "audio/webm",
            "use_local_stt": True,
            "stt_model": "small",
            "stt_language": "hr",
        },
    )

    assert payload["execution_created"] is False
    assert payload["stt"]["provider"] == "local-faster-whisper"
    assert payload["stt"]["confidence"] == 0.83
    assert payload["transcript"]["text"] == "Prikaži zadnje BP24 stanje"
    assert payload["command_card"]["risk_class"] == "safe_local"
    assert seen["audio_path"] == payload["audio_artifact_path"]
    assert seen["model"] == "small"
    assert seen["language"] == "hr"


def test_jarvis_transcribe_accepts_minimax_cleanup_but_preserves_raw_risk(agents_home):
    paths = resolve_paths(None)

    payload = jarvis_transcribe_payload(
        paths,
        {
            "audio_base64": "UklGRg==",
            "audio_mime": "audio/webm",
            "stt_result": {"text": "Deployaj BP24", "provider": "local-faster-whisper", "confidence": 0.91},
            "model_result": {
                "normalized_transcript": "Prikaži zadnje BP24 stanje",
                "semantic_intent": "status lookup",
                "risk_class": "safe_local",
                "voice_reply_short": "Prikazujem status.",
            },
            "advisor_provider": "minimax",
            "advisor_model": "MiniMax-M3",
        },
    )

    assert payload["execution_created"] is False
    assert payload["transcript"]["text"] == "Deployaj BP24"
    assert payload["transcript"]["cleaned_text"] == "Prikaži zadnje BP24 stanje"
    assert payload["advisor"]["provider"] == "minimax"
    assert payload["advisor"]["model"] == "MiniMax-M3"
    assert payload["advisor"]["risk_disagreement"] is True
    assert payload["command_card"]["risk_class"] == "public_gated"
    assert payload["command_card"]["approval_required"] is True
    assert "Deployaj BP24" in payload["command_card"]["gate_text"]


def test_jarvis_model_advisor_keeps_deterministic_gate_authoritative(agents_home):
    paths = resolve_paths(None)
    deterministic = jarvis_preview_payload(paths, {"transcript_text": "Deployaj BP24"})

    payload = jarvis_model_advisor_payload(
        paths,
        {
            "transcript_text": "Deployaj BP24",
            "deterministic_preview": deterministic,
            "model_result": {"semantic_intent": "deploy production", "risk_class": "safe_local", "voice_reply_short": "Mogu deployati."},
            "provider": "minimax",
            "model": "MiniMax-M3",
        },
    )

    assert payload["execution_created"] is False
    assert payload["provider"] == "minimax"
    assert payload["model"] == "MiniMax-M3"
    assert payload["authoritative_risk_class"] == "public_gated"
    assert payload["model_risk_class"] == "safe_local"
    assert payload["risk_disagreement"] is True
    assert payload["command_card"]["risk_class"] == "public_gated"
    assert payload["command_card"]["approval_required"] is True


def test_jarvis_preview_gates_risky_commands_without_execution(agents_home):
    paths = resolve_paths(None)

    safe = jarvis_preview_payload(paths, {"transcript_text": "Prikaži zadnje BP24 stanje"})
    public = jarvis_preview_payload(paths, {"transcript_text": "Pošalji klijentu email"})
    deploy = jarvis_preview_payload(paths, {"transcript_text": "Deployaj BP24"})
    security = jarvis_preview_payload(paths, {"transcript_text": "Pokreni sigurnosni scan klijentove stranice"})

    assert safe["command_card"]["risk_class"] == "safe_local"
    assert safe["command_card"]["approval_required"] is False
    assert safe["command_card"]["execution_created"] is False
    assert public["command_card"]["risk_class"] == "public_gated"
    assert public["command_card"]["approval_required"] is True
    assert public["command_card"]["execution_created"] is False
    assert deploy["command_card"]["risk_class"] == "public_gated"
    assert deploy["command_card"]["approval_required"] is True
    assert deploy["command_card"]["execution_created"] is False
    assert security["command_card"]["risk_class"] == "security_gated"
    assert security["command_card"]["approval_required"] is True
    assert security["command_card"]["execution_created"] is False


def test_jarvis_reply_stores_tts_audio_artifact_without_execution(agents_home):
    paths = resolve_paths(None)

    payload = jarvis_reply_payload(
        paths,
        {
            "text": "Presuda. Rezultat. Sljedeći korak.",
            "audio_base64": "SUQz",
            "audio_mime": "audio/mpeg",
            "provider": "hermes-tts",
            "voice_reply_short": "Presuda. Rezultat. Sljedeći korak.",
        },
    )

    assert payload["local_only"] is True
    assert payload["execution_created"] is False
    assert payload["status"] == "audio_ready"
    assert payload["tts"]["provider"] == "hermes-tts"
    assert payload["tts"]["fallback"] is False
    assert Path(payload["audio_artifact_path"]).exists()
    assert Path(payload["reply_artifact_path"]).exists()
    assert "Presuda. Rezultat. Sljedeći korak." in Path(payload["reply_artifact_path"]).read_text(encoding="utf-8")


def test_jarvis_reply_falls_back_to_text_only_without_audio(agents_home):
    paths = resolve_paths(None)

    payload = jarvis_reply_payload(paths, {"text": "Ovo treba odobrenje. Ništa ne izvršavam."})

    assert payload["local_only"] is True
    assert payload["execution_created"] is False
    assert payload["status"] == "text_only"
    assert payload["tts"]["provider"] == "text-only-fallback"
    assert payload["tts"]["fallback"] is True
    assert payload["audio_artifact_path"] is None
    assert Path(payload["reply_artifact_path"]).exists()


def test_jarvis_reply_can_prepare_hume_octave_request_draft_without_api_call(agents_home):
    paths = resolve_paths(None)

    payload = jarvis_reply_payload(
        paths,
        {
            "text": "Presuda. Jarvis radi lokalno.",
            "provider": "hume-octave",
            "voice_description": "calm Croatian operator voice, concise and warm",
            "format": "mp3",
        },
    )

    assert payload["execution_created"] is False
    assert payload["status"] == "provider_unconfigured"
    assert payload["tts"]["provider"] == "hume-octave"
    assert payload["tts"]["requires_api_key"] is True
    assert payload["tts"]["api_called"] is False
    assert payload["audio_artifact_path"] is None
    assert payload["hume_octave_request"]["utterances"][0]["text"] == "Presuda. Jarvis radi lokalno."
    assert payload["hume_octave_request"]["utterances"][0]["description"] == "calm Croatian operator voice, concise and warm"
    assert payload["hume_octave_request"]["format"]["type"] == "mp3"


def test_root_html_contains_jarvis_oracle_briefing_panel(agents_home):
    service = AgentsOSService(resolve_paths(None))
    html = mission_control_html(service)

    assert "Jarvis / Oracle Briefing" in html
    assert "Record command" in html
    assert "Command Preview" in html
    assert "/api/jarvis/briefing" in html
    assert "/api/jarvis/transcribe" in html
    assert "/api/jarvis/preview" in html
    assert "/api/jarvis/reply" in html
    assert "Voice Reply" in html
    assert "wake/show/build/act" in html
