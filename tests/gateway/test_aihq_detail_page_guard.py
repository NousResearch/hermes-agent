from types import SimpleNamespace

from gateway.run import (
    GatewayRunner,
    _aihq_detail_page_pipeline_guard,
    _aihq_next_patch_version,
    _aihq_joint_meeting_router_intercept,
    _aihq_pipeline_guard,
    _aihq_read_artifact_manifest,
    _aihq_record_artifact_manifest,
)


def _source(platform: str) -> SimpleNamespace:
    return SimpleNamespace(platform=SimpleNamespace(value=platform))


def _slack_source(channel_id: str = "C123", thread_ts: str = "171.000") -> SimpleNamespace:
    return SimpleNamespace(platform=SimpleNamespace(value="slack"), chat_id=channel_id, thread_id=thread_ts)


def test_aihq_detail_page_guard_injects_for_slack_detail_page_request():
    prompt = _aihq_pipeline_guard("곰돌이 상세페이지 만들어줘", _source("slack"))

    assert "canonical 9 pipeline" in prompt
    assert "pipeline_id" in prompt
    assert "/mywiki" in prompt
    assert "Supanova" in prompt
    assert "Open Design AIHQ Supanova" in prompt
    assert "Do not jump directly" in prompt
    assert "YYYYMMDD_프로젝트명_PROGRESS.md" in prompt
    assert "slack-thread-projects.jsonl" in prompt


def test_aihq_detail_page_guard_skips_non_slack_sources():
    prompt = _aihq_pipeline_guard("상세페이지 만들어줘", _source("local"))

    assert prompt == ""


def test_aihq_detail_page_guard_skips_unrelated_slack_messages():
    prompt = _aihq_pipeline_guard("오늘 재고만 확인해줘", _source("slack"))

    assert prompt == ""


def test_aihq_pipeline_guard_injects_for_proposal_request():
    prompt = _aihq_pipeline_guard("이마트24 제안서 만들어줘", _source("slack"))

    assert "canonical 9 pipeline" in prompt
    assert "pipeline_id" in prompt
    assert "actual working team-lead roster" in prompt


def test_aihq_pipeline_guard_injects_for_market_research_report_request():
    prompt = _aihq_pipeline_guard("시장조사 보고서 작성해줘", _source("slack"))

    assert "canonical 9 pipeline" in prompt
    assert "/mywiki" in prompt
    assert "My Note" in prompt


def test_aihq_pipeline_guard_injects_for_automation_dev_request():
    prompt = _aihq_pipeline_guard("Slack 자동화 개발복구 작업 진행해줘", _source("slack"))

    assert "canonical 9 pipeline" in prompt
    assert "Superpowers/GStack" in prompt
    assert "verification plan" in prompt


def test_aihq_detail_page_guard_wrapper_matches_pipeline_guard():
    message = "곰돌이 상세페이지 만들어줘"
    source = _source("slack")

    assert _aihq_detail_page_pipeline_guard(message, source) == _aihq_pipeline_guard(message, source)


def test_aihq_pipeline_guard_resume_without_index_blocks_new_work(tmp_path, monkeypatch):
    index = tmp_path / "missing.jsonl"
    monkeypatch.setattr("gateway.run._AIHQ_THREAD_INDEX_PATH", index)

    prompt = _aihq_pipeline_guard("어제 하던 거 이어서 진행", _slack_source())

    assert "Project Continuity Guard" in prompt
    assert "no project index entry" in prompt
    assert "Do not start a new project" in prompt


def test_aihq_pipeline_guard_resume_injects_canonical_progress(tmp_path, monkeypatch):
    progress = tmp_path / "02_Projects" / "20260522_테스트_PROGRESS.md"
    progress.parent.mkdir(parents=True)
    progress.write_text("- current_phase: GATE 3 시안 승인 대기\n- active_next_action: QA\n", encoding="utf-8")
    index = tmp_path / "04_Data" / "runtime-project-index" / "slack-thread-projects.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text(
        '{"workspace":"hermes","pipeline_id":"detail_page","project_slug":"test","progress_path":"'
        + str(progress)
        + '","output_root":"/tmp/out","slack_channel_id":"C123","slack_thread_ts":"171.000","session_key":"s","run_id":"r","status":"active","updated_at":"2026-05-22T00:00:00+09:00"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.run._AIHQ_THREAD_INDEX_PATH", index)

    prompt = _aihq_pipeline_guard("같은 스레드 이어서 해줘", _slack_source())

    assert "canonical_progress_path" in prompt
    assert str(progress) in prompt
    assert "GATE 3 시안 승인 대기" in prompt
    assert "active_next_action" in prompt


def test_aihq_pipeline_guard_resume_conflict_reports_candidates(tmp_path, monkeypatch):
    index = tmp_path / "04_Data" / "runtime-project-index" / "slack-thread-projects.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text(
        "\n".join(
            [
                '{"workspace":"hermes","pipeline_id":"detail_page","project_slug":"a","progress_path":"/tmp/a_PROGRESS.md","output_root":"/tmp/a","slack_channel_id":"C123","slack_thread_ts":"171.000","session_key":"s","run_id":"r1","status":"active","updated_at":"2026-05-22T00:00:00+09:00"}',
                '{"workspace":"hermes","pipeline_id":"detail_page","project_slug":"b","progress_path":"/tmp/b_PROGRESS.md","output_root":"/tmp/b","slack_channel_id":"C123","slack_thread_ts":"171.000","session_key":"s","run_id":"r2","status":"active","updated_at":"2026-05-22T00:01:00+09:00"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.run._AIHQ_THREAD_INDEX_PATH", index)

    prompt = _aihq_pipeline_guard("계속 진행", _slack_source())

    assert "multiple canonical progress-file candidates" in prompt
    assert "/tmp/a_PROGRESS.md" in prompt
    assert "/tmp/b_PROGRESS.md" in prompt


def test_aihq_pipeline_guard_new_work_does_not_inject_old_progress(tmp_path, monkeypatch):
    index = tmp_path / "04_Data" / "runtime-project-index" / "slack-thread-projects.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text(
        '{"workspace":"hermes","pipeline_id":"detail_page","project_slug":"test","progress_path":"/tmp/test_PROGRESS.md","output_root":"/tmp/out","slack_channel_id":"C123","slack_thread_ts":"171.000","session_key":"s","run_id":"r","status":"active","updated_at":"2026-05-22T00:00:00+09:00"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.run._AIHQ_THREAD_INDEX_PATH", index)

    prompt = _aihq_pipeline_guard("새 상세페이지 만들어줘", _slack_source())

    assert "canonical 9 pipeline" in prompt
    assert "canonical_progress_path" not in prompt


def test_aihq_pipeline_guard_direct_meeting_channel_injects_joint_meeting_guard():
    prompt = _aihq_pipeline_guard("신규 상품 후보를 같이 보자", _slack_source("C0B4TLSMNJG", "1779762018.415589"))

    assert "AIHQ Joint Meeting Guard" in prompt
    assert "정실장" in prompt
    assert "김팀장" in prompt
    assert "meeting-sessions.jsonl" in prompt
    assert "kim-teamlead" in prompt
    assert "do not load AIHQ workflow skills" in prompt
    assert "under 900 Korean characters" in prompt


def test_aihq_pipeline_guard_team_channel_meeting_trigger_transfers_to_meeting_channel():
    prompt = _aihq_pipeline_guard("이 건 회의 진행하자", _slack_source("C0B487QE023", "1779762018.415589"))

    assert "AIHQ Joint Meeting Guard" in prompt
    assert "#aihq-meeting" in prompt
    assert "Do not answer as a normal channel reply" in prompt


def test_aihq_pipeline_guard_meeting_duplicate_scope_is_exact_thread():
    prompt = _aihq_pipeline_guard("같은 안건으로 2라운드 회의 진행해봐", _slack_source("C0B4TLSMNJG", "1779779206.066769"))

    assert "AIHQ Joint Meeting Guard" in prompt
    assert "channel_id + thread_ts" in prompt
    assert "same topic in a different thread is a new meeting" in prompt
    assert "must never block a new direct `#aihq-meeting` agenda" in prompt


def test_aihq_joint_meeting_router_intercepts_meeting_channel_and_team_trigger():
    assert _aihq_joint_meeting_router_intercept(
        "새 안건 회의하자",
        _slack_source("C0B4TLSMNJG", "1779780000.1"),
    )
    assert _aihq_joint_meeting_router_intercept(
        "이 건 회의 진행하자",
        _slack_source("C0B487QE023", "1779780000.2"),
    )
    assert not _aihq_joint_meeting_router_intercept(
        "오늘 상태만 확인",
        _slack_source("C0B487QE023", "1779780000.3"),
    )


def test_aihq_artifact_manifest_records_hash_and_next_patch(tmp_path):
    artifact = tmp_path / "preview_v3_1.html"
    artifact.write_text("<html>v3.1</html>", encoding="utf-8")
    manifest = tmp_path / "artifact_manifest.jsonl"

    record = _aihq_record_artifact_manifest(
        manifest,
        version="v3.1",
        path=str(artifact),
        source_version="v3",
        slack_upload_ts="171.123",
        status="uploaded",
    )
    entries = _aihq_read_artifact_manifest(manifest)

    assert record["sha256"]
    assert entries[0]["version"] == "v3.1"
    assert entries[0]["source_version"] == "v3"
    assert _aihq_next_patch_version("v3", entries) == "v3.2"


def test_aihq_model_routing_applies_work_turn_session_override(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._agent_cache_lock = None

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {"provider": "openai-codex"},
            "aihq_model_routing": {
                "enabled": True,
                "provider": "openai-codex",
                "work_intake_model": "gpt-5.5",
                "work_intake_reasoning_effort": "medium",
            },
        },
    )

    runner._apply_aihq_model_routing_for_turn(
        message="상세페이지 수정해줘",
        source=_slack_source(),
        session_key="agent:main:slack:channel:C123:171.000",
    )

    override = runner._session_model_overrides["agent:main:slack:channel:C123:171.000"]
    assert override["model"] == "gpt-5.5"
    assert override["provider"] == "openai-codex"
    assert runner._session_reasoning_overrides["agent:main:slack:channel:C123:171.000"]
