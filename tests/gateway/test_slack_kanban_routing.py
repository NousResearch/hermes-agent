import json
from pathlib import Path

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.kanban_routing import (
    PROTECTED_ROUTE_STATUS,
    build_routed_kanban_request,
    build_stage0_record_only_decision,
    load_routing_map,
    normalize_routing_key,
    resolve_route,
    route_kanban_create_tokens,
)


def _slack_source(thread_id="1779768050.823539"):
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C0B67947LMA",
        chat_name="invest-system-build",
        chat_type="channel",
        user_id="U123",
        user_name="광현",
        thread_id=thread_id,
    )


def test_normalize_routing_key_preserves_slack_thread():
    key = normalize_routing_key(_slack_source())

    assert key == "slack:C0B67947LMA:1779768050.823539"


def test_resolve_route_prefers_thread_route_over_channel_fallback():
    routing_map = {
        "routes": {
            "slack:C0B67947LMA": {
                "board": "ops-build",
                "anchor_task_id": "t_channel",
            },
            "slack:C0B67947LMA:1779768050.823539": {
                "board": "invest-system-build",
                "anchor_task_id": "t_4778246f",
            },
        }
    }

    route = resolve_route(_slack_source(), routing_map)

    assert route.matched_key == "slack:C0B67947LMA:1779768050.823539"
    assert route.board == "invest-system-build"
    assert route.anchor_task_id == "t_4778246f"


def test_build_routed_request_carries_origin_and_report_to_metadata():
    routing_map = {
        "routes": {
            "slack:C0B67947LMA:1779768050.823539": {
                "board": "invest-system-build",
                "anchor_task_id": "t_4778246f",
            }
        }
    }

    request = build_routed_kanban_request(
        title="Validator follow-up",
        body="fixture-only 작업 계속",
        source=_slack_source(),
        routing_map=routing_map,
    )

    assert request.board == "invest-system-build"
    assert request.parents == ("t_4778246f",)
    assert request.initial_status == "running"
    assert request.metadata["origin"] == {
        "platform": "slack",
        "chat_id": "C0B67947LMA",
        "thread_id": "1779768050.823539",
        "chat_name": "invest-system-build",
        "chat_type": "channel",
        "user_id": "U123",
        "user_name": "광현",
    }
    assert request.metadata["report_to"] == {
        "platform": "slack",
        "chat_id": "C0B67947LMA",
        "thread_id": "1779768050.823539",
    }
    assert request.metadata["routing"]["matched_key"] == "slack:C0B67947LMA:1779768050.823539"


def test_protected_scope_becomes_blocked_approval_request():
    routing_map = {
        "routes": {
            "slack:C0B67947LMA:1779768050.823539": {
                "board": "invest-system-build",
                "anchor_task_id": "t_4778246f",
            }
        }
    }

    request = build_routed_kanban_request(
        title="보호 작업",
        body="M1 SSH로 KCC source 반영하고 gateway 재시작",
        source=_slack_source(),
        routing_map=routing_map,
    )

    assert request.initial_status == PROTECTED_ROUTE_STATUS
    assert request.metadata["routing"]["protected_scope"] is True
    assert "matched_terms" in request.metadata["routing"]


def test_load_routing_map_uses_temp_hermes_home_without_live_writes(tmp_path, monkeypatch):
    empty_home = tmp_path / "empty-home"
    monkeypatch.setenv("HERMES_HOME", str(empty_home))
    missing_routing_file = empty_home / "kanban" / "routing_map.json"

    assert load_routing_map() == {"routes": {}}
    assert not missing_routing_file.exists()

    routing_home = tmp_path / "routing-home"
    monkeypatch.setenv("HERMES_HOME", str(routing_home))
    routing_file = routing_home / "kanban" / "routing_map.json"
    routing_file.parent.mkdir(parents=True)
    routing_file.write_text(
        json.dumps(
            {
                "routes": {
                    "slack:C0B67947LMA:1779768050.823539": {
                        "board": "invest-system-build",
                        "anchor_task_id": "t_4778246f",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    routing_map = load_routing_map()

    assert resolve_route(_slack_source(), routing_map).board == "invest-system-build"


def test_route_kanban_create_tokens_injects_board_parent_status_and_metadata():
    routing_map = {
        "routes": {
            "slack:C0B67947LMA:1779768050.823539": {
                "board": "invest-system-build",
                "anchor_task_id": "t_4778246f",
            }
        }
    }

    tokens, request = route_kanban_create_tokens(
        ["create", "Validator follow-up", "--assignee", "alice"],
        source=_slack_source(),
        routing_map=routing_map,
    )

    assert request is not None
    assert tokens[:3] == ["--board", "invest-system-build", "create"]
    assert "--parent" in tokens
    assert "t_4778246f" in tokens
    assert "--initial-status" in tokens
    assert request.initial_status == "running"
    metadata = json.loads(tokens[tokens.index("--metadata") + 1])
    assert metadata["report_to"]["thread_id"] == "1779768050.823539"


def test_route_kanban_create_tokens_respects_explicit_board():
    tokens, request = route_kanban_create_tokens(
        ["--board", "ops-build", "create", "Manual board"],
        source=_slack_source(),
        routing_map={"routes": {"slack:C0B67947LMA": {"board": "invest-system-build"}}},
    )

    assert tokens == ["--board", "ops-build", "create", "Manual board"]
    assert request is None


def test_protected_scope_is_blocked_even_without_matching_route():
    tokens, request = route_kanban_create_tokens(
        ["create", "승인 필요", "--assignee", "alice", "gateway restart 요청"],
        source=_slack_source(),
        routing_map={"routes": {}},
    )

    assert request is not None
    assert request.board is None
    assert request.initial_status == PROTECTED_ROUTE_STATUS
    assert tokens[0] == "create"
    assert "--board" not in tokens
    assert tokens[tokens.index("--initial-status") + 1] == PROTECTED_ROUTE_STATUS
    metadata = json.loads(tokens[tokens.index("--metadata") + 1])
    assert metadata["routing"]["protected_scope"] is True


def test_route_kanban_create_tokens_preserves_explicit_initial_status():
    routing_map = {
        "routes": {
            "slack:C0B67947LMA:1779768050.823539": {
                "board": "expert-agents-build",
                "anchor_task_id": "t_anchor",
            }
        }
    }

    tokens, request = route_kanban_create_tokens(
        ["create", "manual", "--initial-status", "blocked"],
        source=_slack_source(),
        routing_map=routing_map,
    )

    assert request is not None
    assert tokens.count("--initial-status") == 1
    assert tokens[tokens.index("--initial-status") + 1] == "blocked"


def test_protected_scope_forces_blocked_over_explicit_running_status():
    routing_map = {
        "routes": {
            "slack:C0B67947LMA:1779768050.823539": {
                "board": "expert-agents-build",
                "anchor_task_id": "t_anchor",
            }
        }
    }

    tokens, request = route_kanban_create_tokens(
        ["create", "gateway restart", "--initial-status", "running"],
        source=_slack_source(),
        routing_map=routing_map,
    )

    assert request is not None
    assert request.initial_status == PROTECTED_ROUTE_STATUS
    assert tokens.count("--initial-status") == 1
    assert tokens[tokens.index("--initial-status") + 1] == PROTECTED_ROUTE_STATUS


def test_protected_scope_forces_blocked_over_equals_running_status():
    tokens, request = route_kanban_create_tokens(
        ["create", "M1 작업", "--initial-status=running"],
        source=_slack_source(),
        routing_map={"routes": {}},
    )

    assert request is not None
    assert "--initial-status=blocked" in tokens


def test_route_kanban_create_tokens_merges_existing_metadata_without_losing_origin():
    routing_map = {
        "routes": {
            "slack:C0B67947LMA:1779768050.823539": {
                "board": "expert-agents-build",
                "anchor_task_id": "t_anchor",
            }
        }
    }

    tokens, request = route_kanban_create_tokens(
        ["create", "manual", "--metadata", json.dumps({"manual": "keep"})],
        source=_slack_source(),
        routing_map=routing_map,
    )

    assert request is not None
    assert tokens.count("--metadata") == 1
    metadata = json.loads(tokens[tokens.index("--metadata") + 1])
    assert metadata["manual"] == "keep"
    assert metadata["origin"]["chat_id"] == "C0B67947LMA"
    assert metadata["routing"]["matched_key"] == "slack:C0B67947LMA:1779768050.823539"


def test_route_kanban_create_tokens_does_not_hide_malformed_metadata():
    routing_map = {
        "routes": {
            "slack:C0B67947LMA:1779768050.823539": {
                "board": "expert-agents-build",
                "anchor_task_id": "t_anchor",
            }
        }
    }

    tokens, request = route_kanban_create_tokens(
        ["create", "manual", "--metadata", "{"],
        source=_slack_source(),
        routing_map=routing_map,
    )

    assert request is not None
    assert tokens.count("--metadata") == 1
    assert tokens[tokens.index("--metadata") + 1] == "{"

def _stage0_routing_map():
    return {
        "routes": {
            "slack:C0B67947LMA:1779768050.823539": {
                "board": "ops-build",
                "anchor_task_id": "t_stage0",
            }
        }
    }


def test_stage0_record_only_passes_normal_anchored_instruction_without_dispatch():
    decision = build_stage0_record_only_decision(
        source=_slack_source(),
        text="현재 스레드 내용을 ops-build 후보로 기록해줘",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "PASS_RECORD_ONLY"
    assert decision.board == "ops-build"
    assert decision.anchor_task_id == "t_stage0"
    assert decision.initial_status == "blocked"
    assert decision.dispatch_allowed is False
    assert decision.ready_status_allowed is False
    assert decision.to_metadata()["dispatch_allowed"] is False


def test_stage0_record_only_allows_protected_terms_inside_forbidden_boundary_text():
    decision = build_stage0_record_only_decision(
        source=_slack_source(),
        text="금지 범위: M1 SSH, KCC source, DB write, secret/.env. dry-run만 진행",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "PASS_RECORD_ONLY"
    assert decision.reason == "protected_terms_only_in_boundary_text"
    assert "m1" in decision.protected_terms
    assert "kcc_source" in decision.protected_terms
    assert "secret_plaintext" in decision.protected_terms
    assert decision.dispatch_allowed is False


def test_stage0_record_only_blocks_secret_like_value_and_redacts_preview():
    synthetic_value = "sk_" + "STAGE0" + "A" * 12
    decision = build_stage0_record_only_decision(
        source=_slack_source(),
        text=f"이 토큰을 저장해줘 {synthetic_value}",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "BLOCK_SECRET"
    assert decision.redactions_applied is True
    assert synthetic_value not in decision.sanitized_preview
    assert "[REDACTED]" in decision.sanitized_preview
    assert decision.dispatch_allowed is False


def test_stage0_record_only_blocks_missing_route_or_anchor():
    decision = build_stage0_record_only_decision(
        source=_slack_source(thread_id="missing"),
        text="카드 후보로 기록",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "BLOCKED_REFERENCE_CANDIDATE"
    assert decision.reason == "missing_route_or_anchor"
    assert decision.board is None
    assert decision.dispatch_allowed is False


def test_stage0_record_only_deduplicates_idempotency_key():
    first = build_stage0_record_only_decision(
        source=_slack_source(),
        text="중복 기록 테스트",
        routing_map=_stage0_routing_map(),
    )
    second = build_stage0_record_only_decision(
        source=_slack_source(),
        text="중복 기록 테스트",
        routing_map=_stage0_routing_map(),
        existing_idempotency_keys=[first.idempotency_key],
    )

    assert first.decision == "PASS_RECORD_ONLY"
    assert second.decision == "NO_DUPLICATE"
    assert second.dispatch_allowed is False


def test_stage0_record_only_blocks_dispatch_worker_activation_request():
    decision = build_stage0_record_only_decision(
        source=_slack_source(),
        text="이 카드를 ready로 promote하고 worker dispatch 해줘",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "BLOCK_DISPATCH"
    assert decision.ready_status_allowed is False
    assert decision.dispatch_allowed is False



def test_stage0_boundary_marker_does_not_globally_allow_later_dispatch_request():
    decision = build_stage0_record_only_decision(
        source=_slack_source(),
        text="금지 범위: secret/.env 출력. 그리고 worker dispatch 해줘",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "BLOCK_DISPATCH"
    assert decision.reason == "dispatch_or_worker_activation_request"
    assert decision.dispatch_allowed is False


def test_stage0_boundary_marker_does_not_globally_allow_later_protected_request():
    decision = build_stage0_record_only_decision(
        source=_slack_source(),
        text="dry-run 검토만 한다. M1 SSH 접속해",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "BLOCK_PROTECTED"
    assert "m1" in decision.protected_terms
    assert decision.dispatch_allowed is False


def test_stage0_secret_redaction_handles_hyphenated_synthetic_tokens():
    synthetic_value = "sk-proj-" + "A" * 16
    decision = build_stage0_record_only_decision(
        source=_slack_source(),
        text=f"이 토큰을 저장해줘 {synthetic_value}",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "BLOCK_SECRET"
    assert decision.redactions_applied is True
    assert synthetic_value not in decision.sanitized_preview
    assert "[REDACTED]" in decision.sanitized_preview


def test_stage0_gateway_hook_is_after_auth_and_terminal_before_agent_dispatch():
    run_py = Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    source = run_py.read_text()
    hook_index = source.index("AI Staff OS stage0: optional record-only/fail-closed classifier")
    auth_index = source.index("elif not self._is_user_authorized(source):")
    update_prompt_index = source.index("# Intercept messages that are responses to a pending /update prompt.")
    assert auth_index < hook_index < update_prompt_index
    hook_block = source[hook_index:update_prompt_index]
    assert "event.get_command() is None" in hook_block
    assert "agent dispatch 없이 분류만 완료했습니다" in hook_block
    assert "중복 후보라 dispatch 없이 종료했습니다" in hook_block
    assert "return (" in hook_block


def test_stage0_boundary_list_header_allows_protected_terms_only_in_that_sentence():
    decision = build_stage0_record_only_decision(
        source=_slack_source(),
        text="금지 범위: M1 SSH, KCC source, DB write, secret/.env. dry-run만 진행",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "PASS_RECORD_ONLY"
    assert decision.reason == "protected_terms_only_in_boundary_text"
    assert decision.dispatch_allowed is False


def test_stage0_mixed_same_clause_boundary_and_execution_fails_closed_korean():
    decision = build_stage0_record_only_decision(
        source=_slack_source(),
        text="M1 하지 말고 DB write 해줘",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "BLOCK_PROTECTED"
    assert "db_runtime" in decision.protected_terms
    assert decision.dispatch_allowed is False


def test_stage0_mixed_same_clause_boundary_and_execution_fails_closed_english():
    decision = build_stage0_record_only_decision(
        source=_slack_source(),
        text="do not use M1, run DB migration",
        routing_map=_stage0_routing_map(),
    )

    assert decision.decision == "BLOCK_PROTECTED"
    assert "db_runtime" in decision.protected_terms
    assert decision.dispatch_allowed is False


def test_stage0_duplicate_is_helper_only_until_existing_keys_are_supplied():
    first = build_stage0_record_only_decision(
        source=_slack_source(),
        text="중복 기록 테스트 2",
        routing_map=_stage0_routing_map(),
    )
    without_existing_keys = build_stage0_record_only_decision(
        source=_slack_source(),
        text="중복 기록 테스트 2",
        routing_map=_stage0_routing_map(),
    )
    with_existing_keys = build_stage0_record_only_decision(
        source=_slack_source(),
        text="중복 기록 테스트 2",
        routing_map=_stage0_routing_map(),
        existing_idempotency_keys=[first.idempotency_key],
    )

    assert without_existing_keys.decision == "PASS_RECORD_ONLY"
    assert with_existing_keys.decision == "NO_DUPLICATE"
    assert with_existing_keys.dispatch_allowed is False
