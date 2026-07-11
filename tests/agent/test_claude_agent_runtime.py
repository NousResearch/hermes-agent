from dataclasses import dataclass

from agent.claude_agent_runtime import project_claude_messages
from agent.error_classifier import FailoverReason
from claude_agent_sdk import RateLimitEvent as SDKRateLimitEvent
from claude_agent_sdk import RateLimitInfo


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-sonnet-4-6"
    error: str | None = None


@dataclass
class UserMessage:
    content: list


@dataclass
class ResultMessage:
    session_id: str
    is_error: bool = False
    result: str | None = None
    usage: dict | None = None
    api_error_status: int | None = None
    errors: list[str] | None = None


@dataclass
class RateLimitEvent:
    rate_limit_info: dict


def test_projection_preserves_kanban_tool_sequence_and_final_usage():
    projection = project_claude_messages(
        [
            AssistantMessage(
                [
                    ToolUseBlock(
                        "tool-1",
                        "mcp__hermes__kanban_complete",
                        {"summary": "done"},
                    )
                ]
            ),
            UserMessage([ToolResultBlock("tool-1", '{"success": true}')]),
            AssistantMessage([TextBlock("Completed the card.")]),
            ResultMessage(
                session_id="claude-session",
                result="Completed the card.",
                usage={"input_tokens": 120, "output_tokens": 30},
            ),
        ]
    )

    assert [message["role"] for message in projection.messages] == [
        "assistant",
        "tool",
        "assistant",
    ]
    assert projection.messages[0]["tool_calls"][0]["function"]["name"] == "kanban_complete"
    assert projection.messages[1]["tool_call_id"] == "tool-1"
    assert projection.final_text == "Completed the card."
    assert projection.session_id == "claude-session"
    assert projection.usage == {"input_tokens": 120, "output_tokens": 30}
    assert projection.failure is None


def test_rejected_rate_limit_event_exposes_reset_and_safe_fallback():
    projection = project_claude_messages(
        [
            RateLimitEvent(
                {
                    "status": "rejected",
                    "resetsAt": 1_789_000_000,
                    "rateLimitType": "five_hour",
                }
            )
        ]
    )

    assert projection.failure is not None
    assert projection.failure.reason is FailoverReason.rate_limit
    assert projection.failure.reset_at == 1_789_000_000
    assert projection.failure.replay_safe is True


def test_real_sdk_rate_limit_info_is_projected():
    projection = project_claude_messages(
        [
            SDKRateLimitEvent(
                rate_limit_info=RateLimitInfo(
                    status="rejected",
                    resets_at=1_789_000_001,
                    rate_limit_type="five_hour",
                ),
                uuid="rate-event",
                session_id="session",
            )
        ]
    )

    assert projection.failure is not None
    assert projection.failure.reason is FailoverReason.rate_limit
    assert projection.failure.reset_at == 1_789_000_001


def test_sdk_overage_is_billing_failure_even_when_primary_limit_is_allowed():
    projection = project_claude_messages(
        [
            SDKRateLimitEvent(
                rate_limit_info=RateLimitInfo(
                    status="allowed",
                    rate_limit_type="five_hour",
                    overage_status="allowed_warning",
                ),
                uuid="overage-event",
                session_id="session",
            )
        ]
    )

    assert projection.failure is not None
    assert projection.failure.reason is FailoverReason.billing


def test_real_sdk_allowed_warning_is_observed_without_forcing_fallback():
    projection = project_claude_messages(
        [
            SDKRateLimitEvent(
                rate_limit_info=RateLimitInfo(
                    status="allowed_warning",
                    rate_limit_type="seven_day",
                ),
                uuid="warning-event",
                session_id="session",
            )
        ]
    )

    assert projection.failure is None
    assert projection.warnings == ["Claude Max rate-limit warning: seven_day"]


def test_sdk_billing_and_auth_errors_are_classified_for_fallback():
    billing = project_claude_messages([AssistantMessage([], error="billing_error")])
    auth = project_claude_messages(
        [ResultMessage("session", is_error=True, api_error_status=401)]
    )

    assert billing.failure is not None
    assert billing.failure.reason is FailoverReason.billing
    assert auth.failure is not None
    assert auth.failure.reason is FailoverReason.auth


def test_failure_after_unresolved_tool_call_is_not_replay_safe():
    projection = project_claude_messages(
        [
            AssistantMessage(
                [ToolUseBlock("tool-1", "mcp__hermes__kanban_move", {"column": "Done"})]
            ),
            ResultMessage(
                "session",
                is_error=True,
                api_error_status=429,
                errors=["rate limited"],
            ),
        ]
    )

    assert projection.failure is not None
    assert projection.failure.reason is FailoverReason.rate_limit
    assert projection.failure.replay_safe is False
