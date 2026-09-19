"""GitHub Copilot's Anthropic Messages relay: IDE identity and signed thinking.

Copilot proxies Claude end to end over ``/v1/messages``. Two contracts must hold
on that route, and neither is exercised by the OpenAI-wire Copilot path:

* the relay authenticates the *IDE*, so requests without ``Editor-Version`` are
  rejected with HTTP 400 ``missing Editor-Version header for IDE auth``;
* the relay validates Anthropic's signed thinking blocks, so stripping them (the
  generic third-party behaviour) breaks multi-turn tool loops.
"""
import httpx
import pytest

from agent.anthropic_adapter import build_anthropic_client
from agent.anthropic_endpoints import _is_github_copilot_anthropic_endpoint
from agent.anthropic_message_convert import convert_messages_to_anthropic

COPILOT_BASE = "https://api.enterprise.githubcopilot.com"


class TestCopilotEndpointDetection:
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.githubcopilot.com",
            COPILOT_BASE,
            "https://api.business.githubcopilot.com/v1",
        ],
    )
    def test_recognizes_copilot_hosts(self, base_url):
        assert _is_github_copilot_anthropic_endpoint(base_url) is True

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.anthropic.com",
            "https://githubcopilot.com.attacker.test",
            "https://notgithubcopilot.com",
            None,
            "",
        ],
    )
    def test_rejects_other_and_lookalike_hosts(self, base_url):
        assert _is_github_copilot_anthropic_endpoint(base_url) is False


class TestCopilotIdeIdentityHeaders:
    """The IDE identity must reach the wire, not merely the client kwargs."""

    @staticmethod
    def _captured_request(base_url):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-test",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        client = build_anthropic_client("test-token", base_url=base_url)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        client.messages.create(
            model="claude-opus-test",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
        return captured["request"]

    def test_editor_version_reaches_the_wire(self):
        request = self._captured_request(COPILOT_BASE)
        # Case-insensitive: httpx.Headers normalizes, the relay does not care.
        assert request.headers.get("editor-version")

    def test_copilot_integration_id_reaches_the_wire(self):
        request = self._captured_request(COPILOT_BASE)
        assert request.headers.get("copilot-integration-id")

    def test_non_copilot_endpoint_does_not_get_ide_headers(self):
        request = self._captured_request("https://api.anthropic.com")
        assert request.headers.get("editor-version") is None
        assert request.headers.get("copilot-integration-id") is None


class TestCopilotSignedThinkingReplay:
    """Copilot validates signed thinking; the third-party strip would break it."""

    @staticmethod
    def _signed_tool_turn():
        thinking_block = {
            "type": "thinking",
            "thinking": "Need the exact contents.",
            "signature": "signed-thinking",
        }
        return [
            {"role": "user", "content": "Inspect the file"},
            {
                "role": "assistant",
                "content": "I'll inspect it.",
                "reasoning_details": [thinking_block],
                "anthropic_content_blocks": [
                    thinking_block,
                    {"type": "text", "text": "I'll inspect it."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "a.txt"},
                    },
                ],
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "name": "read_file",
                "content": "contents",
            },
        ]

    def _assistant_blocks(self, base_url):
        _, converted = convert_messages_to_anthropic(
            self._signed_tool_turn(), base_url=base_url, model="claude-opus-test"
        )
        assistant = next(m for m in converted if m["role"] == "assistant")
        return assistant["content"]

    def test_copilot_preserves_the_signed_thinking_block(self):
        blocks = self._assistant_blocks(COPILOT_BASE)
        thinking = [b for b in blocks if b.get("type") == "thinking"]
        assert len(thinking) == 1
        assert thinking[0]["signature"] == "signed-thinking"

    def test_copilot_keeps_the_tool_use_block_alongside_thinking(self):
        blocks = self._assistant_blocks(COPILOT_BASE)
        assert [b.get("type") for b in blocks] == ["thinking", "text", "tool_use"]

    def test_generic_third_party_still_strips_signed_thinking(self):
        """Contrast case: the exemption must be Copilot-specific, not a blanket change."""
        blocks = self._assistant_blocks("https://api.some-third-party.test")
        assert not [b for b in blocks if b.get("type") == "thinking"]
