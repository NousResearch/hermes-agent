"""Regression tests for the SDK request-transform bypass (#93650).

``responses.create`` re-walks the whole request body against the
``ResponseCreateParams`` union graph client-side, holding the GIL. #93650
documents that walk wedging for 12+ hours on a ~1.4 MB conversation and
freezing the entire agent — no in-process watchdog can fire while the GIL
is held, and no socket kill helps a pre-network hang. Bulk wire-format
fields are therefore routed through ``extra_body``, which the SDK merges
into the JSON body *after* the transform.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from agent.codex_runtime import (
    _bound_codex_tool_image_payloads,
    _bypass_sdk_request_transform,
    _is_plain_json_data,
)


def _wire_kwargs():
    return {
        "model": "gpt-5.6-sol",
        "instructions": "You are Hermes.",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Ping"}]},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ],
        "tools": [{"type": "function", "name": "terminal", "parameters": {}}],
        "store": False,
        "stream": True,
        "timeout": 1800.0,
    }


class TestIsPlainJsonData:
    def test_accepts_nested_wire_payloads(self):
        assert _is_plain_json_data(_wire_kwargs()["input"])

    def test_rejects_non_json_leaves(self):
        assert not _is_plain_json_data([{"role": "user", "content": object()}])

    def test_rejects_non_string_dict_keys(self):
        assert not _is_plain_json_data({1: "a"})

    def test_rejects_generators(self):
        assert not _is_plain_json_data((item for item in ()))


class TestBypassSdkRequestTransform:
    def test_moves_bulk_fields_to_extra_body(self):
        kwargs = _wire_kwargs()
        original_input = kwargs["input"]

        bypassed = _bypass_sdk_request_transform(kwargs)

        assert "input" not in bypassed
        assert "tools" not in bypassed
        assert bypassed["extra_body"]["input"] is original_input
        assert bypassed["extra_body"]["tools"] == kwargs["tools"]
        # Scalar configuration stays on the typed path.
        assert bypassed["model"] == "gpt-5.6-sol"
        assert bypassed["stream"] is True
        assert bypassed["timeout"] == 1800.0
        # The caller's mapping is untouched.
        assert kwargs["input"] is original_input
        assert "extra_body" not in kwargs

    def test_merges_with_existing_extra_body_and_keeps_caller_precedence(self):
        kwargs = _wire_kwargs()
        caller_extra = {"prompt_cache_retention": "24h", "input": "explicit-wins"}
        kwargs["extra_body"] = caller_extra

        bypassed = _bypass_sdk_request_transform(kwargs)

        # An explicit extra_body entry wins, exactly as the SDK's
        # post-transform merge would have resolved the collision.
        assert bypassed["extra_body"]["input"] == "explicit-wins"
        assert bypassed["extra_body"]["prompt_cache_retention"] == "24h"
        assert bypassed["extra_body"]["tools"] == kwargs["tools"]
        assert caller_extra == {
            "prompt_cache_retention": "24h",
            "input": "explicit-wins",
        }

    def test_non_json_field_stays_on_typed_sdk_path(self):
        kwargs = _wire_kwargs()
        kwargs["input"] = [{"role": "user", "content": object()}]

        bypassed = _bypass_sdk_request_transform(kwargs)

        assert bypassed["input"] == kwargs["input"]
        assert bypassed["extra_body"] == {"tools": kwargs["tools"]}

    def test_string_input_stays_in_place(self):
        kwargs = _wire_kwargs()
        kwargs["input"] = "plain prompt"
        kwargs.pop("tools")

        bypassed = _bypass_sdk_request_transform(kwargs)

        assert bypassed is kwargs

    def test_env_escape_hatch_restores_passthrough(self, monkeypatch):
        monkeypatch.setenv("HERMES_CODEX_SDK_TRANSFORM", "1")
        kwargs = _wire_kwargs()

        assert _bypass_sdk_request_transform(kwargs) is kwargs


class TestBoundCodexToolImagePayloads:
    @staticmethod
    def _tool_output(call_id: str, image_chars: int):
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": [
                {"type": "input_text", "text": f"screenshot {call_id}"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + ("A" * image_chars),
                },
            ],
        }

    def test_multiple_screenshots_keep_newest_within_aggregate_budget(self):
        request = _wire_kwargs()
        request["input"] = [
            self._tool_output("old", 300_000),
            self._tool_output("new", 300_000),
        ]

        bounded = _bound_codex_tool_image_payloads(request)

        old_output = bounded["input"][0]["output"]
        new_output = bounded["input"][1]["output"]
        assert not any(part.get("type") == "input_image" for part in old_output)
        assert any("combined tool images exceeded" in part.get("text", "") for part in old_output)
        assert any(part.get("type") == "input_image" for part in new_output)
        # Request-local recovery must not poison or rewrite canonical history.
        assert request["input"][0]["output"][1]["type"] == "input_image"

    def test_single_oversized_stored_screenshot_is_removed_for_recovery(self):
        request = _wire_kwargs()
        request["input"] = [self._tool_output("poisoned", 760_000)]

        bounded = _bound_codex_tool_image_payloads(request)

        assert not any(
            part.get("type") == "input_image"
            for part in bounded["input"][0]["output"]
        )

    def test_under_budget_tool_images_and_user_images_are_unchanged(self):
        request = _wire_kwargs()
        user_image = {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64," + ("U" * 400_000)},
            ],
        }
        request["input"] = [user_image, self._tool_output("tool", 400_000)]

        bounded = _bound_codex_tool_image_payloads(request)

        assert bounded is request
        assert bounded["input"][0] is user_image

    def test_extra_body_input_is_bounded_when_it_is_the_effective_payload(self):
        request = _wire_kwargs()
        request.pop("input")
        request["extra_body"] = {
            "input": [
                self._tool_output("old", 300_000),
                self._tool_output("new", 300_000),
            ]
        }

        bounded = _bound_codex_tool_image_payloads(request)

        assert not any(
            part.get("type") == "input_image"
            for part in bounded["extra_body"]["input"][0]["output"]
        )
        assert any(
            part.get("type") == "input_image"
            for part in bounded["extra_body"]["input"][1]["output"]
        )


class TestRunCodexStreamRoutesPayloadViaExtraBody:
    def _make_agent(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://chatgpt.com/backend-api/codex",
            model="gpt-5.6-sol",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._interrupt_requested = False
        return agent

    def test_create_receives_input_via_extra_body(self):
        from agent.codex_runtime import run_codex_stream

        agent = self._make_agent()
        events = [
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id="r1",
                    status="completed",
                    output=[],
                    usage=None,
                ),
            )
        ]
        mock_client = MagicMock()
        mock_client.responses.create.return_value = iter(events)

        run_codex_stream(agent, _wire_kwargs(), client=mock_client)

        create_kwargs = mock_client.responses.create.call_args.kwargs
        assert "input" not in create_kwargs
        assert "tools" not in create_kwargs
        assert create_kwargs["stream"] is True
        assert create_kwargs["extra_body"]["input"][0]["role"] == "user"
        assert create_kwargs["extra_body"]["tools"][0]["name"] == "terminal"
