"""Phase 2 integration tests for CodeAct mode.

Covers:
- Model profile activation for Qwen3 / Gemma4
- Direct code extraction in _invoke_tool dispatch (no re-serialise)
- Envelope format instruction in tool description
- config.yaml override (true / false / 'auto')
- build_codeact_tool_schema envelope_mode flag threading
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# get_codeact_profile — capability activation
# ---------------------------------------------------------------------------

class TestCodeActModelProfiles:
    """Verify CodeAct activation is explicit, not model-name based."""

    def _profile(self, model_name: str, **kwargs) -> dict:
        from agent.model_metadata import get_codeact_profile
        return get_codeact_profile(model_name, **kwargs)

    @pytest.mark.parametrize("model_name", [
        "qwen3-27b",
        "gemma-4-27b-it",
        "gemma3-9b",
        "nemotron3-nano-omni-30b-a3b-q8",
    ])
    def test_model_name_alone_does_not_activate_codeact(self, model_name):
        p = self._profile(model_name)
        assert p["codeact_mode"] is False
        assert p["structured_envelope"] is False
        assert p["envelope_enforcement"] == "none"

    def test_provider_capability_activates_codeact(self):
        custom = [
            {
                "base_url": "https://llm.example.invalid/v1",
                "codeact": {
                    "enabled": True,
                    "structured_envelope": True,
                    "envelope_enforcement": "grammar",
                },
                "models": {"qwen3.6-27b": {}},
            }
        ]
        p = self._profile(
            "qwen3.6-27b",
            base_url="https://llm.example.invalid/v1",
            custom_providers=custom,
        )
        assert p["codeact_mode"] is True
        assert p["structured_envelope"] is True
        assert p["envelope_enforcement"] == "grammar"

    def test_model_capability_overrides_provider_capability(self):
        custom = [
            {
                "base_url": "https://llm.example.invalid/v1",
                "codeact": {
                    "enabled": True,
                    "structured_envelope": True,
                    "envelope_enforcement": "grammar",
                },
                "models": {"plain-model": {"codeact": {"enabled": False}}},
            }
        ]
        p = self._profile(
            "plain-model",
            base_url="https://llm.example.invalid/v1",
            custom_providers=custom,
        )
        assert p["codeact_mode"] is False

    # Cloud models — must NOT activate
    @pytest.mark.parametrize("model_name", [
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "gpt-4o",
        "openai/gpt-4-turbo",
        "anthropic/claude-3.5-sonnet",
    ])
    def test_cloud_models_do_not_activate(self, model_name):
        p = self._profile(model_name)
        assert p["codeact_mode"] is False, f"Expected codeact_mode=False for {model_name!r}"

    def test_empty_model_name_returns_defaults(self):
        p = self._profile("")
        assert p["codeact_mode"] is False

    def test_unknown_model_returns_defaults(self):
        p = self._profile("totally-unknown-model-xyz")
        assert p["codeact_mode"] is False


# ---------------------------------------------------------------------------
# Tool schema — envelope instruction
# ---------------------------------------------------------------------------

class TestRunCodeSchemaEnvelope:
    def test_envelope_instruction_present_when_mode_true(self):
        from agent.codeact_tool import build_run_code_schema
        schema = build_run_code_schema("## web\n  web_search(query) — search", envelope_mode=True)
        desc = schema["description"]
        assert "FORMAT REQUIREMENT" in desc or "JSON object" in desc

    def test_envelope_instruction_absent_when_mode_false(self):
        from agent.codeact_tool import build_run_code_schema
        schema = build_run_code_schema("## web\n  web_search(query) — search", envelope_mode=False)
        desc = schema["description"]
        assert "FORMAT REQUIREMENT" not in desc

    def test_catalogue_always_present(self):
        from agent.codeact_tool import build_run_code_schema
        schema = build_run_code_schema("## web\n  web_search(query) — search", envelope_mode=True)
        assert "web_search" in schema["description"]

    def test_workflow_guidance_is_included_when_provided(self):
        from agent.codeact_tool import build_run_code_schema
        schema = build_run_code_schema(
            "## vision\n  vision_analyze(image_url, question) — analyze",
            envelope_mode=False,
            workflow_guidance="Fast CodeAct workflow rules:\n- Image/OCR/translation: call vision_analyze first.",
        )
        desc = schema["description"]
        assert "Fast workflow hints" in desc
        assert "Image/OCR/translation" in desc
        assert "vision_analyze" in desc

    def test_recipe_catalogue_is_included_when_provided(self):
        from agent.codeact_tool import build_run_code_schema

        schema = build_run_code_schema(
            "## web\n  web_search(query) — search",
            envelope_mode=False,
            recipe_catalogue="  research_web(question, max_sources=8) — gather evidence",
        )
        desc = schema["description"]
        assert "Core recipes:" in desc
        assert "research_web" in desc

    def test_research_first_call_guidance_is_explicit(self):
        from agent.codeact_tool import build_run_code_schema

        schema = build_run_code_schema("## web\n  web_search(query) — search")
        desc = schema["description"]

        assert "result = research_web(question=USER_REQUEST" in desc
        assert "result = medical_pharma_research(question=USER_REQUEST)" in desc
        assert "Do not start by debugging web_search" in desc

    def test_required_fields_unchanged(self):
        from agent.codeact_tool import build_run_code_schema
        schema = build_run_code_schema("", envelope_mode=True)
        required = schema["parameters"]["required"]
        assert "thoughts" in required
        assert "code" in required


class TestBuildCodeActToolSchemaEnvelopeThreading:
    """build_codeact_tool_schema() should pass envelope_mode through."""

    def test_envelope_mode_threaded_to_schema(self):
        from model_tools import build_codeact_tool_schema
        tool = build_codeact_tool_schema(envelope_mode=True)
        desc = tool["function"]["description"]
        assert "FORMAT REQUIREMENT" in desc or "JSON object" in desc

    def test_no_envelope_mode_no_instruction(self):
        from model_tools import build_codeact_tool_schema
        tool = build_codeact_tool_schema(envelope_mode=False)
        desc = tool["function"]["description"]
        assert "FORMAT REQUIREMENT" not in desc


# ---------------------------------------------------------------------------
# Direct dispatch — no re-serialise roundtrip
# ---------------------------------------------------------------------------

class TestDirectCodeExtraction:
    """The _invoke_tool code path must extract function_args['code'] directly."""

    def _call_invoke_tool(self, function_args: dict, kernel_execute_return: str = "ok"):
        """Simulate the run_code branch of _invoke_tool without a real AIAgent."""
        # Replicate the Phase-2 dispatch logic in isolation.
        kernel = MagicMock()
        kernel.execute.return_value = kernel_execute_return

        import json
        import logging
        logger = logging.getLogger("test")

        # Extracted logic from _invoke_tool's run_code branch:
        if isinstance(function_args, dict):
            thoughts = function_args.get("thoughts", "")
            code = function_args.get("code", "").strip()
        else:
            return json.dumps({"error": "unexpected arg type"})

        if not code:
            return json.dumps({"error": "run_code: no code provided in arguments."})

        return kernel.execute(code), kernel

    def test_extracts_code_from_dict(self):
        result, kernel = self._call_invoke_tool(
            {"thoughts": "searching", "code": "web_search(query='test')"}
        )
        kernel.execute.assert_called_once_with("web_search(query='test')")

    def test_empty_code_returns_error(self):
        result = self._call_invoke_tool({"thoughts": "oops", "code": ""})
        if isinstance(result, tuple):
            result = result[0]
        assert "error" in result

    def test_thoughts_not_executed(self):
        thoughts = "I should search for something"
        code = "x = 42"
        result, kernel = self._call_invoke_tool({"thoughts": thoughts, "code": code})
        # Only code should be passed to execute; thoughts is never executed
        kernel.execute.assert_called_once_with("x = 42")
        args = kernel.execute.call_args[0][0]
        assert thoughts not in args

    def test_code_stripped_of_whitespace(self):
        result, kernel = self._call_invoke_tool({"thoughts": "", "code": "  x = 1  "})
        kernel.execute.assert_called_once_with("x = 1")


# ---------------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------------

class TestCodeActConfigOverride:
    def _apply_override(self, model_name: str, config_value) -> dict:
        """Simulate the config override logic from run_agent.py.__init__."""
        from agent.model_metadata import get_codeact_profile
        profile = get_codeact_profile(model_name)

        # Replicate the override block:
        _override = config_value
        if _override is True:
            profile["codeact_mode"] = True
            if not profile.get("structured_envelope"):
                profile["structured_envelope"] = True
            if not profile.get("envelope_enforcement") or profile.get("envelope_enforcement") == "none":
                profile["envelope_enforcement"] = "prompt"
        elif _override is False:
            profile["codeact_mode"] = False
        # "auto" → no change

        return profile

    def test_force_on_for_cloud_model(self):
        profile = self._apply_override("claude-sonnet-4-6", True)
        assert profile["codeact_mode"] is True

    def test_force_off_for_qwen(self):
        profile = self._apply_override("qwen3.6-27b", False)
        assert profile["codeact_mode"] is False

    def test_auto_preserves_default_disabled_profile(self):
        profile = self._apply_override("qwen3.6-27b", "auto")
        assert profile["codeact_mode"] is False

    def test_auto_preserves_model_default_claude(self):
        profile = self._apply_override("claude-sonnet-4-6", "auto")
        assert profile["codeact_mode"] is False

    def test_force_on_sets_envelope_defaults(self):
        """When forcing on for a model with no profile, defaults should be set."""
        profile = self._apply_override("totally-unknown-model", True)
        assert profile["codeact_mode"] is True
        assert profile["structured_envelope"] is True
        assert profile["envelope_enforcement"] == "prompt"
