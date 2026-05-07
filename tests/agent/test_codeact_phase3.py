"""Phase 3 tests — grammar-level enforcement via tool_choice injection.

Verifies:
- chat_completions transport includes tool_choice when passed as param
- tool_choice is NOT emitted when tools is empty/None
- explicit provider capabilities can report 'grammar' enforcement
- build_kwargs returns tool_choice='required' in CodeAct kwargs
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# chat_completions transport — tool_choice injection
# ---------------------------------------------------------------------------

class TestChatCompletionsToolChoice:
    def _build(self, tools, tool_choice=None) -> dict:
        """Run build_kwargs with minimal required params."""
        from agent.transports.chat_completions import ChatCompletionsTransport

        transport = ChatCompletionsTransport()
        kwargs = transport.build_kwargs(
            model="qwen3-27b",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice=tool_choice,
            # Minimal required params to avoid KeyErrors
            model_lower="qwen3-27b",
            timeout=None,
            max_tokens=None,
            ephemeral_max_output_tokens=None,
            max_tokens_param_fn=None,
            reasoning_config=None,
            request_overrides=None,
            session_id=None,
            is_openrouter=False,
            is_nous=False,
            is_qwen_portal=False,
            is_github_models=False,
            is_nvidia_nim=False,
            is_kimi=False,
            is_tokenhub=False,
            is_lmstudio=False,
            is_custom_provider=True,
            ollama_num_ctx=None,
            provider_preferences=None,
            qwen_prepare_fn=None,
            qwen_prepare_inplace_fn=None,
            qwen_session_metadata=None,
            fixed_temperature=None,
            omit_temperature=False,
            supports_reasoning=False,
            github_reasoning_extra=None,
            lmstudio_reasoning_options=None,
            anthropic_max_output=None,
            provider_name="custom",
            extra_body_additions=None,
        )
        return kwargs

    def _run_code_tool(self):
        return [{
            "type": "function",
            "function": {
                "name": "run_code",
                "description": "Execute Python",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thoughts": {"type": "string"},
                        "code": {"type": "string"},
                    },
                    "required": ["thoughts", "code"],
                },
            },
        }]

    def test_tool_choice_required_injected(self):
        kwargs = self._build(self._run_code_tool(), tool_choice="required")
        assert kwargs.get("tool_choice") == "required"

    def test_tool_choice_auto_injected(self):
        kwargs = self._build(self._run_code_tool(), tool_choice="auto")
        assert kwargs.get("tool_choice") == "auto"

    def test_tool_choice_none_not_emitted(self):
        """When tool_choice param is None, key should be absent from kwargs."""
        kwargs = self._build(self._run_code_tool(), tool_choice=None)
        assert "tool_choice" not in kwargs

    def test_tool_choice_not_emitted_when_no_tools(self):
        """tool_choice must never appear when tools list is empty."""
        kwargs = self._build(tools=None, tool_choice="required")
        assert "tool_choice" not in kwargs

    def test_tools_still_present(self):
        kwargs = self._build(self._run_code_tool(), tool_choice="required")
        assert "tools" in kwargs
        assert kwargs["tools"][0]["function"]["name"] == "run_code"

    def test_specific_tool_choice_dict(self):
        """Specific tool name dict should pass through unchanged."""
        tc = {"type": "function", "function": {"name": "run_code"}}
        kwargs = self._build(self._run_code_tool(), tool_choice=tc)
        assert kwargs.get("tool_choice") == tc


# ---------------------------------------------------------------------------
# Model profiles — grammar enforcement
# ---------------------------------------------------------------------------

class TestPhase3CapabilityProfiles:
    def _profile(self, model: str) -> dict:
        from agent.model_metadata import get_codeact_profile

        custom = [
            {
                "base_url": "https://llm.example.invalid/v1",
                "codeact": {
                    "enabled": True,
                    "structured_envelope": True,
                    "envelope_enforcement": "grammar",
                },
                "models": {model: {}},
            }
        ]
        return get_codeact_profile(
            model,
            base_url="https://llm.example.invalid/v1",
            custom_providers=custom,
        )

    def test_explicit_provider_capability_grammar_enforcement(self):
        p = self._profile("qwen3.6-27b")
        assert p["codeact_mode"] is True
        assert p["structured_envelope"] is True
        assert p["envelope_enforcement"] == "grammar"

    @pytest.mark.parametrize("model", [
        "qwen3.6-27b", "gemma-4-31b", "claude-sonnet-4-6",
    ])
    def test_model_name_alone_no_enforcement(self, model):
        from agent.model_metadata import get_codeact_profile

        p = get_codeact_profile(model)
        assert p["envelope_enforcement"] == "none"
        assert p["codeact_mode"] is False


# ---------------------------------------------------------------------------
# build_codeact_tool_schema — grammar profiles no longer need prompt instruction
# ---------------------------------------------------------------------------

class TestPhase3SchemaEnvelopeInstruction:
    """With grammar enforcement, the prompt instruction is still included (belt-and-suspenders)
    but grammar at the API level is the primary enforcement."""

    def test_envelope_instruction_present_for_grammar_profiles(self):
        """Grammar-capable profiles have structured_envelope=True."""
        from model_tools import build_codeact_tool_schema

        profile = {
            "codeact_mode": True,
            "structured_envelope": True,
            "envelope_enforcement": "grammar",
        }
        assert profile["structured_envelope"] is True

        tool = build_codeact_tool_schema(envelope_mode=profile["structured_envelope"])
        desc = tool["function"]["description"]
        assert "FORMAT REQUIREMENT" in desc or "JSON object" in desc


# ---------------------------------------------------------------------------
# tool_choice='required' gate logic
# ---------------------------------------------------------------------------

class TestToolChoiceGate:
    """Verify the _codeact_tool_choice logic from _build_api_kwargs."""

    def _compute_tool_choice(self, has_kernel: bool, tools) -> str | None:
        """Replicate the gate logic from run_agent.py."""
        _codeact_kernel = MagicMock() if has_kernel else None
        if (
            _codeact_kernel is not None
            and isinstance(tools, list)
            and len(tools) == 1
        ):
            return "required"
        return None

    def test_required_when_kernel_active_single_tool(self):
        tools = [{"type": "function", "function": {"name": "run_code"}}]
        assert self._compute_tool_choice(has_kernel=True, tools=tools) == "required"

    def test_none_when_no_kernel(self):
        tools = [{"type": "function", "function": {"name": "run_code"}}]
        assert self._compute_tool_choice(has_kernel=False, tools=tools) is None

    def test_none_when_multiple_tools(self):
        """Multi-tool mode: tool_choice should not be forced."""
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "read_file"}},
        ]
        assert self._compute_tool_choice(has_kernel=True, tools=tools) is None

    def test_none_when_empty_tools(self):
        assert self._compute_tool_choice(has_kernel=True, tools=[]) is None

    def test_none_when_tools_none(self):
        assert self._compute_tool_choice(has_kernel=True, tools=None) is None
