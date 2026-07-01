"""
HermesAgent for tau2-bench evaluation.

Implements the tau2 HalfDuplexAgent interface using litellm with OpenRouter,
matching the inference path used across the rest of the Hermes Agent codebase.

Usage:
    python environments/benchmarks/taubench/run_eval.py \\
        --model anthropic/claude-sonnet-4-5 \\
        --base-url openrouter \\
        --env retail
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import litellm
from pydantic import BaseModel

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from environments.tool_call_parsers import get_parser

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool


class HermesAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list


class HermesAgent(HalfDuplexAgent[HermesAgentState]):
    """
    tau2 HalfDuplexAgent backed by litellm, using OpenRouter (or any
    OpenAI-compatible endpoint).

    Registered as "hermes_agent" in the tau2 registry by run_eval.py.
    """

    SYSTEM_PROMPT = (
        "You are a customer service agent that helps the user according to the "
        "<policy> provided below.\n"
        "In each turn you can either:\n"
        "- Send a message to the user.\n"
        "- Make a tool call.\n"
        "You cannot do both at the same time.\n\n"
        "Try to be helpful and always follow the policy. "
        "Always make sure you generate valid JSON only.\n\n"
        "IMPORTANT: You MUST only use the exact tool names provided to you. "
        "Do NOT invent or guess tool names. If a tool call fails, check the "
        "available tools list and use the correct name.\n\n"
        "<policy>\n{domain_policy}\n</policy>"
    )

    # System prompt variant for qwen3_coder tool format — tools are embedded
    # directly in the system prompt as <tools> XML instead of passed via the
    # OpenAI tools= parameter.
    SYSTEM_PROMPT_QWEN3_CODER = (
        "You are a customer service agent that helps the user according to the "
        "<policy> provided below.\n"
        "In each turn you can either:\n"
        "- Send a message to the user.\n"
        "- Make a tool call.\n"
        "You cannot do both at the same time.\n\n"
        "Try to be helpful and always follow the policy. "
        "Always make sure you generate valid JSON only.\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n{tools_json}\n</tools>\n\n"
        "<policy>\n{domain_policy}\n</policy>"
    )

    # System prompt variant for Hermes tool format — tools are embedded in the
    # system prompt as JSON and the model outputs tool calls as \fncall{...}\fn tags
    # instead of OpenAI tool_calls.
    # Use @@@TOOLS_JSON@@@ and @@@DOMAIN_POLICY@@@ as placeholders to avoid
    # conflicts with literal braces in the template (no .format() needed).
    SYSTEM_PROMPT_HERMES = (
        "You are a customer service agent that helps the user according to the "
        "<policy> provided below.\n"
        "In each turn you can either:\n"
        "- Send a message to the user.\n"
        "- Make a tool call.\n"
        "You cannot do both at the same time.\n\n"
        "Try to be helpful and always follow the policy. "
        "Always make sure you generate valid JSON only.\n\n"
        "You have access to the following functions:\n\n"
        "@@@TOOLS_JSON@@@\n\n"
        "To call a function, respond with a single JSON object wrapped in \fncall{...}\fn tags:\n"
        "```\ncall{\"name\": \"function_name\", \"arguments\": {\"arg\": \"value\"}}\n```\n"
        "You may make multiple calls by emitting multiple \fncall{...}\fn tags.\n\n"
        "<policy>\n@@@DOMAIN_POLICY@@@\n</policy>"
    )

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        thinking: bool = False,
        tool_parser: Optional[str] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.thinking = thinking
        self.tool_parser = tool_parser
        self._parser = get_parser(tool_parser) if tool_parser else None

        # OpenRouter requires specific headers; pass them via litellm extra_headers
        self._extra_headers: dict = {}
        if base_url and "openrouter" in base_url.lower():
            self._extra_headers = {
                "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                "X-Title": "Hermes Agent",
            }

    @property
    def system_prompt(self) -> str:
        if self.tool_parser == "qwen3_coder" and self.tools:
            tools_json = json.dumps(
                [t.openai_schema for t in self.tools], indent=2, ensure_ascii=False
            )
            # Escape braces for .format(): replace { with {{ and } with }}
            tools_json_escaped = tools_json.replace("{", "{{").replace("}", "}}")
            return self.SYSTEM_PROMPT_QWEN3_CODER.format(
                tools_json=tools_json_escaped,
                domain_policy=self.domain_policy,
            )
        if self.tool_parser == "hermes" and self.tools:
            tools_json = json.dumps(
                [t.openai_schema for t in self.tools], indent=2, ensure_ascii=False
            )
            # Use replace instead of format to avoid conflicts with literal braces
            return (self.SYSTEM_PROMPT_HERMES
                    .replace("@@@TOOLS_JSON@@@", tools_json)
                    .replace("@@@DOMAIN_POLICY@@@", self.domain_policy))
        # Default case: native OpenAI function calling (tools passed via API)
        prompt = self.SYSTEM_PROMPT.format(domain_policy=self.domain_policy)
        # Append tool signatures to reinforce correct names AND parameter names.
        # This helps weaker models that don't strictly read the tools= schema.
        if self.tools:
            sigs = []
            for t in self.tools:
                schema = t.openai_schema
                fn = schema.get("function", {})
                params = fn.get("parameters", {}).get("properties", {})
                param_str = ", ".join(f"{k}: {v.get('type', '?')}" for k, v in params.items())
                sigs.append(f"  {t.name}({param_str})")
            prompt += "\n\nAvailable tools — use ONLY these exact function and parameter names:\n" + "\n".join(sigs)
        return prompt

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> HermesAgentState:
        return HermesAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history or []),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: HermesAgentState
    ) -> tuple[AssistantMessage, HermesAgentState]:
        # Append incoming message(s) to history
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Build litellm-compatible message list
        all_messages = state.system_messages + state.messages
        lm_messages = [_to_litellm_message(m) for m in all_messages]

        kwargs = dict(
            model=self.model,
            messages=lm_messages,
            temperature=self.temperature,
        )
        # Only pass tools via OpenAI tools= parameter when NOT using hermes parser.
        # For hermes, tools are embedded in the system prompt.
        if self.tools and self.tool_parser != "hermes":
            kwargs["tools"] = [t.openai_schema for t in self.tools]
            kwargs["tool_choice"] = "auto"  # Match tau2's default behavior
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        # Enable thinking/reasoning mode. OpenRouter exposes this as
        # `include_reasoning` for nemotron (per supported_parameters in the
        # model metadata). Pass via extra_body to bypass litellm filtering.
        if self.thinking:
            kwargs["extra_body"] = {"include_reasoning": True}
        # Only pass base_url when model doesn't already have a provider prefix
        # (litellm uses either the prefix OR base_url, not both)
        if self.base_url and not self.model.startswith("openrouter/"):
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers

        response = litellm.completion(**kwargs)
        assistant_msg = _litellm_response_to_assistant_message(response, parser=self._parser)

        state.messages.append(assistant_msg)
        return assistant_msg, state


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _to_litellm_message(msg) -> dict:
    """Convert a tau2 message object to a litellm-compatible dict."""
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content or ""}

    if isinstance(msg, UserMessage):
        if msg.tool_calls:
            # User tool calls (tau2 v2 feature — user has tools too)
            return {
                "role": "user",
                "content": msg.content or "",
                "tool_calls": [_tool_call_to_dict(tc) for tc in msg.tool_calls],
            }
        return {"role": "user", "content": msg.content or ""}

    if isinstance(msg, AssistantMessage):
        d: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [_tool_call_to_dict(tc) for tc in msg.tool_calls]
        return d

    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": msg.id,
            "content": msg.content or "",
        }

    # Fallback
    return {"role": getattr(msg, "role", "user"), "content": str(getattr(msg, "content", ""))}


def _tool_call_to_dict(tc: ToolCall) -> dict:
    import json
    return {
        "id": tc.id or "call_0",
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": json.dumps(tc.arguments),
        },
    }


def _litellm_response_to_assistant_message(response, parser=None) -> AssistantMessage:
    """Convert a litellm ModelResponse to a tau2 AssistantMessage."""
    import json

    choice = response.choices[0]
    msg = choice.message

    content = msg.content or ""
    tool_calls_raw = getattr(msg, "tool_calls", None)

    tau2_tool_calls: Optional[list[ToolCall]] = None

    if parser and content:
        # Use the custom tool parser (e.g. qwen3_coder) to extract tool calls
        # from the raw text response.
        parsed_content, parsed_tool_calls = parser.parse(content)
        if parsed_tool_calls:
            content = parsed_content or ""
            tau2_tool_calls = []
            for tc in parsed_tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tau2_tool_calls.append(
                    ToolCall(
                        id=tc.id or "call_0",
                        name=tc.function.name,
                        arguments=arguments,
                        requestor="assistant",
                    )
                )
    elif tool_calls_raw:
        tau2_tool_calls = []
        for tc in tool_calls_raw:
            if hasattr(tc, "function"):
                name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tau2_tool_calls.append(
                    ToolCall(
                        id=tc.id or "call_0",
                        name=name,
                        arguments=arguments,
                        requestor="assistant",
                    )
                )

    cost = None
    try:
        cost = litellm.completion_cost(response)
    except Exception:
        pass

    usage = None
    if hasattr(response, "usage") and response.usage:
        usage = dict(response.usage)

    return AssistantMessage(
        role="assistant",
        content=content if not tau2_tool_calls else None,
        tool_calls=tau2_tool_calls,
        cost=cost,
        usage=usage,
    )


def create_hermes_agent(tools: list[Tool], domain_policy: str, **kwargs) -> HermesAgent:
    """
    Factory function registered with the tau2 registry.

    Expected kwargs:
        model (str): litellm model string
        base_url (str): API base URL (optional)
        api_key (str): API key (optional)
        temperature (float): sampling temperature (default 0.0)
        top_p (float): nucleus sampling (optional)
        max_tokens (int): max tokens (optional)
        thinking (bool): enable reasoning/thinking mode (default False)
    """
    return HermesAgent(
        tools=tools,
        domain_policy=domain_policy,
        model=kwargs["model"],
        base_url=kwargs.get("base_url"),
        api_key=kwargs.get("api_key"),
        temperature=kwargs.get("temperature", 0.0),
        top_p=kwargs.get("top_p"),
        max_tokens=kwargs.get("max_tokens"),
        thinking=kwargs.get("thinking", False),
        tool_parser=kwargs.get("tool_parser"),
    )
