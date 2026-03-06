"""Inject tool definitions into system prompts for models without structured tool calling."""

import json
import os


def format_tools_for_prompt(tools: list[dict]) -> str:
    tool_defs = []
    for tool in tools:
        func = tool.get("function", {})
        tool_defs.append(json.dumps({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        }))

    tools_block = "\n".join(tool_defs)

    return (
        "You are a function-calling agent. You MUST use your tools to complete tasks.\n"
        "To call a tool, output a <tool_call> XML block with valid JSON inside:\n"
        "\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {"param": "value"}}\n'
        "</tool_call>\n"
        "\n"
        "RULES:\n"
        "- ALWAYS call tools to take action. NEVER just describe what you would do.\n"
        "- If a tool fails, try a different approach. Do NOT give up.\n"
        "- If asked to create files/directories that don't exist, CREATE them.\n"
        "- You may call multiple tools in one response.\n"
        "- After each tool call you will receive a <tool_response> with the result.\n"
        "- Only give your final answer (without tool_call tags) after completing all actions.\n"
        "\n"
        "Available tools:\n"
        "<tools>\n"
        f"{tools_block}\n"
        "</tools>"
    )


def inject_tools_into_system_prompt(system_prompt: str, tools: list[dict]) -> str:
    return system_prompt + "\n\n" + format_tools_for_prompt(tools)


def needs_tool_injection(model: str) -> bool:
    if os.environ.get("HERMES_FORCE_TOOL_INJECTION"):
        return True
    if model.startswith("local/"):
        return True
    if "qwen" in model.lower() and "openrouter" not in model.lower():
        return True
    return False


def format_tool_response(tool_call_id: str, name: str, result: str) -> str:
    return "<tool_response>" + json.dumps({"name": name, "result": result}) + "</tool_response>"
