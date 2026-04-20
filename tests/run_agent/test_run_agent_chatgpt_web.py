import json
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


DEFAULT_WEB_BASE = "https://chatgpt.com/backend-api/f"


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _build_agent(monkeypatch, *, model="gpt-5-thinking"):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model=model,
        provider="chatgpt-web",
        api_mode="chatgpt_web",
        base_url=DEFAULT_WEB_BASE,
        api_key="chatgpt-web-token",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._save_session_log = lambda messages: None
    return agent


def test_build_api_kwargs_chatgpt_web_carries_thread_state(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent._chatgpt_web_conversation_id = "conv_existing"
    agent._chatgpt_web_parent_message_id = "msg_existing"

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello from Hermes."},
    ])

    assert kwargs["model"] == "gpt-5-thinking"
    assert kwargs["conversation_id"] == "conv_existing"
    assert kwargs["parent_message_id"] == "msg_existing"
    assert kwargs["messages"][-1]["content"] == "Hello from Hermes."
    assert "tools" not in kwargs


def test_agent_init_prefers_runtime_chatgpt_web_state_over_pool(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)

    class _Pool:
        def has_credentials(self):
            return True

        def select(self):
            return SimpleNamespace(
                session_token="stale-session",
                cookie_header="stale-cookie=1",
                browser_cookies=[{"name": "stale", "value": "1"}],
                user_agent="Mozilla/Stale",
                device_id="stale-device",
            )

        def peek(self):
            return self.select()

        def entries(self):
            return [self.select()]

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: _Pool())

    agent = run_agent.AIAgent(
        model="gpt-5-thinking",
        provider="chatgpt-web",
        api_mode="chatgpt_web",
        base_url=DEFAULT_WEB_BASE,
        api_key="chatgpt-web-token",
        chatgpt_web_session_token="runtime-session",
        chatgpt_web_cookie_header="runtime-cookie=1",
        chatgpt_web_browser_cookies=[{"name": "runtime", "value": "1"}],
        chatgpt_web_user_agent="Mozilla/Runtime",
        chatgpt_web_device_id="runtime-device",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    assert agent._chatgpt_web_session_token == "runtime-session"
    assert agent._chatgpt_web_cookie_header == "runtime-cookie=1"
    assert agent._chatgpt_web_browser_cookies == [{"name": "runtime", "value": "1"}]
    assert agent._chatgpt_web_user_agent == "Mozilla/Runtime"
    assert agent._chatgpt_web_device_id == "runtime-device"


def test_build_api_kwargs_chatgpt_web_uses_latest_user_turn_for_tool_selection(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "execute_code", "description": "Run Python code", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What time is it?"},
        {"role": "assistant", "content": "I can check that."},
        {"role": "user", "content": "Use Hermes tools to run Python that prints 6*7. Answer only the result."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: execute_code.' in rewritten_user
    assert '"name": "execute_code"' in rewritten_user
    assert '"code": "print(6*7)"' in rewritten_user


def test_conversation_turn_preview_handles_multimodal_messages(monkeypatch):
    agent = _build_agent(monkeypatch)

    preview = agent._conversation_turn_preview([
        {"type": "text", "text": "Look at this local image."},
        {"type": "input_image", "image_url": "C:/tmp/red-square.png"},
    ])

    assert "Look at this local image." in preview
    assert "[image:C:/tmp/red-square.png]" in preview


def test_select_chatgpt_web_tools_prefers_explicit_sequence(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "execute_code", "description": "Run Python code", "parameters": {"type": "object"}}},
    ]

    first = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "First use search_files, then terminal, then execute_code."},
    ])
    second = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "First use search_files, then terminal, then execute_code."},
        {"role": "tool", "tool_name": "search_files", "content": "{}"},
    ])
    third = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "First use search_files, then terminal, then execute_code."},
        {"role": "tool", "tool_name": "search_files", "content": "{}"},
        {"role": "tool", "tool_name": "terminal", "content": "{}"},
    ])

    assert [tool["function"]["name"] for tool in first] == ["search_files"]
    assert [tool["function"]["name"] for tool in second] == ["terminal"]
    assert [tool["function"]["name"] for tool in third] == ["execute_code"]



def test_select_chatgpt_web_tools_prefers_terminal_for_working_directory(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "Use Hermes tools to print the current working directory. Answer only the path."},
    ])

    assert [tool["function"]["name"] for tool in selected] == ["terminal"]



def test_select_chatgpt_web_tools_does_not_confuse_memory_usage_with_memory_tool(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "memory", "description": "Store memory", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "What are the top processes on this machine by memory usage? Use terminal and answer with the top process name only."},
    ])

    assert [tool["function"]["name"] for tool in selected] == ["terminal"]


def test_select_chatgpt_web_tools_skips_plain_greeting(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "hello welcome to termux"},
    ])

    assert selected == []



def test_build_api_kwargs_chatgpt_web_skips_local_tool_loop_for_plain_greeting(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent._chatgpt_web_conversation_id = "conv_existing"
    agent._chatgpt_web_parent_message_id = "msg_existing"
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hello welcome to termux"},
    ])

    assert kwargs["history_and_training_disabled"] is False
    assert kwargs["conversation_id"] == "conv_existing"
    assert kwargs["parent_message_id"] == "msg_existing"
    assert kwargs["messages"][-1]["content"] == "hello welcome to termux"
    assert "<tool_call>" not in kwargs["instructions"]


def test_build_api_kwargs_chatgpt_web_includes_richer_hermes_intro(monkeypatch):
    agent = _build_agent(monkeypatch)

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hello welcome to termux"},
    ])

    assert "Hermes Agent web-model runtime" in kwargs["instructions"]
    assert "Skills are first-class Hermes artifacts" in kwargs["instructions"]
    assert "Do not ask for permission to continue" in kwargs["instructions"]


def test_chatgpt_web_build_skill_content_returns_reusable_template(monkeypatch):
    agent = _build_agent(monkeypatch)

    content = agent._chatgpt_web_build_skill_content(
        "chatgpt-web-e2e-temp-skill",
        "How to say hello and verify the output",
    )

    assert "## Purpose" in content
    assert "## Workflow" in content
    assert "## Validation" in content
    assert "## Pitfalls" in content



def test_build_api_kwargs_chatgpt_web_prefers_terminal_for_platform_details(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "execute command to check platform details then tell me what system you are running on"},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert kwargs["history_and_training_disabled"] is True
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert '"command": "uname -a"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_prefers_terminal_for_whoami(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Try terminal tool and check whoami on it"},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert kwargs["history_and_training_disabled"] is True
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert '"command": "whoami"' in rewritten_user
    assert "Do not answer the user yet." in rewritten_user

def test_build_api_kwargs_chatgpt_web_prefers_terminal_for_path_exists_check(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "In a fresh chat, check whether /tmp/hermes-web-soak/repo/.git exists and answer only yes or no. Do not reclone anything."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert '"command": "[ -e \'/tmp/hermes-web-soak/repo/.git\' ] && echo yes || echo no"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_prefers_terminal_for_top_processes(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Use terminal and show the top processes using the most memory. Answer briefly."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert '"command": "ps aux --sort=-%mem | head -n 10"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_prefers_combined_terminal_command_for_whoami_pwd_topproc(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                "Use terminal to run whoami, then use terminal to run pwd, then use terminal to show the top "
                "processes using the most memory. Keep going automatically until the task is complete. "
                "Final answer exactly as three lines: USER=<username> PWD=<path> TOPPROC=<first process>."
            ),
        },
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert '"command": "whoami && pwd && ps aux --sort=-%mem | head -n 2"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_prefers_cronjob_for_simple_schedule(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "cronjob", "description": "Manage cron jobs", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": "Create a cron job named disk-check every 1h to use terminal to run df -h and report disk usage.",
        },
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: cronjob.' in rewritten_user
    assert '"action": "create"' in rewritten_user
    assert '"name": "disk-check"' in rewritten_user
    assert '"schedule": "every 1h"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_prefers_cronjob_over_nested_terminal_phrase(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "cronjob", "description": "Manage cron jobs", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": "Create a cron job named hermes-live-soak-20260420 every 1h to use terminal to run date and report it. Answer only created.",
        },
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: cronjob.' in rewritten_user
    assert '"action": "create"' in rewritten_user
    assert '"name": "hermes-live-soak-20260420"' in rewritten_user
    assert '"prompt": "use terminal to run date and report it"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_terminal_clone_prefills_args(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                "Use the terminal tool to clone https://github.com/NousResearch/hermes-agent.git "
                "with depth 1 into /tmp/hermes-soak and then tell me the branch."
            ),
        },
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert kwargs["history_and_training_disabled"] is True
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert 'Use these exact arguments for this turn:' in rewritten_user
    assert '"command": "git clone --depth 1' in rewritten_user
    assert "/tmp/hermes-soak" in rewritten_user


def test_select_chatgpt_web_tools_picks_terminal_for_natural_language_clone_request(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {
            "role": "user",
            "content": (
                "Clone the GitHub repository https://github.com/octocat/Hello-World.git with depth 1 into "
                "C:/Users/adyba/AppData/Local/Temp/hermes-live-soak-20260420-b/workspace/octocat-hello-world. "
                "Keep going automatically until the clone is complete, then answer only with the exact repo path."
            ),
        },
    ])

    assert [tool["function"]["name"] for tool in selected] == ["terminal"]


def test_build_api_kwargs_chatgpt_web_natural_language_clone_prefills_terminal_args(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                "Clone the GitHub repository https://github.com/octocat/Hello-World.git with depth 1 into "
                "C:/Users/adyba/AppData/Local/Temp/hermes-live-soak-20260420-b/workspace/octocat-hello-world. "
                "Keep going automatically until the clone is complete, then answer only with the exact repo path."
            ),
        },
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert 'Use these exact arguments for this turn:' in rewritten_user
    assert '"command": "git clone --depth 1' in rewritten_user
    assert "C:/Users/adyba/AppData/Local/Temp/hermes-live-soak-20260420-b/workspace/octocat-hello-world" in rewritten_user


def test_build_api_kwargs_chatgpt_web_prefers_terminal_for_path_exists_check(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "In a fresh chat, check whether /tmp/hermes-web-soak/repo/.git exists and answer only yes or no. Do not reclone anything."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert '"command": "[ -e \'/tmp/hermes-web-soak/repo/.git\' ] && echo yes || echo no"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_prefers_terminal_for_top_processes(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Use terminal and show the top processes using the most memory. Answer briefly."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert '"command": "ps aux --sort=-%mem | head -n 10"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_prefers_combined_terminal_command_for_whoami_pwd_topproc(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                "Use terminal to run whoami, then use terminal to run pwd, then use terminal to show the top "
                "processes using the most memory. Keep going automatically until the task is complete. "
                "Final answer exactly as three lines: USER=<username> PWD=<path> TOPPROC=<first process>."
            ),
        },
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert '"command": "whoami && pwd && ps aux --sort=-%mem | head -n 2"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_prefers_cronjob_for_simple_schedule(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "cronjob", "description": "Manage cron jobs", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": "Create a cron job named disk-check every 1h to use terminal to run df -h and report disk usage.",
        },
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: cronjob.' in rewritten_user
    assert '"action": "create"' in rewritten_user
    assert '"name": "disk-check"' in rewritten_user
    assert '"schedule": "every 1h"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_prefers_cronjob_over_nested_terminal_phrase(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "cronjob", "description": "Manage cron jobs", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": "Create a cron job named hermes-live-soak-20260420 every 1h to use terminal to run date and report it. Answer only created.",
        },
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: cronjob.' in rewritten_user
    assert '"action": "create"' in rewritten_user
    assert '"name": "hermes-live-soak-20260420"' in rewritten_user
    assert '"prompt": "use terminal to run date and report it"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_terminal_clone_prefills_args(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                "Use the terminal tool to clone https://github.com/NousResearch/hermes-agent.git "
                "with depth 1 into /tmp/hermes-soak and then tell me the branch."
            ),
        },
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert kwargs["history_and_training_disabled"] is True
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert 'Use these exact arguments for this turn:' in rewritten_user
    assert '"command": "git clone --depth 1' in rewritten_user
    assert "/tmp/hermes-soak" in rewritten_user


def test_select_chatgpt_web_tools_picks_terminal_for_natural_language_clone_request(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {
            "role": "user",
            "content": (
                "Clone the GitHub repository https://github.com/octocat/Hello-World.git with depth 1 into "
                "C:/Users/adyba/AppData/Local/Temp/hermes-live-soak-20260420-b/workspace/octocat-hello-world. "
                "Keep going automatically until the clone is complete, then answer only with the exact repo path."
            ),
        },
    ])

    assert [tool["function"]["name"] for tool in selected] == ["terminal"]


def test_build_api_kwargs_chatgpt_web_natural_language_clone_prefills_terminal_args(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                "Clone the GitHub repository https://github.com/octocat/Hello-World.git with depth 1 into "
                "C:/Users/adyba/AppData/Local/Temp/hermes-live-soak-20260420-b/workspace/octocat-hello-world. "
                "Keep going automatically until the clone is complete, then answer only with the exact repo path."
            ),
        },
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert 'Use these exact arguments for this turn:' in rewritten_user
    assert '"command": "git clone --depth 1' in rewritten_user
    assert "C:/Users/adyba/AppData/Local/Temp/hermes-live-soak-20260420-b/workspace/octocat-hello-world" in rewritten_user


def test_wrap_chatgpt_web_response_synthesizes_terminal_call_for_whoami(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Try terminal tool and check whoami on it"},
    ])

    wrapped = agent._wrap_chatgpt_web_response({
        "content": "It seems like there's an issue with accessing the terminal tool right now.",
        "finish_reason": "stop",
    })

    tool_calls = wrapped.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.name == "terminal"
    assert json.loads(tool_calls[0].function.arguments) == {"command": "whoami"}
    assert wrapped.choices[0].message.content == ""


def test_wrap_chatgpt_web_response_infers_followup_tool_call_from_model_prose(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Use terminal and show the top processes using the most memory. Answer briefly."},
    ])

    wrapped = agent._wrap_chatgpt_web_response({
        "content": "I can continue by checking the top processes if you want.",
        "finish_reason": "stop",
    })

    tool_calls = wrapped.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.name == "terminal"
    assert json.loads(tool_calls[0].function.arguments) == {"command": "ps aux --sort=-%mem | head -n 10"}
    assert wrapped.choices[0].message.content == ""


def test_select_chatgpt_web_tools_stops_after_terminal_output_already_contains_top_process(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "What are the top processes on this machine by memory usage? Use terminal and answer with the top process name only."},
        {
            "role": "tool",
            "tool_name": "terminal",
            "content": json.dumps({
                "output": (
                    "your 131072x1 screen size is bogus. expect trouble\n"
                    "USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
                    "root         377  0.0  0.2 2256636 68428 ?       Sl   16:02   0:02 "
                    "/snap/ubuntu-desktop-installer/1286/usr/bin/python3.10 -m subiquity.cmd.server"
                ),
                "exit_code": 0,
                "error": None,
            }),
        },
    ])

    assert selected == []


def test_repair_terminal_completion_response_uses_successful_tool_output_instead_of_permission_prompt(monkeypatch):
    agent = _build_agent(monkeypatch)

    repaired = agent._chatgpt_web_repair_terminal_completion_response(
        "What are the top processes on this machine by memory usage? Use terminal and answer with the top process name only.",
        "Would you like me to try again later or assist you in another way?",
        [
            {
                "role": "tool",
                "content": json.dumps({
                    "output": (
                        "your 131072x1 screen size is bogus. expect trouble\n"
                        "USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
                        "root         377  0.0  0.2 2256636 68428 ?       Sl   16:02   0:02 "
                        "/snap/ubuntu-desktop-installer/1286/usr/bin/python3.10 -m subiquity.cmd.server"
                    ),
                    "exit_code": 0,
                    "error": None,
                }),
            },
        ],
    )

    assert repaired == "python3.10"


def test_repair_terminal_completion_response_formats_three_line_terminal_summary(monkeypatch):
    agent = _build_agent(monkeypatch)

    repaired = agent._chatgpt_web_repair_terminal_completion_response(
        (
            "Use terminal to run whoami, then use terminal to run pwd, then use terminal to show the top "
            "processes using the most memory. Keep going automatically until the task is complete. "
            "Do not ask permission again. Final answer exactly as three lines: "
            "USER=<username> PWD=<path> TOPPROC=<first process>."
        ),
        "I can continue by checking the top processes if you want.",
        [
            {
                "role": "tool",
                "content": json.dumps({
                    "output": (
                        "tdamre\n"
                        "/mnt/c/Users/adyba/AppData/Local/Temp/hermes-termux-rebase\n"
                        "your 131072x1 screen size is bogus. expect trouble\n"
                        "USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
                        "root         377  0.0  0.2 2256636 68428 ?       Sl   16:02   0:02 "
                        "/snap/ubuntu-desktop-installer/1286/usr/bin/python3.10 -m subiquity.cmd.server"
                    ),
                    "exit_code": 0,
                    "error": None,
                }),
            },
        ],
    )

    assert repaired == (
        "USER=tdamre\n"
        "PWD=/mnt/c/Users/adyba/AppData/Local/Temp/hermes-termux-rebase\n"
        "TOPPROC=python3.10"
    )


def test_repair_terminal_completion_response_uses_yes_no_for_path_exists(monkeypatch):
    agent = _build_agent(monkeypatch)

    repaired = agent._chatgpt_web_repair_terminal_completion_response(
        "In a fresh chat, check whether /tmp/hermes-web-soak/repo/.git exists and answer only yes or no. Do not reclone anything.",
        "It seems that I cannot proceed with the requested tool at this moment.",
        [
            {
                "role": "tool",
                "content": json.dumps({
                    "output": "yes\n",
                    "exit_code": 0,
                    "error": None,
                }),
            },
        ],
    )

    assert repaired == "yes"


def test_select_chatgpt_web_tools_stops_after_terminal_output_already_contains_whoami_result(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {"role": "user", "content": "Try terminal tool and check whoami on it. Keep going automatically until the task is complete. Answer only the result."},
        {
            "role": "tool",
            "tool_name": "terminal",
            "content": json.dumps({
                "output": "ubuntu\n",
                "exit_code": 0,
                "error": None,
            }),
        },
    ])

    assert selected == []


def test_repair_terminal_completion_response_uses_whoami_output_for_result_mode(monkeypatch):
    agent = _build_agent(monkeypatch)

    repaired = agent._chatgpt_web_repair_terminal_completion_response(
        "Try terminal tool and check whoami on it. Keep going automatically until the task is complete. Answer only the result.",
        "The system user is \"oai\".",
        [
            {
                "role": "tool",
                "content": json.dumps({
                    "output": "ubuntu\n",
                    "exit_code": 0,
                    "error": None,
                }),
            },
        ],
    )

    assert repaired == "ubuntu"


def test_repair_terminal_completion_response_prefers_exact_repo_path_from_tool_output(monkeypatch):
    agent = _build_agent(monkeypatch)

    repaired = agent._chatgpt_web_repair_terminal_completion_response(
        (
            "Clone the GitHub repository https://github.com/octocat/Hello-World.git with depth 1 into "
            "/tmp/hermes-web-soak/hello-world. Keep going automatically until the clone is complete, "
            "then answer only with the exact repo path."
        ),
        "/tm/hermes-web-soak/hello-world",
        [
            {
                "role": "tool",
                "content": json.dumps({
                    "output": "Cloning into '/tmp/hermes-web-soak/hello-world'...\n",
                    "exit_code": 0,
                    "error": None,
                }),
            },
        ],
    )

    assert repaired == "/tmp/hermes-web-soak/hello-world"


def test_repair_answer_only_response_uses_verified_when_read_back_contains_marker(monkeypatch):
    agent = _build_agent(monkeypatch)

    repaired = agent._chatgpt_web_repair_answer_only_response(
        (
            "Use read_file to inspect /tmp/hermes-web-soak/repo/README, then use patch to append the exact text "
            "LIVE_PATCH_MARKER at the end of that file, then use read_file again to verify the marker is present. "
            "Keep going automatically until the task is complete. Answer only verified."
        ),
        "There was an issue with tool usage, but I can proceed.",
        [
            {
                "role": "tool",
                "content": json.dumps({
                    "content": "1|Hello World!\n2|LIVE_PATCH_MARKER\n",
                    "truncated": False,
                }),
            },
        ],
    )

    assert repaired == "verified"


def test_chatgpt_web_tool_args_infers_patch_append_replace(monkeypatch):
    agent = _build_agent(monkeypatch)

    args = agent._chatgpt_web_tool_args(
        "patch",
        [
            {
                "role": "user",
                "content": (
                    "Use read_file to inspect /tmp/hermes-web-soak/repo/README, then use patch to append the exact text "
                    "LIVE_PATCH_MARKER at the end of that file, then use read_file again to verify the marker is present. "
                    "Keep going automatically until the task is complete. Answer only verified."
                ),
            },
            {
                "role": "tool",
                "content": json.dumps({
                    "content": "     1|Hello World!\n     2|\n",
                    "truncated": False,
                }),
            },
        ],
    )

    assert args == {
        "mode": "replace",
        "path": "/tmp/hermes-web-soak/repo/README",
        "old_string": "Hello World!\n",
        "new_string": "Hello World!\nLIVE_PATCH_MARKER\n",
    }


def test_chatgpt_web_tool_args_infers_browser_sequence_args(monkeypatch):
    agent = _build_agent(monkeypatch)
    payload_messages = [
        {
            "role": "user",
            "content": (
                "Use terminal to get the current branch of /tmp/hermes-web-soak/hello-world, then use browser_navigate "
                "and browser_vision to open https://en.wikipedia.org/wiki/OpenAI and read the visible page title from "
                "the screenshot. Keep going automatically until the task is complete."
            ),
        },
        {
            "role": "tool",
            "content": json.dumps({"output": "master\n", "exit_code": 0, "error": None}),
        },
    ]

    navigate_args = agent._chatgpt_web_tool_args("browser_navigate", payload_messages)
    vision_args = agent._chatgpt_web_tool_args("browser_vision", payload_messages)

    assert navigate_args == {"url": "https://en.wikipedia.org/wiki/OpenAI"}
    assert vision_args == {"question": "What is the visible page title text in the screenshot?"}


def test_chatgpt_web_tool_args_infers_followup_terminal_verify_step(monkeypatch):
    agent = _build_agent(monkeypatch)

    args = agent._chatgpt_web_tool_args(
        "terminal",
        [
            {
                "role": "user",
                "content": (
                    "Use terminal to append the exact text LIVE_SOAK_MARKER to /tmp/hermes-web-soak/repo/README, "
                    "then use terminal to verify the marker exists. Keep going automatically until the task is complete "
                    "and answer only verified."
                ),
            },
            {
                "role": "tool",
                "content": json.dumps({
                    "output": "",
                    "exit_code": 0,
                    "error": None,
                }),
            },
        ],
    )

    assert args == {
        "command": "grep -Fqx -- 'LIVE_SOAK_MARKER' '/tmp/hermes-web-soak/repo/README' && echo verified || echo missing"
    }


def test_chatgpt_web_tool_args_infers_followup_terminal_branch_step(monkeypatch):
    agent = _build_agent(monkeypatch)

    args = agent._chatgpt_web_tool_args(
        "terminal",
        [
            {
                "role": "user",
                "content": (
                    "Use terminal to clone https://github.com/octocat/Hello-World.git with depth 1 into "
                    "/tmp/hermes-web-soak/hello-world. After cloning, use terminal to print the current branch. "
                    "Keep going automatically until the task is complete. Answer only the branch name."
                ),
            },
            {
                "role": "tool",
                "content": json.dumps({
                    "output": "Cloning into '/tmp/hermes-web-soak/hello-world'...\n",
                    "exit_code": 0,
                    "error": None,
                }),
            },
        ],
    )

    assert args == {
        "command": "git -C '/tmp/hermes-web-soak/hello-world' rev-parse --abbrev-ref HEAD"
    }


def test_chatgpt_web_tool_args_infers_create_then_list_for_cron_sequence(monkeypatch):
    agent = _build_agent(monkeypatch)

    create_args = agent._chatgpt_web_tool_args(
        "cronjob",
        [
            {
                "role": "user",
                "content": (
                    "Create a cron job named hermes-live-soak every 1h to use terminal to run date, "
                    "then list jobs. Keep going automatically until the task is complete. "
                    "Answer only created."
                ),
            },
        ],
    )

    list_args = agent._chatgpt_web_tool_args(
        "cronjob",
        [
            {
                "role": "user",
                "content": (
                    "Create a cron job named hermes-live-soak every 1h to use terminal to run date, "
                    "then list jobs. Keep going automatically until the task is complete."
                ),
            },
            {
                "role": "tool",
                "content": json.dumps({
                    "success": True,
                    "message": "Cron job hermes-live-soak created.",
                }),
            },
        ],
    )

    assert create_args == {
        "action": "create",
        "name": "hermes-live-soak",
        "schedule": "every 1h",
        "prompt": "use terminal to run date",
    }
    assert list_args == {"action": "list"}


def test_chatgpt_web_tool_args_infers_remove_for_cron_delete_request(monkeypatch):
    agent = _build_agent(monkeypatch)

    args = agent._chatgpt_web_tool_args(
        "cronjob",
        [
            {
                "role": "user",
                "content": "Remove the cron job named hermes-live-soak. Answer only removed.",
            },
        ],
    )

    assert args == {"action": "list"}


def test_chatgpt_web_tool_args_infers_remove_job_id_after_cron_list(monkeypatch):
    agent = _build_agent(monkeypatch)

    args = agent._chatgpt_web_tool_args(
        "cronjob",
        [
            {
                "role": "user",
                "content": "Remove the cron job named hermes-live-soak. Keep going automatically until the task is complete.",
            },
            {
                "role": "tool",
                "content": json.dumps({
                    "success": True,
                    "count": 2,
                    "jobs": [
                        {"id": "job-1", "name": "other-job"},
                        {"id": "job-2", "name": "hermes-live-soak"},
                    ],
                }),
            },
        ],
    )

    assert args == {"action": "remove", "job_id": "job-2"}


def test_select_chatgpt_web_tools_routes_natural_multi_tool_sequence(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "browser_navigate", "description": "Navigate browser", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "browser_vision", "description": "Analyze browser screenshot", "parameters": {"type": "object"}}},
    ]

    user_content = (
        "In a fresh chat, confirm whether /tmp/hermes-web-soak/hello-world exists, then check whether "
        "LIVE_SOAK_MARKER exists in the repo README, then open https://en.wikipedia.org/wiki/OpenAI and "
        "read the visible title from the screenshot. Keep going automatically until the task is complete."
    )

    selected_initial = agent._select_chatgpt_web_tools([
        {"role": "user", "content": user_content},
    ])
    selected_after_terminal = agent._select_chatgpt_web_tools([
        {"role": "user", "content": user_content},
        {
            "role": "tool",
            "tool_name": "terminal",
            "content": json.dumps({"output": "yes\n", "exit_code": 0, "error": None}),
        },
    ])
    selected_after_search = agent._select_chatgpt_web_tools([
        {"role": "user", "content": user_content},
        {
            "role": "tool",
            "tool_name": "terminal",
            "content": json.dumps({"output": "yes\n", "exit_code": 0, "error": None}),
        },
        {
            "role": "tool",
            "tool_name": "search_files",
            "content": json.dumps({"matches": [{"path": "/tmp/hermes-web-soak/hello-world/README"}], "total_count": 1}),
        },
    ])
    selected_after_navigate = agent._select_chatgpt_web_tools([
        {"role": "user", "content": user_content},
        {
            "role": "tool",
            "tool_name": "terminal",
            "content": json.dumps({"output": "yes\n", "exit_code": 0, "error": None}),
        },
        {
            "role": "tool",
            "tool_name": "search_files",
            "content": json.dumps({"matches": [{"path": "/tmp/hermes-web-soak/hello-world/README"}], "total_count": 1}),
        },
        {
            "role": "tool",
            "tool_name": "browser_navigate",
            "content": json.dumps({"success": True, "url": "https://en.wikipedia.org/wiki/OpenAI", "title": "OpenAI"}),
        },
    ])

    assert [tool["function"]["name"] for tool in selected_initial] == ["terminal"]
    assert [tool["function"]["name"] for tool in selected_after_terminal] == ["search_files"]
    assert [tool["function"]["name"] for tool in selected_after_search] == ["browser_navigate"]
    assert [tool["function"]["name"] for tool in selected_after_navigate] == ["browser_vision"]


def test_select_chatgpt_web_tools_does_not_treat_patch_or_terminal_as_forced_patch(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "patch", "description": "Patch files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {
            "role": "user",
            "content": (
                "Inspect /tmp/hermes-web-soak/hello-world/README, then append the exact text LIVE_SOAK_MARKER "
                "to the end of that file using patch or terminal, then verify the marker is present. "
                "Keep going automatically until the task is complete. Answer only verified."
            ),
        },
    ])

    assert [tool["function"]["name"] for tool in selected] == ["read_file"]


def test_select_chatgpt_web_tools_routes_patch_followup_to_terminal_verify(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "patch", "description": "Patch files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {
            "role": "user",
            "content": (
                "Inspect /tmp/hermes-web-soak/hello-world/README, then append the exact text LIVE_SOAK_MARKER "
                "to the end of that file using patch or terminal, then verify the marker is present. "
                "Keep going automatically until the task is complete. Answer only verified."
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "patch",
                        "arguments": "{\"mode\":\"patch\",\"path\":\"/tmp/hermes-web-soak/hello-world/README\",\"patch\":\"\\nLIVE_SOAK_MARKER\"}",
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps({"success": True}),
        },
    ])

    assert [tool["function"]["name"] for tool in selected] == ["terminal"]


def test_chatgpt_web_tool_args_infers_initial_terminal_branch_step_for_repo_path(monkeypatch):
    agent = _build_agent(monkeypatch)

    args = agent._chatgpt_web_tool_args(
        "terminal",
        [
            {
                "role": "user",
                "content": (
                    "Use terminal to get the current branch of /tmp/hermes-web-soak/hello-world, then use browser_navigate "
                    "and browser_vision to open https://en.wikipedia.org/wiki/OpenAI and read the visible page title from "
                    "the screenshot. Keep going automatically until the task is complete and do not ask for permission "
                    "again. Answer exactly these three lines and nothing else: BRANCH=<branch>, TITLE=<title>, STATUS=ready."
                ),
            },
        ],
    )

    assert args == {
        "command": "git -C '/tmp/hermes-web-soak/hello-world' rev-parse --abbrev-ref HEAD"
    }


def test_wrap_chatgpt_web_response_infers_followup_tool_call_from_saved_turn_context(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]
    agent._chatgpt_web_selected_tool_names = ["terminal"]
    agent._chatgpt_web_selected_tool_payload_messages = [
        {
            "role": "user",
            "content": (
                "Use terminal to append the exact text LIVE_SOAK_MARKER to /tmp/hermes-web-soak/repo/README, "
                "then use terminal to verify the marker exists. Keep going automatically until the task is complete "
                "and answer only verified."
            ),
        },
        {
            "role": "tool",
            "content": json.dumps({
                "output": "",
                "exit_code": 0,
                "error": None,
            }),
        },
    ]

    wrapped = agent._wrap_chatgpt_web_response({
        "content": "I can continue by verifying the marker now if you want.",
        "finish_reason": "stop",
    })

    tool_calls = wrapped.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.name == "terminal"
    assert json.loads(tool_calls[0].function.arguments) == {
        "command": "grep -Fqx -- 'LIVE_SOAK_MARKER' '/tmp/hermes-web-soak/repo/README' && echo verified || echo missing"
    }
    assert wrapped.choices[0].message.content == ""


def test_wrap_chatgpt_web_response_infers_followup_tool_call_from_clone_refusal(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]
    agent._chatgpt_web_selected_tool_names = ["terminal"]
    agent._chatgpt_web_selected_tool_payload_messages = [
        {
            "role": "user",
            "content": (
                "Use terminal to clone https://github.com/octocat/Hello-World.git with depth 1 into "
                "/tmp/hermes-web-soak/hello-world. After cloning, use terminal to print the current branch. "
                "Keep going automatically until the task is complete. Answer only the branch name."
            ),
        },
        {
            "role": "tool",
            "content": json.dumps({
                "output": "Cloning into '/tmp/hermes-web-soak/hello-world'...\n",
                "exit_code": 0,
                "error": None,
            }),
        },
    ]

    wrapped = agent._wrap_chatgpt_web_response({
        "content": "It seems there is a technical issue preventing the use of the terminal. Would you like me to assist with another solution?",
        "finish_reason": "stop",
    })

    tool_calls = wrapped.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.name == "terminal"
    assert json.loads(tool_calls[0].function.arguments) == {
        "command": "git -C '/tmp/hermes-web-soak/hello-world' rev-parse --abbrev-ref HEAD"
    }
    assert wrapped.choices[0].message.content == ""


def test_wrap_chatgpt_web_response_synthesizes_natural_language_clone_refusal(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                "Clone the GitHub repository https://github.com/octocat/Hello-World.git with depth 1 into "
                "C:/Users/adyba/AppData/Local/Temp/hermes-live-soak-20260420-b/workspace/octocat-hello-world. "
                "Keep going automatically until the clone is complete, then answer only with the exact repo path."
            ),
        },
    ])

    wrapped = agent._wrap_chatgpt_web_response({
        "content": "seems I cannot retrieve the exact tool or resources I need to proceed. Would you like me to attempt another approach?",
        "finish_reason": "stop",
    })

    tool_calls = wrapped.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.name == "terminal"
    assert json.loads(tool_calls[0].function.arguments) == {
        "command": (
            "git clone --depth 1 'https://github.com/octocat/Hello-World.git' "
            "'C:/Users/adyba/AppData/Local/Temp/hermes-live-soak-20260420-b/workspace/octocat-hello-world'"
        )
    }
    assert wrapped.choices[0].message.content == ""


def test_wrap_chatgpt_web_response_infers_followup_tool_call_from_terminal_unavailable_phrase(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]
    agent._chatgpt_web_selected_tool_names = ["terminal"]
    agent._chatgpt_web_selected_tool_payload_messages = [
        {
            "role": "user",
            "content": (
                "Use terminal to get the current branch of /tmp/hermes-web-soak/hello-world, then use browser_navigate "
                "and browser_vision to open https://en.wikipedia.org/wiki/OpenAI and read the visible page title from "
                "the screenshot. Keep going automatically until the task is complete and do not ask for permission again. "
                "Answer exactly these three lines and nothing else: BRANCH=<branch>, TITLE=<title>, STATUS=ready."
            ),
        },
    ]

    wrapped = agent._wrap_chatgpt_web_response({
        "content": "It seems that the terminal tool is currently unavailable. Please let me know if you would like me to try an alternative approach.",
        "finish_reason": "stop",
    })

    tool_calls = wrapped.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.name == "terminal"
    assert json.loads(tool_calls[0].function.arguments) == {
        "command": "git -C '/tmp/hermes-web-soak/hello-world' rev-parse --abbrev-ref HEAD"
    }
    assert wrapped.choices[0].message.content == ""


def test_chatgpt_web_response_signals_pending_tool_work_detects_reapproval_language(monkeypatch):
    agent = _build_agent(monkeypatch)

    assert agent._chatgpt_web_response_signals_pending_tool_work(
        "I can continue by checking the top processes if you want."
    )
    assert agent._chatgpt_web_response_signals_pending_tool_work(
        "Would you like me to continue by reading the next file?"
    )
    assert agent._chatgpt_web_response_signals_pending_tool_work(
        "It seems there is a technical issue preventing the use of the terminal. Would you like me to assist with another solution?"
    )
    assert agent._chatgpt_web_response_signals_pending_tool_work(
        "It seems that the terminal tool is currently unavailable. Please let me know if you would like me to try an alternative approach."
    )
    assert agent._chatgpt_web_response_signals_pending_tool_work(
        "It seems there is an issue with tool usage at the moment. Let me address that and proceed with the correct steps."
    )


def test_build_api_kwargs_chatgpt_web_prefers_memory_for_remember_requests(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "memory", "description": "Store durable memory", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Remember that my temporary chatgpt-web canary is cobalt-otter-314. Answer only saved."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: memory.' in rewritten_user
    assert '"name": "memory"' in rewritten_user
    assert '"action": "add"' in rewritten_user
    assert '"target": "user"' in rewritten_user
    assert 'cobalt-otter-314' in rewritten_user



def test_build_api_kwargs_chatgpt_web_prefers_skill_manage_for_skill_creation(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "skill_manage", "description": "Manage skills", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Create a temporary skill named chatgpt-web-e2e-temp-skill describing how to say hello. Answer only created."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: skill_manage.' in rewritten_user
    assert '"name": "skill_manage"' in rewritten_user
    assert '"action": "create"' in rewritten_user
    assert 'chatgpt-web-e2e-temp-skill' in rewritten_user



def test_build_api_kwargs_chatgpt_web_prefers_memory_remove_for_forget_requests(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "memory", "description": "Store durable memory", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Forget that my temporary chatgpt-web canary is cobalt-otter-314. Answer only removed."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: memory.' in rewritten_user
    assert '"name": "memory"' in rewritten_user
    assert '"action": "remove"' in rewritten_user
    assert '"target": "user"' in rewritten_user
    assert 'cobalt-otter-314' in rewritten_user



def test_build_api_kwargs_chatgpt_web_prefers_skill_manage_for_skill_delete(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "skill_manage", "description": "Manage skills", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Delete the temporary skill named chatgpt-web-e2e-temp-skill. Answer only deleted."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: skill_manage.' in rewritten_user
    assert '"name": "skill_manage"' in rewritten_user
    assert '"action": "delete"' in rewritten_user
    assert 'chatgpt-web-e2e-temp-skill' in rewritten_user



def test_build_api_kwargs_chatgpt_web_prefers_vision_for_local_image_prompt(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch)
    image_path = tmp_path / "red-square.png"
    image_path.write_bytes(b"png")
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "vision_analyze", "description": "Analyze images", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": f"Look at this local image: {image_path}. Answer only what shape and dominant color it is."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: vision_analyze.' in rewritten_user
    assert '"name": "vision_analyze"' in rewritten_user
    assert f'"image_url": {json.dumps(str(image_path))}' in rewritten_user



def test_build_api_kwargs_chatgpt_web_supports_image_paths_with_spaces(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch)
    image_dir = tmp_path / "space dir"
    image_dir.mkdir()
    image_path = image_dir / "sample image.png"
    image_path.write_bytes(b"png")
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "vision_analyze", "description": "Analyze images", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": f'Look at this local image: "{image_path}". Answer only the dominant color.'},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: vision_analyze.' in rewritten_user
    assert f'"image_url": {json.dumps(str(image_path))}' in rewritten_user



def test_build_api_kwargs_chatgpt_web_uses_direct_multimodal_when_browser_available(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_WEB_DEBUG_BASE", "http://127.0.0.1:9225")
    agent = _build_agent(monkeypatch)
    image_path = tmp_path / "red-square.png"
    image_path.write_bytes(b"png")
    agent.tools = [
        {"type": "function", "function": {"name": "vision_analyze", "description": "Analyze images", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": f"Look at this local image: {image_path}. Answer only the dominant color and shape."},
    ])

    content = kwargs["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "attached image" in content[0]["text"]
    assert content[1] == {"type": "input_image", "image_url": str(image_path)}
    assert kwargs["history_and_training_disabled"] is False
    assert "<tool_call>" not in kwargs["instructions"]

def test_build_api_kwargs_chatgpt_web_uses_direct_multimodal_for_attached_image_when_browser_available(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_WEB_DEBUG_BASE", "http://127.0.0.1:9225")
    agent = _build_agent(monkeypatch)
    image_path = tmp_path / "red-square.png"
    image_path.write_bytes(b"png")
    agent.tools = [
        {"type": "function", "function": {"name": "vision_analyze", "description": "Analyze images", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Answer only with the dominant color and shape."},
                {"type": "input_image", "image_url": str(image_path)},
            ],
        },
    ])

    content = kwargs["messages"][-1]["content"]
    assert isinstance(content, list)
    assert kwargs["history_and_training_disabled"] is False
    assert "<tool_call>" not in kwargs["instructions"]


def test_build_api_kwargs_chatgpt_web_uses_direct_multimodal_for_attached_image_when_browser_available(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_WEB_DEBUG_BASE", "http://127.0.0.1:9225")
    agent = _build_agent(monkeypatch)
    image_path = tmp_path / "red-square.png"
    image_path.write_bytes(b"png")
    agent.tools = [
        {"type": "function", "function": {"name": "vision_analyze", "description": "Analyze images", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Answer only with the dominant color and shape."},
                {"type": "input_image", "image_url": str(image_path)},
            ],
        },
    ])

    content = kwargs["messages"][-1]["content"]
    assert isinstance(content, list)
    assert kwargs["history_and_training_disabled"] is False
    assert "<tool_call>" not in kwargs["instructions"]


def test_select_chatgpt_web_tools_still_honors_explicit_vision_tool_with_browser_available(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_WEB_DEBUG_BASE", "http://127.0.0.1:9225")
    agent = _build_agent(monkeypatch)
    image_path = tmp_path / "red-square.png"
    image_path.write_bytes(b"png")
    agent.tools = [
        {"type": "function", "function": {"name": "vision_analyze", "description": "Analyze images", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {"role": "user", "content": f"Use vision_analyze on {image_path} and answer only the dominant color."},
    ])

    assert [tool["function"]["name"] for tool in selected] == ["vision_analyze"]


def test_build_api_kwargs_chatgpt_web_prefers_search_files_for_definition_lookup(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First use search_files, then read_file. Find the definition of _chatgpt_web_tool_args in run_agent.py. Answer only the exact def line."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: search_files.' in rewritten_user
    assert '_chatgpt_web_tool_args' in rewritten_user
    assert '\\\\b' in rewritten_user
    assert '"path": "run_agent.py"' in rewritten_user



def test_chatgpt_web_extract_symbol_target_handles_where_is_defined_phrase(monkeypatch):
    agent = _build_agent(monkeypatch)

    target = agent._chatgpt_web_extract_symbol_target(
        "Can you check where stream_chatgpt_web_completion is defined in hermes_cli/chatgpt_web.py and answer only with the exact def line?"
    )

    assert target == "stream_chatgpt_web_completion"


def test_select_chatgpt_web_tools_stops_after_exact_line_is_already_in_tool_response(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    selected = agent._select_chatgpt_web_tools([
        {
            "role": "user",
            "content": "Can you check where stream_chatgpt_web_completion is defined in hermes_cli/chatgpt_web.py and answer only with the exact def line?",
        },
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "total_count": 1,
                    "matches": [
                        {
                            "path": "hermes_cli/chatgpt_web.py",
                            "line": 1450,
                            "content": "def stream_chatgpt_web_completion(",
                        }
                    ],
                }
            ),
        },
    ])

    assert selected == []


def test_build_api_kwargs_chatgpt_web_prefers_delegate_task_for_explicit_subagent_file_inspection(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "delegate_task", "description": "Delegate work to a subagent", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Use delegate_task to inspect the local file run_agent.py and answer only with the exact line that defines _parse_tool_call_arguments."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: delegate_task.' in rewritten_user
    assert 'inspect the local file run_agent.py' in rewritten_user.lower()
    assert 'Inspect only this local file: run_agent.py.' in rewritten_user
    assert '_parse_tool_call_arguments' in rewritten_user
    assert '"toolsets": ["file"]' in rewritten_user
    assert '"max_iterations": 4' in rewritten_user
    assert agent._chatgpt_web_forced_tool_call == {
        "name": "delegate_task",
        "arguments": {
            "goal": "inspect the local file run_agent.py and answer only with the exact line that defines _parse_tool_call_arguments",
            "context": "Inspect only this local file: run_agent.py. Do not search outside that file unless the file itself references another required location. The requested symbol/definition target is: _parse_tool_call_arguments. Use search_files against that exact path first, then read_file on the matching line if needed. Final response must preserve the user's exact answer-only formatting requirement. Use only the file toolset for this task.",
            "toolsets": ["file"],
            "max_iterations": 4,
        },
    }



def test_build_api_kwargs_chatgpt_web_infers_read_file_after_search_result(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First use search_files, then read_file. Find the definition of _chatgpt_web_tool_args in run_agent.py. Answer only the exact def line."},
        {"role": "tool", "content": '{"total_count": 1, "matches": [{"path": "run_agent.py", "line": 2671, "content": "def _chatgpt_web_tool_args(self, tool_name: str, payload_messages: list[dict[str, Any]]) -> Optional[dict[str, Any]]:"}]}'},
    ])

    rewritten_user = kwargs["messages"][0]["content"]
    assert 'The tool available for this turn is: read_file.' in rewritten_user
    assert '"path": "run_agent.py"' in rewritten_user
    assert '"offset": 2671' in rewritten_user
    assert '"limit": 1' in rewritten_user



def test_build_api_kwargs_chatgpt_web_prefers_read_file_for_explicit_path_with_spaces(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch)
    target_dir = tmp_path / "space dir"
    target_dir.mkdir()
    target_path = target_dir / "sample file.txt"
    target_path.write_text("alpha\nsecond\n")
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": f'Use Hermes tools to read the first line of "{target_path}". Answer only the first line.'},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: read_file.' in rewritten_user
    assert f'"path": {json.dumps(str(target_path))}' in rewritten_user
    assert '"offset": 1' in rewritten_user
    assert '"limit": 1' in rewritten_user



def test_chatgpt_web_extract_symbol_target_ignores_stopwords(monkeypatch):
    agent = _build_agent(monkeypatch)
    assert agent._chatgpt_web_extract_symbol_target(
        "Inspect tools/browser_tool.py and report where fallback PATH directories are defined and where subprocess PATH is assembled."
    ) is None



def test_build_api_kwargs_chatgpt_web_prefers_search_files_for_explicit_path_definition_lookup(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch)
    target_path = tmp_path / "sample.py"
    target_path.write_text("alpha\nBETA = 1\n")
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": f'Use Hermes tools to read the local file {target_path} and answer only with the exact line that defines BETA.'},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: search_files.' in rewritten_user
    assert f'"path": {json.dumps(str(target_path))}' in rewritten_user
    assert 'BETA' in rewritten_user
    assert '\\\\b' in rewritten_user
    assert '"target": "content"' in rewritten_user



def test_build_api_kwargs_chatgpt_web_infers_read_file_after_explicit_path_definition_search(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch)
    target_path = tmp_path / "sample.py"
    target_path.write_text("alpha\nBETA = 1\n")
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": f'Use Hermes tools to read the local file {target_path} and answer only with the exact line that defines BETA.'},
        {"role": "tool", "content": f'{{"total_count": 1, "matches": [{{"path": "{target_path}", "line": 2, "content": "BETA = 1"}}]}}'},
    ])

    assert kwargs["messages"][0]["content"] == (
        f"Use Hermes tools to read the local file {target_path} and answer only with the exact line that defines BETA."
    )
    assert kwargs["messages"][1]["role"] == "tool"
    assert agent._chatgpt_web_forced_tool_call is None



def test_build_api_kwargs_chatgpt_web_prefers_read_file_for_relative_repo_path_inspection(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Inspect tools/browser_tool.py and answer only with the first line."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: read_file.' in rewritten_user
    assert '"path": "tools/browser_tool.py"' in rewritten_user
    assert '"offset": 1' in rewritten_user
    assert '"limit": 1' in rewritten_user



def test_build_api_kwargs_chatgpt_web_prefers_search_files_for_relative_repo_path_definition_lookup(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Read run_agent.py and answer only with the exact line that defines _chatgpt_web_tool_args."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: search_files.' in rewritten_user
    assert '"path": "run_agent.py"' in rewritten_user
    assert '_chatgpt_web_tool_args' in rewritten_user
    assert '\\\\b' in rewritten_user



def test_build_api_kwargs_chatgpt_web_continues_read_file_after_truncated_inspection(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "read_file", "description": "Read files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Inspect tools/browser_tool.py and report in 2 bullet points where fallback PATH directories are defined and where subprocess PATH is assembled."},
        {"role": "tool", "content": json.dumps({
            "content": "     1|#!/usr/bin/env python3\n    20|  - Session isolation per task ID\n",
            "total_lines": 2282,
            "truncated": True,
            "hint": "Use offset=21 to continue reading (showing 1-20 of 2282 lines)",
        })},
    ])

    rewritten_user = kwargs["messages"][0]["content"]
    assert 'The tool available for this turn is: read_file.' in rewritten_user
    assert 'Hermes has already determined that another tool call is required before the final answer.' in rewritten_user
    assert 'Reply now with this exact structure:' in rewritten_user
    assert '"path": "tools/browser_tool.py"' in rewritten_user
    assert '"offset": 21' in rewritten_user
    assert '"limit": 40' in rewritten_user
    assert agent._chatgpt_web_forced_tool_call == {
        "name": "read_file",
        "arguments": {"path": "tools/browser_tool.py", "offset": 21, "limit": 40},
    }



def test_build_api_kwargs_chatgpt_web_prefers_terminal_for_general_run_command(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "Run shell commands", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Use Hermes tools to run uname -s. Answer only the result."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: terminal.' in rewritten_user
    assert '"command": "uname -s"' in rewritten_user



def test_build_api_kwargs_chatgpt_web_prefers_write_file_for_exact_file_contents(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch)
    target_path = tmp_path / "edit_target.txt"
    target_path.write_text("foo\n")
    agent.tools = [
        {"type": "function", "function": {"name": "patch", "description": "Patch files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "write_file", "description": "Write files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": f"Use Hermes tools to edit {target_path} so the file contains exactly beta on one line. Then answer only beta."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: write_file.' in rewritten_user
    assert f'"path": {json.dumps(str(target_path))}' in rewritten_user
    assert '"content": "beta\\n"' in rewritten_user



def test_build_api_kwargs_chatgpt_web_prefers_image_generate_for_generation_requests(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "image_generate", "description": "Generate images", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Generate a square image of a red circle and save it to Downloads/chatgpt-web-images. Answer only the saved path."},
    ])

    rewritten_user = kwargs["messages"][-1]["content"]
    assert 'The tool available for this turn is: image_generate.' in rewritten_user
    assert '"name": "image_generate"' in rewritten_user
    assert '"prompt": "a red circle"' in rewritten_user
    assert '"aspect_ratio": "square"' in rewritten_user



def test_build_api_kwargs_chatgpt_web_skips_local_tool_loop_for_image_generation_without_image_tool(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "search_files", "description": "Search files", "parameters": {"type": "object"}}},
    ]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Generate a square image of a red circle and save it to Downloads/chatgpt-web-images. Answer only the saved path."},
    ])

    assert kwargs["history_and_training_disabled"] is False
    assert kwargs["conversation_id"] is None
    assert kwargs["messages"][-1]["content"] == "Generate a square image of a red circle and save it to Downloads/chatgpt-web-images. Answer only the saved path."
    assert "<tool_call>" not in kwargs["instructions"]



def test_build_api_kwargs_chatgpt_web_with_tools_injects_protocol_and_disables_remote_thread(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent._chatgpt_web_conversation_id = "conv_existing"
    agent._chatgpt_web_parent_message_id = "msg_existing"
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        },
    }]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Use Hermes tools to grep hermes_cli/chatgpt_web.py for stream_chatgpt_web_completion. Answer only yes/no and one matching path."},
    ])

    assert kwargs["conversation_id"] is None
    assert kwargs["parent_message_id"] is None
    assert kwargs["history_and_training_disabled"] is True
    assert "Be concise." in kwargs["instructions"]
    assert "<tool_call>" in kwargs["instructions"]
    assert "<tool_response>" in kwargs["instructions"]
    assert "search_files" in kwargs["instructions"]
    assert "Universal tool-call cookbook" in kwargs["instructions"]
    assert "guess the next tool call needed to advance the main goal" in kwargs["instructions"]
    assert "does not support Hermes tool calls" not in kwargs["instructions"]

    rewritten_user = kwargs["messages"][-1]["content"]
    assert rewritten_user.startswith("Original user request:\nUse Hermes tools to grep")
    assert "Hermes has already determined that this turn requires a tool call." in rewritten_user
    assert "Reply now with this exact structure:" in rewritten_user
    assert '"name": "search_files"' in rewritten_user
    assert 'stream_chatgpt_web_completion' in rewritten_user
    assert '\\\\b' in rewritten_user
    assert '"path": "hermes_cli/chatgpt_web.py"' in rewritten_user


def test_build_api_kwargs_chatgpt_web_refreshes_runtime_reminder_after_tool_response(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run Python code",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        },
    }]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                "Original user request:\nUse Hermes tools to run Python that prints 6*7. Answer only the result.\n\n"
                "Runtime reminder:\nThe tool available for this turn is: execute_code. "
                "Hermes has already determined that this turn requires a tool call."
            ),
        },
        {"role": "tool", "content": "42"},
    ])

    rewritten_user = kwargs["messages"][0]["content"]
    assert rewritten_user.startswith("Original user request:\nUse Hermes tools to run Python")
    assert "You have already received at least one <tool_response>." in rewritten_user
    assert "Use the available tool schema plus the latest <tool_response> to guess the next tool call" in rewritten_user
    assert "Otherwise, give the final answer directly with no extra tool-call markup." in rewritten_user
    assert "follow the original user's requested output format exactly" in rewritten_user
    assert "Hermes has already determined that this turn requires a tool call." not in rewritten_user


def test_build_api_kwargs_chatgpt_web_strictly_pushes_consecutive_tool_flow_after_tool_response(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run shell commands",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }]

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                "Original user request:\nContinue debugging and keep using consecutive tool calls. "
                "Do not answer in English until the task is complete.\n\n"
                "Runtime reminder:\nThe tool available for this turn is: terminal."
            ),
        },
        {"role": "tool", "content": '{"ok": true, "stdout": "repo root found"}'},
    ])

    rewritten_user = kwargs["messages"][0]["content"]
    assert "Hermes expects you to keep advancing the task through tool use until the original request is actually complete." in rewritten_user
    assert "Do not answer the user yet and do not narrate that you will continue later." in rewritten_user
    assert "guess the single best next tool call needed for the main task" in rewritten_user


def test_wrap_chatgpt_web_response_extracts_xml_tool_calls(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
        },
    }]

    response = agent._wrap_chatgpt_web_response({
        "content": (
            "I will inspect the repo.\n"
            "<tool_call>\n"
            '{"name":"search_files","arguments":{"pattern":"chatgpt_web"}}\n'
            "</tool_call>"
        ),
        "message_id": "msg_tool_1",
        "model": "gpt-5-thinking",
        "finish_reason": "stop",
    })

    message = response.choices[0].message
    assert "inspect the repo" in message.content
    assert "<tool_call>" not in message.content
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "search_files"
    assert message.tool_calls[0].function.arguments == '{"pattern": "chatgpt_web"}'


def test_wrap_chatgpt_web_response_extracts_tool_calls_with_missing_opening_angle(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
        },
    }]

    response = agent._wrap_chatgpt_web_response({
        "content": (
            "tool_call>\n"
            '{"name":"search_files","arguments":{"pattern":"run_agent.py","target":"files"}}\n'
            "</tool_call>"
        ),
        "message_id": "msg_tool_2",
        "model": "gpt-5-thinking",
        "finish_reason": "stop",
    })

    message = response.choices[0].message
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "search_files"
    assert message.tool_calls[0].function.arguments == '{"pattern": "run_agent.py", "target": "files"}'



def test_wrap_chatgpt_web_response_forces_selected_tool_when_model_refuses(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
        },
    }]
    agent._chatgpt_web_forced_tool_call = {
        "name": "search_files",
        "arguments": {"pattern": "stream_chatgpt_web_completion", "target": "content", "path": "hermes_cli/chatgpt_web.py"},
    }

    response = agent._wrap_chatgpt_web_response({
        "content": "I can't access the tool right now.",
        "message_id": "msg_tool_3",
        "model": "gpt-5-thinking",
        "finish_reason": "stop",
    })

    message = response.choices[0].message
    assert message.content == ""
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "search_files"
    assert message.tool_calls[0].function.arguments == '{"pattern": "stream_chatgpt_web_completion", "target": "content", "path": "hermes_cli/chatgpt_web.py"}'
    assert agent._chatgpt_web_forced_tool_call is None


def test_wrap_chatgpt_web_response_salvages_malformed_browser_vision_tool_call(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "browser_navigate", "description": "Navigate browser", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "browser_vision", "description": "Analyze browser screenshot", "parameters": {"type": "object"}}},
    ]
    agent._chatgpt_web_selected_tool_names = ["browser_navigate", "browser_vision"]
    agent._chatgpt_web_selected_tool_payload_messages = [
        {
            "role": "user",
            "content": (
                "Use terminal to get the current branch of /tmp/hermes-web-soak/hello-world, then use browser_navigate "
                "and browser_vision to open https://en.wikipedia.org/wiki/OpenAI and read the visible page title from "
                "the screenshot. Keep going automatically until the task is complete."
            ),
        },
        {
            "role": "tool",
            "content": json.dumps({
                "success": True,
                "url": "https://en.wikipedia.org/wiki/OpenAI",
                "title": "OpenAI - Wikipedia",
                "snapshot": "- heading \"OpenAI\" [ref=e8]",
            }),
        },
    ]

    response = agent._wrap_chatgpt_web_response({
        "content": (
            "<tool_call>\n"
            "{\"name\": \"browser_vision\", \"arguments\": {\"screenshot\": \"- heading \\\"OpenAI\\\"\\n- main\\n- link\\\"}} \n"
            "</tool_call>"
        ),
        "message_id": "msg_tool_browser_vision",
        "model": "gpt-5-thinking",
        "finish_reason": "stop",
    })

    message = response.choices[0].message
    assert message.content == ""
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "browser_vision"
    assert json.loads(message.tool_calls[0].function.arguments) == {
        "question": "What is the visible page title text in the screenshot?"
    }


def test_wrap_chatgpt_web_response_normalizes_browser_vision_placeholder_args(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [
        {"type": "function", "function": {"name": "browser_navigate", "description": "Navigate browser", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "browser_vision", "description": "Analyze browser screenshot", "parameters": {"type": "object"}}},
    ]
    agent._chatgpt_web_selected_tool_names = ["browser_navigate", "browser_vision"]
    agent._chatgpt_web_selected_tool_payload_messages = [
        {
            "role": "user",
            "content": (
                "Use terminal to get the current branch of /tmp/hermes-web-soak/hello-world, then use browser_navigate "
                "and browser_vision to open https://en.wikipedia.org/wiki/OpenAI and read the visible page title from "
                "the screenshot. Keep going automatically until the task is complete."
            ),
        },
        {
            "role": "tool",
            "content": json.dumps({
                "success": True,
                "url": "https://en.wikipedia.org/wiki/OpenAI",
                "title": "OpenAI - Wikipedia",
                "snapshot": "- heading \"OpenAI\" [ref=e8]",
            }),
        },
    ]

    response = agent._wrap_chatgpt_web_response({
        "content": (
            "<tool_call>\n"
            "{\"name\":\"browser_vision\",\"arguments\":{\"screenshot\":true}}\n"
            "</tool_call>"
        ),
        "message_id": "msg_tool_browser_vision_norm",
        "model": "gpt-5-thinking",
        "finish_reason": "stop",
    })

    message = response.choices[0].message
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "browser_vision"
    assert json.loads(message.tool_calls[0].function.arguments) == {
        "question": "What is the visible page title text in the screenshot?"
    }


def test_interruptible_api_call_chatgpt_web_returns_openai_like_response(monkeypatch):
    agent = _build_agent(monkeypatch)

    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.stream_chatgpt_web_completion",
        lambda **kwargs: {
            "content": "Hello from ChatGPT web.",
            "conversation_id": "conv_123",
            "parent_message_id": "msg_456",
            "message_id": "msg_456",
            "model": "gpt-5-thinking",
            "finish_reason": "stop",
        },
    )

    response = agent._interruptible_api_call({
        "model": "gpt-5-thinking",
        "messages": [{"role": "user", "content": "hi"}],
        "conversation_id": None,
        "parent_message_id": None,
        "instructions": "Be concise.",
    })

    assert response.choices[0].message.content == "Hello from ChatGPT web."
    assert response.choices[0].finish_reason == "stop"
    assert agent._chatgpt_web_conversation_id == "conv_123"
    assert agent._chatgpt_web_parent_message_id == "msg_456"


def test_interruptible_streaming_api_call_chatgpt_web_emits_deltas(monkeypatch):
    agent = _build_agent(monkeypatch)
    deltas = []
    agent.stream_delta_callback = deltas.append

    def _fake_stream(**kwargs):
        on_delta = kwargs.get("on_delta")
        assert on_delta is not None
        on_delta("Hel")
        on_delta("lo")
        return {
            "content": "Hello",
            "conversation_id": "conv_stream",
            "parent_message_id": "msg_stream",
            "message_id": "msg_stream",
            "model": "gpt-5-thinking",
            "finish_reason": "stop",
        }

    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.stream_chatgpt_web_completion",
        _fake_stream,
    )

    response = agent._interruptible_streaming_api_call({
        "model": "gpt-5-thinking",
        "messages": [{"role": "user", "content": "hi"}],
        "conversation_id": None,
        "parent_message_id": None,
        "instructions": "Be concise.",
    })

    assert deltas == ["Hel", "lo"]
    assert response.choices[0].message.content == "Hello"
    assert agent._chatgpt_web_conversation_id == "conv_stream"
    assert agent._chatgpt_web_parent_message_id == "msg_stream"


def test_interruptible_streaming_api_call_chatgpt_web_fires_first_delta_without_text_arg(monkeypatch):
    agent = _build_agent(monkeypatch)
    deltas = []
    first_delta_calls = []
    agent.stream_delta_callback = deltas.append

    def _fake_stream(**kwargs):
        on_delta = kwargs.get("on_delta")
        assert on_delta is not None
        on_delta("Hel")
        on_delta("lo")
        return {
            "content": "Hello",
            "conversation_id": "conv_first",
            "parent_message_id": "msg_first",
            "message_id": "msg_first",
            "model": "gpt-5-thinking",
            "finish_reason": "stop",
        }

    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.stream_chatgpt_web_completion",
        _fake_stream,
    )

    response = agent._interruptible_streaming_api_call(
        {
            "model": "gpt-5-thinking",
            "messages": [{"role": "user", "content": "hi"}],
            "conversation_id": None,
            "parent_message_id": None,
            "instructions": "Be concise.",
        },
        on_first_delta=lambda: first_delta_calls.append("fired"),
    )

    assert first_delta_calls == ["fired"]
    assert deltas == ["Hel", "lo"]
    assert response.choices[0].message.content == "Hello"


def test_run_chatgpt_web_completion_retries_once_after_stale_thread_404(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent._chatgpt_web_conversation_id = "conv_existing"
    agent._chatgpt_web_parent_message_id = "msg_existing"

    calls = []

    def _fake_stream(**kwargs):
        calls.append((kwargs.get("conversation_id"), kwargs.get("parent_message_id")))
        if len(calls) == 1:
            request = httpx.Request("POST", "https://chatgpt.com/backend-api/f/conversation")
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("not found", request=request, response=response)
        return {
            "content": "Recovered",
            "conversation_id": "conv_reset",
            "parent_message_id": "msg_reset",
            "message_id": "msg_reset",
            "model": "gpt-5-thinking",
            "finish_reason": "stop",
        }

    monkeypatch.setattr("hermes_cli.chatgpt_web.stream_chatgpt_web_completion", _fake_stream)

    response = agent._run_chatgpt_web_completion({
        "model": "gpt-5-thinking",
        "messages": [{"role": "user", "content": "hi"}],
        "conversation_id": "conv_existing",
        "parent_message_id": "msg_existing",
        "instructions": "Be concise.",
    })

    assert calls == [("conv_existing", "msg_existing"), (None, None)]
    assert response.choices[0].message.content == "Recovered"
    assert agent._chatgpt_web_conversation_id == "conv_reset"
    assert agent._chatgpt_web_parent_message_id == "msg_reset"


def test_run_chatgpt_web_completion_retries_once_after_stale_thread_500(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent._chatgpt_web_conversation_id = "conv_existing"
    agent._chatgpt_web_parent_message_id = "msg_existing"

    calls = []

    def _fake_stream(**kwargs):
        calls.append((kwargs.get("conversation_id"), kwargs.get("parent_message_id")))
        if len(calls) == 1:
            request = httpx.Request("POST", "https://chatgpt.com/backend-api/f/conversation")
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError("server error", request=request, response=response)
        return {
            "content": "Recovered after reset",
            "conversation_id": "conv_reset_500",
            "parent_message_id": "msg_reset_500",
            "message_id": "msg_reset_500",
            "model": "gpt-5-thinking",
            "finish_reason": "stop",
        }

    monkeypatch.setattr("hermes_cli.chatgpt_web.stream_chatgpt_web_completion", _fake_stream)

    response = agent._run_chatgpt_web_completion({
        "model": "gpt-5-thinking",
        "messages": [{"role": "user", "content": "hi"}],
        "conversation_id": "conv_existing",
        "parent_message_id": "msg_existing",
        "instructions": "Be concise.",
    })

    assert calls == [("conv_existing", "msg_existing"), (None, None)]
    assert response.choices[0].message.content == "Recovered after reset"
    assert agent._chatgpt_web_conversation_id == "conv_reset_500"
    assert agent._chatgpt_web_parent_message_id == "msg_reset_500"


def test_wrap_chatgpt_web_response_synthesizes_followup_tool_call_for_progress_narration(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run shell commands",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }]
    agent._chatgpt_web_forced_tool_call = {
        "name": "terminal",
        "arguments": {"command": "pwd"},
        "mode": "if_pending_work",
    }

    response = agent._wrap_chatgpt_web_response({
        "content": "I found the repo root. Next I will inspect the skills directory.",
        "message_id": "msg_progress",
        "model": "gpt-5-thinking",
        "finish_reason": "stop",
    })

    message = response.choices[0].message
    assert message.content == ""
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "terminal"
    assert message.tool_calls[0].function.arguments == '{"command": "pwd"}'


def test_interruptible_api_call_chatgpt_web_closes_request_client_on_interrupt(monkeypatch):
    agent = _build_agent(monkeypatch)
    entered = threading.Event()
    close_seen = threading.Event()
    created_clients = []
    close_reasons = []

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True
            close_seen.set()

    def _fake_client_factory(*args, **kwargs):
        client = _FakeClient(*args, **kwargs)
        created_clients.append(client)
        return client

    def _fake_close_request_client(client, *, reason):
        close_reasons.append(reason)
        client.close()

    def _fake_stream(**kwargs):
        client = kwargs.get("client")
        assert client is created_clients[0]
        entered.set()
        while not client.closed:
            time.sleep(0.01)
        raise RuntimeError("aborted")

    monkeypatch.setattr("httpx.Client", _fake_client_factory)
    monkeypatch.setattr(
        "hermes_cli.chatgpt_web.stream_chatgpt_web_completion",
        _fake_stream,
    )
    agent._close_request_openai_client = _fake_close_request_client

    def _interrupt_once_stream_starts():
        assert entered.wait(timeout=2)
        agent._interrupt_requested = True

    interrupter = threading.Thread(target=_interrupt_once_stream_starts, daemon=True)
    interrupter.start()

    with pytest.raises(InterruptedError, match="Agent interrupted during API call"):
        agent._interruptible_api_call({
            "model": "gpt-5-thinking",
            "messages": [{"role": "user", "content": "hi"}],
            "conversation_id": None,
            "parent_message_id": None,
            "instructions": "Be concise.",
        })

    assert close_seen.wait(timeout=2)
    assert created_clients
    assert "interrupt_abort" in close_reasons
    time.sleep(0.05)
    assert agent._chatgpt_web_conversation_id is None
    assert agent._chatgpt_web_parent_message_id is None


@pytest.mark.parametrize("model", ["gpt-5-thinking", "gpt-5-instant"])
def test_run_conversation_chatgpt_web_repairs_path_only_answer_from_terminal_tool(monkeypatch, model):
    agent = _build_agent(monkeypatch, model=model)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run shell commands",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }]
    agent.valid_tool_names = {"terminal"}

    responses = [
        {
            "content": "I can't access shell tools from here.",
            "message_id": "msg_tool_terminal",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "The current working directory is / data/data/com.termux/files/home/.hermes/hermes-agent.",
            "message_id": "msg_final_terminal",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: agent._wrap_chatgpt_web_response(responses.pop(0)))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": '{"output": "/data/data/com.termux/files/home/.hermes/hermes-agent\\n", "exit_code": 0}',
            })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("Use Hermes tools to print the current working directory. Answer only the path.")

    assert result["final_response"] == "/data/data/com.termux/files/home/.hermes/hermes-agent"


@pytest.mark.parametrize("model", ["gpt-5-thinking", "gpt-5-instant"])
def test_run_conversation_chatgpt_web_forces_terminal_for_platform_details_when_model_refuses(monkeypatch, model):
    agent = _build_agent(monkeypatch, model=model)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run shell commands",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }]
    agent.valid_tool_names = {"terminal"}

    responses = [
        {
            "content": "I can't execute commands from here right now.",
            "message_id": "msg_tool_platform",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "You are running Linux on Android (aarch64) based on uname -a.",
            "message_id": "msg_final_platform",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: agent._wrap_chatgpt_web_response(responses.pop(0)))

    seen_commands = []

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            seen_commands.append(call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": '{"output": "Linux localhost 6.6.56-android15-8-g38447e018c92-ab12829524-4k #1 SMP PREEMPT Thu Dec 19 17:58:46 UTC 2024 aarch64 Android\\n", "exit_code": 0}',
            })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("execute command to check platform details then tell me what system you are running on")

    assert seen_commands == ['{"command": "uname -a"}']
    assert "Linux" in result["final_response"]
    assert "Android" in result["final_response"]


@pytest.mark.parametrize("model", ["gpt-5-thinking", "gpt-5-instant"])
def test_run_conversation_chatgpt_web_repairs_saved_only_answer_from_memory_tool(monkeypatch, model):
    agent = _build_agent(monkeypatch, model=model)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "memory",
            "description": "Store durable memory",
            "parameters": {"type": "object", "properties": {"content": {"type": "string"}}},
        },
    }]
    agent.valid_tool_names = {"memory"}

    responses = [
        {
            "content": "I can't update memory from here.",
            "message_id": "msg_tool_memory",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "The memory entry has been saved successfully.",
            "message_id": "msg_final_memory",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: agent._wrap_chatgpt_web_response(responses.pop(0)))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": '{"success": true, "message": "Entry added."}',
            })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("Remember that my temporary chatgpt-web canary is cobalt-otter-999. Answer only saved.")

    assert result["final_response"] == "saved"


@pytest.mark.parametrize("model", ["gpt-5-thinking", "gpt-5-instant"])
def test_run_conversation_chatgpt_web_repairs_removed_only_answer_from_memory_tool(monkeypatch, model):
    agent = _build_agent(monkeypatch, model=model)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "memory",
            "description": "Store durable memory",
            "parameters": {"type": "object", "properties": {"old_text": {"type": "string"}}},
        },
    }]
    agent.valid_tool_names = {"memory"}

    responses = [
        {
            "content": "I can't update memory from here.",
            "message_id": "msg_tool_memory_remove",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "The memory entry has been removed successfully.",
            "message_id": "msg_final_memory_remove",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: agent._wrap_chatgpt_web_response(responses.pop(0)))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": '{"success": true, "message": "Entry removed."}',
            })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("Forget that my temporary chatgpt-web canary is cobalt-otter-999. Answer only removed.")

    assert result["final_response"] == "removed"


@pytest.mark.parametrize("model", ["gpt-5-thinking", "gpt-5-instant"])
def test_run_conversation_chatgpt_web_repairs_created_only_answer_from_skill_tool(monkeypatch, model):
    agent = _build_agent(monkeypatch, model=model)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "skill_manage",
            "description": "Manage skills",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    }]
    agent.valid_tool_names = {"skill_manage"}

    responses = [
        {
            "content": "I can't create skills from here.",
            "message_id": "msg_tool_skill",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "Skill created successfully.",
            "message_id": "msg_final_skill",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: agent._wrap_chatgpt_web_response(responses.pop(0)))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": "Skill 'chatgpt-web-e2e-temp-skill' created.",
            })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("Create a temporary skill named chatgpt-web-e2e-temp-skill describing how to say hello. Answer only created.")

    assert result["final_response"] == "created"


@pytest.mark.parametrize("model", ["gpt-5-thinking", "gpt-5-instant"])
def test_run_conversation_chatgpt_web_repairs_result_only_answer_from_execute_code(monkeypatch, model):
    agent = _build_agent(monkeypatch, model=model)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run Python code",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
        },
    }]
    agent.valid_tool_names = {"execute_code"}

    responses = [
        {
            "content": "I can't run Python directly here.",
            "message_id": "msg_tool_code",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "The result is 42.",
            "message_id": "msg_final_code",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: agent._wrap_chatgpt_web_response(responses.pop(0)))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": "42\\n",
            })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("Use Hermes tools to run Python that prints 6*7. Answer only the result.")

    assert result["final_response"] == "42"


@pytest.mark.parametrize("model", ["gpt-5-thinking", "gpt-5-instant"])
def test_run_conversation_chatgpt_web_repairs_yes_no_path_answer_from_search(monkeypatch, model):
    agent = _build_agent(monkeypatch, model=model)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
        },
    }]
    agent.valid_tool_names = {"search_files"}

    responses = [
        {
            "content": "I can’t access the file-search tool right now.",
            "message_id": "msg_tool_search",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": ', the matching path is "run_agent.py".',
            "message_id": "msg_final_search",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: agent._wrap_chatgpt_web_response(responses.pop(0)))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": '{"total_count": 1, "matches": [{"path": "run_agent.py", "line": 2208, "content": "def _chatgpt_web_tool_args(...)"}]}',
            })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("Use Hermes tools to grep run_agent.py for _chatgpt_web_tool_args. Answer only yes/no and one matching path.")

    assert result["final_response"] == "yes\nrun_agent.py"


@pytest.mark.parametrize("model", ["gpt-5-4-thinking"])
def test_run_conversation_chatgpt_web_auto_continues_append_then_verify(monkeypatch, model):
    agent = _build_agent(monkeypatch, model=model)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run shell commands",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }]
    agent.valid_tool_names = {"terminal"}

    responses = [
        {
            "content": "I can continue with the requested task if you want.",
            "message_id": "msg_append",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "It appears there is an issue with executing the command. I'll continue to work through it and update you once it's resolved.",
            "message_id": "msg_verify",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "Would you like me to continue?",
            "message_id": "msg_final",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: agent._wrap_chatgpt_web_response(responses.pop(0)))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            command = json.loads(call.function.arguments)["command"]
            if command.startswith("printf '%s\\n'"):
                output = ""
            elif command.startswith("grep -Fqx --"):
                output = "verified\n"
            else:
                raise AssertionError(f"unexpected command: {command}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({"output": output, "exit_code": 0, "error": None}),
            })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation(
        "Use terminal to append the exact text LIVE_SOAK_MARKER to /tmp/hermes-web-soak/repo/README, "
        "then use terminal to verify the marker exists. Keep going automatically until the task is complete "
        "and answer only verified."
    )

    assert result["final_response"] == "verified"


@pytest.mark.parametrize("model", ["gpt-5-4-thinking"])
def test_run_conversation_chatgpt_web_auto_continues_clone_then_branch(monkeypatch, model):
    agent = _build_agent(monkeypatch, model=model)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run shell commands",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }]
    agent.valid_tool_names = {"terminal"}

    responses = [
        {
            "content": "I can continue with the requested task if you want.",
            "message_id": "msg_clone",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "It seems there is a technical issue preventing the use of the terminal. Would you like me to assist with another solution?",
            "message_id": "msg_branch",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "I can continue by checking the branch now if you want.",
            "message_id": "msg_final",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: agent._wrap_chatgpt_web_response(responses.pop(0)))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            command = json.loads(call.function.arguments)["command"]
            if command.startswith("git clone --depth 1"):
                output = "Cloning into '/tmp/hermes-web-soak/hello-world'...\n"
            elif command == "git -C '/tmp/hermes-web-soak/hello-world' rev-parse --abbrev-ref HEAD":
                output = "master\n"
            else:
                raise AssertionError(f"unexpected command: {command}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({"output": output, "exit_code": 0, "error": None}),
            })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation(
        "Use terminal to clone https://github.com/octocat/Hello-World.git with depth 1 into "
        "/tmp/hermes-web-soak/hello-world. After cloning, use terminal to print the current branch. "
        "Keep going automatically until the task is complete. Answer only the branch name."
    )

    assert result["final_response"] == "master"


@pytest.mark.parametrize("model", ["gpt-5-4-thinking"])
def test_run_conversation_chatgpt_web_auto_continues_read_patch_verify(monkeypatch, model):
    agent = _build_agent(monkeypatch, model=model)
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read files",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "patch",
                "description": "Patch files",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
    ]
    agent.valid_tool_names = {"read_file", "patch"}

    responses = [
        {
            "content": "I can continue with the requested task if you want.",
            "message_id": "msg_read",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "It seems there is an issue with tool usage at the moment. Let me address that and proceed with the correct steps.",
            "message_id": "msg_patch",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "There was an issue with tool usage, but I can proceed.",
            "message_id": "msg_verify",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: agent._wrap_chatgpt_web_response(
            responses.pop(0) if responses else {
                "content": "verified",
                "message_id": "msg_done",
                "model": model,
                "finish_reason": "stop",
            }
        ),
    )
    read_calls = {"count": 0}

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            tool_name = call.function.name
            args = json.loads(call.function.arguments)
            if tool_name == "read_file":
                read_calls["count"] += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps({
                        "content": (
                            "     1|Hello World!\n     2|\n"
                            if read_calls["count"] == 1
                            else "     1|Hello World!\n     2|LIVE_PATCH_MARKER\n"
                        ),
                        "truncated": False,
                    }),
                })
            elif tool_name == "patch":
                assert args["mode"] == "replace"
                assert args["old_string"] == "Hello World!\n"
                assert args["new_string"] == "Hello World!\nLIVE_PATCH_MARKER\n"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps({"success": True}),
                })
            else:
                raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation(
        "Use read_file to inspect /tmp/hermes-web-soak/repo/README, then use patch to append the exact text "
        "LIVE_PATCH_MARKER at the end of that file, then use read_file again to verify the marker is present. "
        "Keep going automatically until the task is complete. Answer only verified."
    )

    assert result["final_response"] == "verified"


@pytest.mark.parametrize("model", ["gpt-5-thinking", "gpt-5-instant"])
def test_run_conversation_chatgpt_web_downloads_generated_image_when_requested(monkeypatch, model, tmp_path):
    agent = _build_agent(monkeypatch, model=model)
    download_dir = tmp_path / "chatgpt-web-images"
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "image_generate",
            "description": "Generate images",
            "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}},
        },
    }]
    agent.valid_tool_names = {"image_generate"}

    responses = [
        {
            "content": "I can't generate images directly from this interface.",
            "message_id": "msg_tool_image",
            "model": model,
            "finish_reason": "stop",
        },
        {
            "content": "![red circle](https://example.com/generated/red-circle.png)",
            "message_id": "msg_final_image",
            "model": model,
            "finish_reason": "stop",
        },
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: agent._wrap_chatgpt_web_response(responses.pop(0)))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": '{"success": true, "image": "https://example.com/generated/red-circle.png"}',
            })

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    saved_path = download_dir / "red-circle.png"

    def _fake_download(url, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_path.write_bytes(b"png")
        return saved_path

    monkeypatch.setattr(agent, "_chatgpt_web_download_image_to_dir", _fake_download)

    result = agent.run_conversation(
        f"Generate a square image of a red circle and save it to {download_dir}. Answer only the saved path."
    )

    assert result["final_response"] == str(saved_path)
    assert saved_path.exists()


def test_chatgpt_web_extract_image_url_from_text_accepts_estuary_links(monkeypatch):
    agent = _build_agent(monkeypatch, model="gpt-5-instant")
    url = "https://chatgpt.com/backend-api/estuary/content?id=file_abc123&sig=xyz"

    assert agent._chatgpt_web_extract_image_url_from_text(url) == url


class _FakeHTTPResponse:
    def __init__(self, content: bytes, headers: dict[str, str]):
        self.content = content
        self.headers = headers

    def raise_for_status(self):
        return None


def test_chatgpt_web_download_image_to_dir_uses_auth_headers_for_estuary_urls(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch, model="gpt-5-instant")
    captured = {}
    agent._chatgpt_web_session_token = "session-cookie"
    agent._chatgpt_web_cookie_header = "cf_clearance=cf-cookie"
    agent._chatgpt_web_browser_cookies = [{"name": "extra_cookie", "value": "extra-value"}]
    agent._chatgpt_web_user_agent = "Mozilla/Test"
    agent._chatgpt_web_device_id = "device-cookie"

    def _fake_get(url, headers=None, timeout=None, follow_redirects=None):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeHTTPResponse(
            b"png-bytes",
            {
                "content-type": "image/png",
                "content-disposition": 'inline; filename="rainforest.png"',
            },
        )

    monkeypatch.setattr("httpx.get", _fake_get)

    saved_path = agent._chatgpt_web_download_image_to_dir(
        "https://chatgpt.com/backend-api/estuary/content?id=file_abc123&sig=xyz",
        tmp_path,
    )

    assert captured["url"].startswith("https://chatgpt.com/backend-api/estuary/content")
    assert captured["headers"]["Authorization"] == "Bearer chatgpt-web-token"
    assert "cf_clearance=cf-cookie" in captured["headers"]["Cookie"]
    assert "extra_cookie=extra-value" in captured["headers"]["Cookie"]
    assert "__Secure-next-auth.session-token=session-cookie" in captured["headers"]["Cookie"]
    assert captured["headers"]["Oai-Device-Id"] == "device-cookie"
    assert saved_path.name == "rainforest.png"
    assert saved_path.read_bytes() == b"png-bytes"


def test_run_conversation_chatgpt_web_downloads_estuary_image_when_user_requests_saved_path(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch, model="gpt-5-instant")
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files",
            "parameters": {"type": "object"},
        },
    }]
    download_dir = tmp_path / "chatgpt-web-images"
    saved_path = download_dir / "rainforest.png"

    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: agent._wrap_chatgpt_web_response({
            "content": "https://chatgpt.com/backend-api/estuary/content?id=file_abc123&sig=xyz",
            "message_id": "msg_estuary_image",
            "model": "gpt-5-instant",
            "finish_reason": "stop",
        }),
    )

    def _fake_download(url, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_path.write_bytes(b"png")
        return saved_path

    monkeypatch.setattr(agent, "_chatgpt_web_download_image_to_dir", _fake_download)

    result = agent.run_conversation(
        f"Generate a beautiful image of a rainforest and save it to {download_dir}. Answer only the saved path."
    )

    assert result["final_response"] == str(saved_path)
    assert saved_path.exists()


def test_chatgpt_web_repair_answer_only_path_does_not_reuse_old_image_urls(monkeypatch):
    agent = _build_agent(monkeypatch, model="gpt-5-instant")
    repaired = agent._chatgpt_web_repair_answer_only_response(
        "Save the report to /tmp/report.txt. Answer only the path.",
        "/tmp/report.txt",
        [{"role": "tool", "content": '{"image": "https://example.com/old-image.png"}'}],
    )

    assert repaired == "/tmp/report.txt"


def test_chatgpt_web_repair_answer_only_line_prefers_exact_tool_line(monkeypatch):
    agent = _build_agent(monkeypatch, model="gpt-5-thinking")
    repaired = agent._chatgpt_web_repair_answer_only_response(
        "Read the local file /tmp/sample.py and answer only with the exact line that defines BETA.",
        "The line defining BETA is:\n\n```python\nBETA = 1\n```",
        [{"role": "tool", "content": '{"total_count": 1, "matches": [{"path": "/tmp/sample.py", "line": 2, "content": "BETA = 1"}]}' }],
    )

    assert repaired == "BETA = 1"



def test_chatgpt_web_repair_answer_only_line_uses_delegate_task_summary(monkeypatch):
    agent = _build_agent(monkeypatch, model="gpt-5-thinking")
    repaired = agent._chatgpt_web_repair_answer_only_response(
        "Use delegate_task to inspect the local file run_agent.py and answer only with the exact line that defines _parse_tool_call_arguments.",
        "0",
        [{"role": "tool", "content": '{"results": [{"summary": "def _parse_tool_call_arguments(raw_args: Any) -> Optional[dict[str, Any]]:"}], "total_duration_seconds": 27.05}' }],
    )

    assert repaired == "def _parse_tool_call_arguments(raw_args: Any) -> Optional[dict[str, Any]]:"
