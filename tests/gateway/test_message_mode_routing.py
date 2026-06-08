"""Gateway prefix-based message mode routing tests."""

import dis
import inspect
import json

import pytest

from gateway.config import Platform
from gateway.message_modes import GatewayMessageMode, resolve_gateway_message_mode
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner, _route_gateway_message_mode
from gateway.session import SessionSource, build_session_key


def test_run_agent_closure_marks_rebound_toolsets_nonlocal():
    source = inspect.getsource(GatewayRunner._run_agent)

    assert "nonlocal message, enabled_toolsets, disabled_toolsets" in source


def test_handle_message_with_agent_receives_routed_message_mode():
    signature = inspect.signature(GatewayRunner._handle_message_with_agent)
    assert "message_mode" in signature.parameters

    instructions = list(dis.get_instructions(GatewayRunner._handle_message_with_agent))
    assert not [
        instruction
        for instruction in instructions
        if instruction.opname == "LOAD_GLOBAL" and instruction.argval == "_gateway_message_mode"
    ]

    outer_source = inspect.getsource(GatewayRunner._handle_message)
    assert "message_mode=_gateway_message_mode" in outer_source


def test_default_message_stays_dev_mode():
    route = resolve_gateway_message_mode("接着开发招生 CRM")

    assert route.name == "dev"
    assert route.message == "接着开发招生 CRM"
    assert route.session_scope is None
    assert route.enabled_toolsets is None
    assert route.required_skills == ("using-superpowers", "project-dev-workflow")
    assert route.skip_context_files is False
    assert route.skip_memory is False


def test_non_dev_mode_default_does_not_inherit_dev_workflow_skill():
    route = GatewayMessageMode(name="ops", message="检查 gateway")

    assert route.required_skills == ("using-superpowers",)


def test_ops_required_route_skill_preload_does_not_include_dev_workflow(monkeypatch):
    from gateway.run import _prepare_route_required_skills

    calls = []

    def fake_skill_view(name, task_id=None, side_effect_free=False, **_kwargs):
        calls.append((name, task_id, side_effect_free))
        return json.dumps({"success": True, "name": name, "content": f"# {name}\nloaded"})

    monkeypatch.setattr("tools.skills_tool.skill_view", fake_skill_view)
    route = GatewayMessageMode(name="ops", message="检查 gateway")

    enabled_toolsets, combined_prompt = _prepare_route_required_skills(
        ["terminal"],
        "base prompt",
        route,
        task_id="task-ops",
    )

    assert "skills" in enabled_toolsets
    assert "skill_view(name=\"using-superpowers\")" in combined_prompt
    assert "project-dev-workflow" not in combined_prompt
    assert calls == [("using-superpowers", "task-ops", False)]


def test_lite_prefix_strips_text_and_limits_context_and_tools():
    route = resolve_gateway_message_mode("日常聊天 总结这个抖音视频")

    assert route.name == "lite"
    assert route.prefix == "日常聊天"
    assert route.message == "总结这个抖音视频"
    assert route.session_scope == "lite"
    assert route.skip_context_files is True
    assert route.load_soul_identity is True
    assert route.skip_memory is True
    assert route.enabled_toolsets == ("web", "vision", "image_gen", "session_search", "clarify", "skills")
    assert route.required_skills == (
        "yume-skill/using-superpowers",
        "yume-skill/verification-before-completion",
    )
    assert route.sticky_mode == "lite"
    assert route.control_response == ""
    assert "lightweight" in route.system_prompt
    assert "nearest visible topic" in route.system_prompt
    assert "limit=1" in route.system_prompt
    assert "Avoid broad `OR` history searches" in route.system_prompt


def test_lite_prefix_accepts_plus_separator():
    route = resolve_gateway_message_mode("日常聊天+1+1=？")

    assert route.name == "lite"
    assert route.message == "1+1=？"


def test_ops_prefix_strips_text_and_selects_ops_tools():
    route = resolve_gateway_message_mode("运维 看一下 QQ bot 延迟")

    assert route.name == "ops"
    assert route.prefix == "运维"
    assert route.message == "看一下 QQ bot 延迟"
    assert route.session_scope == "ops"
    assert route.skip_context_files is True
    assert route.load_soul_identity is True
    assert route.skip_memory is False
    assert "terminal" in route.enabled_toolsets
    assert "skills" in route.enabled_toolsets
    assert route.required_skills == ("using-superpowers",)
    assert route.sticky_mode == "ops"
    assert route.control_response == ""
    assert "Hermes operations" in route.system_prompt


def test_content_prefix_strips_text_and_selects_content_creation_route():
    route = resolve_gateway_message_mode("内容创意路由 3-6年级奥数第一天发什么")

    assert route.name == "content"
    assert route.prefix == "内容创意路由"
    assert route.message == "3-6年级奥数第一天发什么"
    assert route.session_scope == "content"
    assert route.skip_context_files is True
    assert route.load_soul_identity is True
    assert route.skip_memory is False
    assert route.enabled_toolsets == (
        "web",
        "vision",
        "image_gen",
        "video",
        "session_search",
        "clarify",
    )
    assert route.required_skills == (
        "brand-content-marketing-advisor",
        "content-trend-radar",
        "hook-title-lab",
        "xhs-graphic-generator",
        "douyin-script-director",
    )
    assert route.expose_skill_tools is False
    assert "project-dev-workflow" not in route.required_skills
    assert "内容创意路由" in route.system_prompt
    assert "publish-ready content" in route.system_prompt
    assert "Xiaohongshu" in route.system_prompt
    assert "Douyin" in route.system_prompt
    assert "does not do development or operations" in route.system_prompt
    assert "Do not write code" in route.system_prompt


def test_content_prefix_short_form_accepts_plus_separator():
    route = resolve_gateway_message_mode("内容创意+小红书图文怎么写")

    assert route.name == "content"
    assert route.prefix == "内容创意"
    assert route.message == "小红书图文怎么写"


def test_dev_prefix_switches_back_to_default_route_and_strips_text():
    route = resolve_gateway_message_mode("开发 继续做网关修复", active_mode="lite")

    assert route.name == "dev"
    assert route.message == "继续做网关修复"
    assert route.sticky_mode == "dev"
    assert route.control_response == ""
    assert route.session_scope is None
    assert route.required_skills == ("using-superpowers", "project-dev-workflow")


def test_slash_commands_are_not_mode_routed():
    route = resolve_gateway_message_mode("/new")

    assert route.name == "dev"
    assert route.message == "/new"
    assert route.session_scope is None
    assert route.required_skills == ("using-superpowers",)


def test_sticky_lite_mode_routes_followup_without_prefix():
    route = resolve_gateway_message_mode("1+1=？", active_mode="lite")

    assert route.name == "lite"
    assert route.message == "1+1=？"
    assert route.session_scope == "lite"
    assert route.skip_context_files is True
    assert route.skip_memory is True


def test_lite_hermes_intent_lazily_adds_hermes_skill():
    route = resolve_gateway_message_mode("日常聊天 Hermes gateway 路由怎么配置")

    assert route.name == "lite"
    assert route.message == "Hermes gateway 路由怎么配置"
    assert route.required_skills == (
        "yume-skill/using-superpowers",
        "yume-skill/verification-before-completion",
        "autonomous-ai-agents/hermes-agent",
    )


def test_lite_qq_intent_lazily_adds_qq_control_skill():
    route = resolve_gateway_message_mode("日常聊天 QQ 私聊和群聊怎么区分")

    assert route.name == "lite"
    assert route.required_skills == (
        "yume-skill/using-superpowers",
        "yume-skill/verification-before-completion",
        "qq-control-console",
    )


def test_lite_debug_intent_lazily_adds_systematic_debugging_skill():
    route = resolve_gateway_message_mode("日常聊天 为什么报错失败")

    assert route.name == "lite"
    assert route.required_skills == (
        "yume-skill/using-superpowers",
        "yume-skill/verification-before-completion",
        "yume-skill/systematic-debugging",
    )


def test_lite_execution_task_escalates_to_dev_route_and_strips_prefix():
    route = resolve_gateway_message_mode("日常聊天 建一个 GitHub 库")

    assert route.name == "dev"
    assert route.message == "建一个 GitHub 库"
    assert route.session_scope is None
    assert route.required_skills == ("using-superpowers", "project-dev-workflow")
    assert route.sticky_mode == "dev"


def test_sticky_lite_execution_followup_escalates_to_dev_route():
    route = resolve_gateway_message_mode("修复 lite 路由", active_mode="lite")

    assert route.name == "dev"
    assert route.message == "修复 lite 路由"
    assert route.session_scope is None
    assert route.sticky_mode == "dev"


def test_switch_to_lite_mode_sets_sticky_control_response():
    route = resolve_gateway_message_mode("切到日常聊天路由")

    assert route.name == "lite"
    assert route.sticky_mode == "lite"
    assert "日常聊天" in route.control_response


def test_switch_to_content_mode_sets_sticky_control_response():
    route = resolve_gateway_message_mode("切到内容创意路由")

    assert route.name == "content"
    assert route.sticky_mode == "content"
    assert "内容创意" in route.control_response


def test_sticky_content_mode_routes_followup_without_prefix():
    route = resolve_gateway_message_mode("帮我做小红书图文和抖音脚本", active_mode="content")

    assert route.name == "content"
    assert route.message == "帮我做小红书图文和抖音脚本"
    assert route.session_scope == "content"
    assert route.required_skills == (
        "brand-content-marketing-advisor",
        "content-trend-radar",
        "hook-title-lab",
        "xhs-graphic-generator",
        "douyin-script-director",
    )


def test_content_route_does_not_auto_escalate_development_requests():
    route = resolve_gateway_message_mode("内容创意 改代码实现这个路由")

    assert route.name == "content"
    assert route.message == "改代码实现这个路由"
    assert route.session_scope == "content"
    assert "terminal" not in route.enabled_toolsets
    assert "file" not in route.enabled_toolsets
    assert "skills" not in route.enabled_toolsets
    assert "todo" not in route.enabled_toolsets
    assert "project-dev-workflow" not in route.required_skills
    assert "outside 内容创意路由" in route.system_prompt


def test_explicit_dev_prefix_can_leave_sticky_content_route():
    route = resolve_gateway_message_mode("开发 继续改代码", active_mode="content")

    assert route.name == "dev"
    assert route.message == "继续改代码"
    assert route.sticky_mode == "dev"
    assert route.session_scope is None


def test_routed_event_uses_stripped_text_and_scoped_session_key():
    source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    event = MessageEvent(text="日常聊天：帮我解释 Harness", source=source)

    routed_event, route = _route_gateway_message_mode(event)

    assert route.name == "lite"
    assert routed_event.text == "帮我解释 Harness"
    assert routed_event.source.chat_id == "chat-1"
    assert routed_event.source.session_scope == "lite"
    assert build_session_key(routed_event.source) == "agent:main:qqbot:dm:chat-1:mode:lite"
    assert build_session_key(source) == "agent:main:qqbot:dm:chat-1"


def test_threaded_dm_scoped_key_extends_base_key_for_command_lookup():
    source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        thread_id="topic-1",
        chat_type="dm",
        user_id="user-1",
    )
    scoped_source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        thread_id="topic-1",
        chat_type="dm",
        user_id="user-1",
        session_scope="lite",
    )

    base_key = build_session_key(source)
    scoped_key = build_session_key(scoped_source)

    assert base_key == "agent:main:qqbot:dm:chat-1:topic-1"
    assert scoped_key == "agent:main:qqbot:dm:chat-1:topic-1:mode:lite"
    assert scoped_key.startswith(f"{base_key}:mode:")


def test_sticky_route_scopes_event_without_repeating_prefix():
    source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    event = MessageEvent(text="1+1=？", source=source)

    routed_event, route = _route_gateway_message_mode(event, active_mode="lite")

    assert route.name == "lite"
    assert routed_event.text == "1+1=？"
    assert routed_event.source.session_scope == "lite"
    assert build_session_key(routed_event.source) == "agent:main:qqbot:dm:chat-1:mode:lite"
    assert build_session_key(source) == "agent:main:qqbot:dm:chat-1"


def test_dev_prefix_route_keeps_default_session_and_strips_prefix():
    source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    event = MessageEvent(text="开发 继续做网关修复", source=source)

    routed_event, route = _route_gateway_message_mode(event, active_mode="lite")

    assert route.name == "dev"
    assert route.sticky_mode == "dev"
    assert routed_event.text == "继续做网关修复"
    assert routed_event.source.session_scope is None
    assert build_session_key(routed_event.source) == "agent:main:qqbot:dm:chat-1"


def test_ops_and_default_sessions_are_isolated_for_same_chat():
    base = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    ops = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        session_scope="ops",
    )

    assert build_session_key(base) == "agent:main:qqbot:dm:chat-1"
    assert build_session_key(ops) == "agent:main:qqbot:dm:chat-1:mode:ops"


def test_content_and_default_sessions_are_isolated_for_same_chat():
    base = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    content = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        session_scope="content",
    )

    assert build_session_key(base) == "agent:main:qqbot:dm:chat-1"
    assert build_session_key(content) == "agent:main:qqbot:dm:chat-1:mode:content"


def test_content_routed_event_uses_scoped_session_key():
    source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    event = MessageEvent(text="内容创意：给我小红书图文和抖音视频内容", source=source)

    routed_event, route = _route_gateway_message_mode(event)

    assert route.name == "content"
    assert routed_event.text == "给我小红书图文和抖音视频内容"
    assert routed_event.source.session_scope == "content"
    assert build_session_key(routed_event.source) == "agent:main:qqbot:dm:chat-1:mode:content"


def test_sticky_route_keys_do_not_bleed_between_dm_sources_without_chat_id():
    first = SessionSource(
        platform=Platform.QQBOT,
        chat_id="",
        chat_type="dm",
        user_id="user-1",
    )
    second = SessionSource(
        platform=Platform.QQBOT,
        chat_id="",
        chat_type="dm",
        user_id="user-2",
    )

    first_key = build_session_key(first)
    second_key = build_session_key(second)

    assert first_key == "agent:main:qqbot:dm:user-1"
    assert second_key == "agent:main:qqbot:dm:user-2"
    assert first_key != second_key


def test_scoped_group_route_key_is_suffix_of_its_base_conversation_key():
    base = SessionSource(
        platform=Platform.QQBOT,
        chat_id="group-1",
        chat_type="group",
        user_id="user-1",
    )
    ops = SessionSource(
        platform=Platform.QQBOT,
        chat_id="group-1",
        chat_type="group",
        user_id="user-1",
        session_scope="ops",
    )
    other_user_ops = SessionSource(
        platform=Platform.QQBOT,
        chat_id="group-1",
        chat_type="group",
        user_id="user-2",
        session_scope="ops",
    )

    base_key = build_session_key(base)

    assert base_key == "agent:main:qqbot:group:group-1:user-1"
    assert build_session_key(ops) == f"{base_key}:mode:ops"
    assert build_session_key(other_user_ops) == "agent:main:qqbot:group:group-1:user-2:mode:ops"


def test_content_route_skills_are_preloaded_without_dev_or_ops_skills(monkeypatch):
    from gateway.run import _prepare_route_required_skills

    calls = []

    def fake_skill_view(name, file_path=None, task_id=None, preprocess=True):
        calls.append((name, file_path, task_id, preprocess))
        return json.dumps({"success": True, "name": name, "content": f"# {name}\nloaded"})

    monkeypatch.setattr("tools.skills_tool.skill_view", fake_skill_view)
    route = resolve_gateway_message_mode("内容创意 做一套小红书图文和抖音脚本")

    enabled_toolsets, combined_prompt = _prepare_route_required_skills(
        ["web"],
        "base prompt",
        route,
        task_id="task-content",
    )

    assert enabled_toolsets == ["web"]
    assert "Required route skills" in combined_prompt
    assert "skill_view(name=\"brand-content-marketing-advisor\")" in combined_prompt
    assert "skill_view(name=\"content-trend-radar\")" in combined_prompt
    assert "skill_view(name=\"hook-title-lab\")" in combined_prompt
    assert "skill_view(name=\"xhs-graphic-generator\")" in combined_prompt
    assert "skill_view(name=\"douyin-script-director\")" in combined_prompt
    assert "project-dev-workflow" not in combined_prompt
    assert "hermes-runtime-ops" not in combined_prompt
    assert calls == [
        ("brand-content-marketing-advisor", None, "task-content", False),
        ("content-trend-radar", None, "task-content", False),
        ("hook-title-lab", None, "task-content", False),
        ("xhs-graphic-generator", None, "task-content", False),
        ("douyin-script-director", None, "task-content", False),
    ]


@pytest.mark.parametrize(
    ("message", "expected_skill"),
    [
        ("内容创意 小红书封面视觉排版怎么做", "xhs-visual-director"),
        ("内容创意 把这段内容做成社媒卡片", "social-card-composer"),
        ("内容创意 复盘这周笔记数据和评论反馈", "content-feedback-loop"),
        ("内容创意 拆解这个短视频为什么火", "video-analysis-director"),
        ("内容创意 找南沙本地达人和教育博主合作", "influencer-discovery-advisor"),
        ("内容创意 把这篇文案改得更像人话，去AI味", "humanizer"),
        ("内容创意 做一张数学思维图解信息图", "baoyu-infographic"),
        ("内容创意 做成四格知识漫画", "baoyu-comic"),
        ("内容创意 给这篇文章统一插画和配图风格", "baoyu-article-illustrator"),
    ],
)
def test_content_route_intent_based_preload_adds_specialist_skill(message, expected_skill):
    route = resolve_gateway_message_mode(message)

    assert route.name == "content"
    assert expected_skill in route.required_skills
    assert "project-dev-workflow" not in route.required_skills
    assert "hermes-runtime-ops" not in route.required_skills


def test_required_route_skill_missing_adds_explicit_feedback_to_prompt(monkeypatch):
    from gateway.run import _prepare_route_required_skills

    def fake_skill_view(name, file_path=None, task_id=None, preprocess=True):
        if name == "xhs-visual-director":
            return json.dumps({"success": False, "error": "not installed"})
        return json.dumps({"success": True, "name": name, "content": f"# {name}\nloaded"})

    monkeypatch.setattr("tools.skills_tool.skill_view", fake_skill_view)
    route = resolve_gateway_message_mode("内容创意 小红书封面视觉排版怎么做")

    enabled_toolsets, combined_prompt = _prepare_route_required_skills(
        ["web"],
        "base prompt",
        route,
        task_id="task-content-missing",
    )

    assert enabled_toolsets == ["web"]
    assert "Required route skill preload warning" in combined_prompt
    assert "xhs-visual-director" in combined_prompt
    assert "not installed" in combined_prompt
    assert "先简短告诉用户" in combined_prompt


@pytest.mark.asyncio
async def test_content_route_end_to_end_passes_intent_skills_to_aiagent(monkeypatch):
    from collections import OrderedDict
    from types import SimpleNamespace

    import run_agent as run_agent_mod
    import gateway.run as gateway_run

    captured = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.tools = [{"name": "web_search"}]
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id")
            self.session_prompt_tokens = 0
            self.session_completion_tokens = 0
            self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=0)

        def run_conversation(self, message, **kwargs):
            captured["run_message"] = message
            captured["run_kwargs"] = kwargs
            return {
                "final_response": "内容产出完成",
                "messages": [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "内容产出完成"},
                ],
                "api_calls": 1,
                "completed": True,
            }

    def fake_skill_view(name, file_path=None, task_id=None, preprocess=True):
        captured.setdefault("skills", []).append(name)
        return json.dumps({"success": True, "name": name, "content": f"# {name}\nloaded"})

    monkeypatch.setattr(run_agent_mod, "AIAgent", FakeAIAgent)
    monkeypatch.setattr("tools.skills_tool.skill_view", fake_skill_view)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "agent": {"disabled_toolsets": []},
            "display": {
                "tool_progress": "off",
                "long_running_notifications": False,
                "interim_assistant_messages": False,
            },
        },
    )

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = SimpleNamespace(streaming=SimpleNamespace(enabled=False, transport="off"))
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.session_store = SimpleNamespace(_entries={})
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = None
    runner._draining = False
    runner._ephemeral_system_prompt = ""
    runner._fallback_model = None
    runner._pending_model_notes = {}
    runner._pending_skills_reload_notes = {}
    runner._prefill_messages = None
    runner._provider_routing = {}
    runner._reasoning_config = None
    runner._service_tier = None
    runner._session_db = None
    runner._running_agents = {}
    runner._get_proxy_url = lambda: None
    runner._resolve_session_agent_runtime = lambda **_kwargs: (
        "test-model",
        {"provider": "test-provider", "base_url": "https://example.invalid", "api_key": "test-key"},
    )
    runner._resolve_session_reasoning_config = lambda **_kwargs: None
    runner._load_service_tier = lambda: None
    runner._resolve_turn_agent_config = lambda message, model, runtime: {
        "model": model,
        "runtime": runtime,
        "request_overrides": {},
    }
    runner._thread_metadata_for_source = lambda source, event_message_id=None: None
    runner._consume_pending_native_image_paths = lambda session_key: []
    runner._is_session_run_current = lambda session_key, run_generation: True
    runner._update_runtime_status = lambda status: None

    async def run_inline(fn):
        return fn()

    runner._run_in_executor_with_context = run_inline

    source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    event = MessageEvent(text="内容创意 小红书封面视觉排版怎么做", source=source)
    routed_event, route = _route_gateway_message_mode(event)
    session_key = build_session_key(routed_event.source)

    result = await GatewayRunner._run_agent(
        runner,
        routed_event.text,
        "base context",
        [],
        routed_event.source,
        "sess-content",
        session_key=session_key,
        message_mode=route,
    )

    assert result["final_response"] == "内容产出完成"
    assert captured["run_message"] == "小红书封面视觉排版怎么做"
    assert captured["run_kwargs"] == {"conversation_history": [], "task_id": "sess-content"}
    assert captured["init"]["enabled_toolsets"] == sorted(route.enabled_toolsets)
    assert captured["init"]["skip_context_files"] is True
    assert captured["init"]["load_soul_identity"] is True
    assert captured["init"]["skip_memory"] is False
    assert captured["init"]["platform"] == "qqbot"
    assert captured["init"]["gateway_session_key"] == "agent:main:qqbot:dm:chat-1:mode:content"
    assert "skill_view(name=\"xhs-visual-director\")" in captured["init"]["ephemeral_system_prompt"]
    assert "skill_view(name=\"project-dev-workflow\")" not in captured["init"]["ephemeral_system_prompt"]
    assert captured["skills"] == list(route.required_skills)


def test_required_route_skills_are_preloaded_and_force_skills_toolset(monkeypatch):
    from gateway.message_modes import GatewayMessageMode
    from gateway.run import _prepare_route_required_skills

    calls = []

    def fake_skill_view(name, file_path=None, task_id=None, preprocess=True):
        calls.append((name, file_path, task_id, preprocess))
        return json.dumps({"success": True, "name": name, "content": f"# {name}\nloaded"})

    monkeypatch.setattr("tools.skills_tool.skill_view", fake_skill_view)

    route = resolve_gateway_message_mode("继续开发")

    enabled_toolsets, combined_prompt = _prepare_route_required_skills(
        ["terminal", "file"],
        "base prompt",
        route,
        task_id="task-1",
    )

    assert enabled_toolsets == ["terminal", "file", "skills"]
    assert "base prompt" in combined_prompt
    assert "Required route skills" in combined_prompt
    assert "skill_view(name=\"using-superpowers\")" in combined_prompt
    assert "skill_view(name=\"project-dev-workflow\")" in combined_prompt
    assert "# using-superpowers\nloaded" in combined_prompt
    assert "# project-dev-workflow\nloaded" in combined_prompt
    assert calls == [
        ("using-superpowers", None, "task-1", False),
        ("project-dev-workflow", None, "task-1", False),
    ]


def test_route_without_required_skills_does_not_force_skills_toolset(monkeypatch):
    from gateway.message_modes import GatewayMessageMode
    from gateway.run import _prepare_route_required_skills

    def fail_skill_view(*args, **kwargs):
        raise AssertionError("skill_view should not be called")

    monkeypatch.setattr("tools.skills_tool.skill_view", fail_skill_view)

    route = GatewayMessageMode(name="lite", message="闲聊", required_skills=())

    enabled_toolsets, combined_prompt = _prepare_route_required_skills(
        ["web", "clarify"],
        "base prompt",
        route,
    )

    assert enabled_toolsets == ["web", "clarify"]
    assert combined_prompt == "base prompt"


class _ApprovalTestAdapter:
    def __init__(self):
        self.resumed_chats = []

    def resume_typing_for_chat(self, chat_id):
        self.resumed_chats.append(chat_id)


class _CommandScopeSessionStore:
    def get_or_create_session(self, source):
        from types import SimpleNamespace

        return SimpleNamespace(session_key=build_session_key(source), session_id="sess-1")

    def load_transcript(self, session_id):
        return []


def _approval_test_runner():
    from collections import OrderedDict
    from types import SimpleNamespace
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.session_store = _CommandScopeSessionStore()
    runner.config = SimpleNamespace(
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner.adapters = {Platform.QQBOT: _ApprovalTestAdapter()}
    runner._pending_approvals = {}
    runner._gateway_message_mode_overrides = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = None
    runner._session_db = None
    return runner


def _queue_blocking_approval(session_key: str):
    from tools import approval as approval_mod

    entry = approval_mod._ApprovalEntry({
        "command": "rm -rf /tmp/demo",
        "pattern_key": "recursive delete",
        "pattern_keys": ["recursive delete"],
        "description": "recursive delete",
    })
    with approval_mod._lock:
        approval_mod._gateway_queues[session_key] = [entry]
    return entry


def _clear_blocking_approvals():
    from tools import approval as approval_mod

    with approval_mod._lock:
        approval_mod._gateway_queues.clear()


@pytest.mark.asyncio
async def test_plain_approve_resolves_ops_scoped_pending_approval():
    runner = _approval_test_runner()
    source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    scoped_key = build_session_key(SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        session_scope="ops",
    ))
    entry = _queue_blocking_approval(scoped_key)

    try:
        response = await runner._handle_approve_command(MessageEvent(text="/approve", source=source))
    finally:
        _clear_blocking_approvals()

    assert entry.result == "once"
    assert entry.event.is_set()
    assert "批准" in response or "approved" in response.lower()
    assert runner.adapters[Platform.QQBOT].resumed_chats == ["chat-1"]


@pytest.mark.asyncio
async def test_plain_deny_resolves_ops_scoped_pending_approval():
    runner = _approval_test_runner()
    source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    scoped_key = build_session_key(SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        session_scope="ops",
    ))
    entry = _queue_blocking_approval(scoped_key)

    try:
        response = await runner._handle_deny_command(MessageEvent(text="/deny", source=source))
    finally:
        _clear_blocking_approvals()

    assert entry.result == "deny"
    assert entry.event.is_set()
    assert "拒绝" in response or "denied" in response.lower()
    assert runner.adapters[Platform.QQBOT].resumed_chats == ["chat-1"]


def _base_and_ops_sources():
    base = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    ops = SessionSource(
        platform=Platform.QQBOT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        session_scope="ops",
    )
    return base, ops, build_session_key(base), build_session_key(ops)


@pytest.mark.asyncio
async def test_plain_stop_targets_sticky_ops_running_session(monkeypatch):
    runner = _approval_test_runner()
    base, _ops, base_key, ops_key = _base_and_ops_sources()
    runner._gateway_message_mode_overrides[base_key] = "ops"
    runner._running_agents[ops_key] = object()

    called = {}

    async def fake_interrupt(session_key, source, **kwargs):
        called["session_key"] = session_key
        called["chat_id"] = source.chat_id
        called["kwargs"] = kwargs

    monkeypatch.setattr(runner, "_interrupt_and_clear_session", fake_interrupt)

    response = await runner._handle_stop_command(MessageEvent(text="/stop", source=base))

    assert called["session_key"] == ops_key
    assert called["chat_id"] == "chat-1"
    assert called["kwargs"]["interrupt_reason"]
    assert "停止" in str(response) or "stopped" in str(response).lower()


@pytest.mark.asyncio
async def test_plain_yolo_toggles_sticky_ops_session_not_base():
    from tools import approval as approval_mod

    runner = _approval_test_runner()
    base, _ops, base_key, ops_key = _base_and_ops_sources()
    runner._gateway_message_mode_overrides[base_key] = "ops"
    approval_mod.clear_session(base_key)
    approval_mod.clear_session(ops_key)

    try:
        await runner._handle_yolo_command(MessageEvent(text="/yolo", source=base))
        assert not approval_mod.is_session_yolo_enabled(base_key)
        assert approval_mod.is_session_yolo_enabled(ops_key)
    finally:
        approval_mod.clear_session(base_key)
        approval_mod.clear_session(ops_key)


@pytest.mark.asyncio
async def test_plain_usage_reads_sticky_ops_running_agent():
    from types import SimpleNamespace

    runner = _approval_test_runner()
    base, _ops, base_key, ops_key = _base_and_ops_sources()
    runner._gateway_message_mode_overrides[base_key] = "ops"
    runner._running_agents[ops_key] = SimpleNamespace(
        provider=None,
        base_url=None,
        api_key=None,
        model="ops-model",
        session_total_tokens=123,
        session_api_calls=1,
        session_input_tokens=10,
        session_output_tokens=5,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        get_rate_limit_state=lambda: None,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=0,
            context_length=0,
            compression_count=0,
        ),
    )

    response = await runner._handle_usage_command(MessageEvent(text="/usage", source=base))

    assert "ops-model" in response
    assert "123" in response
