import pytest

from agent import backpack_advisor
from agent.backpack_advisor import (
    BACKPACK_ADVISOR_STRATEGY_VERSION,
    build_backpack_candidate_hints,
    build_candidate_hints,
    should_build_backpack_candidate_hints,
)


def test_grouped_strategy_has_formal_version():
    assert BACKPACK_ADVISOR_STRATEGY_VERSION == "grouped-hints-v1"


@pytest.mark.parametrize(
    ("message", "case_label"),
    [
        ("", "empty"),
        ("   ", "blank"),
        ("?", "punctuation"),
        ("hello", "social"),
        ("你好", "social"),
        ("早上好", "social"),
        ("thanks", "thanks"),
        ("谢谢", "thanks"),
        ("ok", "ack"),
        ("好的", "ack"),
        ("收到", "ack"),
        ("嗯", "ack"),
        ("继续", "bare continuation"),
        ("继续吧", "bare continuation"),
        ("什么意思", "underspecified chat"),
        ("这个是什么意思", "underspecified chat"),
        ("你怎么看", "general chat"),
        ("这样可以吗", "general chat"),
        ("你是谁", "identity chat"),
        ("how are you", "social"),
    ],
)
def test_should_not_build_hints_for_non_task_chat(message, case_label):
    assert should_build_backpack_candidate_hints(message) is False


@pytest.mark.parametrize(
    ("message", "case_label"),
    [
        ("README.md", "bare file path"),
        ("https://example.com", "bare url"),
        ("Traceback in parser", "error shape"),
        ("出现异常", "cjk error shape"),
        ("继续跑测试", "continuation with action"),
        ("继续修刚才那个失败", "continuation with task"),
        ("继续查 metadata", "continuation with search"),
        ("什么意思，解释 README.md", "question with file object"),
        ("读一下 README.md", "file read"),
        ("打开 core/protocol.md", "file read"),
        ("搜索 Backpack metadata", "search"),
        ("查找 metadata", "search"),
        ("修一下 parser 的失败测试", "edit failure"),
        ("实现这个 edge case", "implement"),
        ("运行测试", "terminal"),
        ("验证 package structure", "verify"),
        ("这个测试失败怎么办", "uncertain failure"),
        ("这个报错什么意思", "uncertain error"),
        ("这个函数要不要改", "uncertain code object"),
        ("不知道该选哪个方案", "clarification workflow"),
        ("这个需求不清楚，先问我", "clarification workflow"),
        ("之前我们怎么处理 Backpack 的", "session history"),
        ("看一下 https://example.com", "browser"),
        ("分析这张截图", "vision"),
        ("列个任务清单", "todo"),
        ("派一个子 agent 调查", "delegate"),
        ("use the skill for debugging", "skill explicit"),
        ("which tool should handle this log error", "tool uncertainty"),
    ],
)
def test_should_build_hints_for_external_capability_requests(message, case_label):
    assert should_build_backpack_candidate_hints(message) is True


def test_skipped_requests_do_not_load_backpack_catalogs(monkeypatch):
    monkeypatch.setattr(backpack_advisor, "_tool_catalog", lambda: (_ for _ in ()).throw(AssertionError("loaded tools")))
    monkeypatch.setattr(backpack_advisor, "_skill_catalog", lambda: (_ for _ in ()).throw(AssertionError("loaded skills")))

    for message in ["你好", "谢谢", "ok", "继续", "什么意思", "你怎么看"]:
        assert build_backpack_candidate_hints(message, {"tool_backpack", "skill_backpack"}) == ""


@pytest.mark.parametrize("message", ["README.md", "这个测试失败怎么办", "继续跑测试"])
def test_trigger_requests_load_backpack_catalogs(monkeypatch, message):
    calls = {"tools": 0, "skills": 0}

    def tool_catalog():
        calls["tools"] += 1
        return []

    def skill_catalog():
        calls["skills"] += 1
        return []

    monkeypatch.setattr(backpack_advisor, "_tool_catalog", tool_catalog)
    monkeypatch.setattr(backpack_advisor, "_skill_catalog", skill_catalog)

    build_backpack_candidate_hints(message, {"tool_backpack", "skill_backpack"})

    assert calls == {"tools": 1, "skills": 1}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("读一下 README.md", "tool.read_file"),
        ("这个测试失败怎么办", "skill.systematic-debugging"),
        ("不知道该选哪个方案", "tool.clarify"),
        ("继续跑测试", "tool.terminal"),
    ],
)
def test_task_requests_generate_targeted_hints(message, expected):
    catalog = [
        {"id": "tool.read_file", "kind": "tool", "name": "read_file", "description": "Read file contents"},
        {"id": "tool.terminal", "kind": "tool", "name": "terminal", "description": "Run terminal commands and tests"},
        {"id": "tool.clarify", "kind": "tool", "name": "clarify", "description": "Ask clarifying questions"},
        {"id": "skill.systematic-debugging", "kind": "skill", "name": "systematic-debugging", "description": "Diagnose failures and root causes"},
    ]

    hints = build_candidate_hints(message, catalog, limit=5)

    assert expected in hints


def test_build_candidate_hints_prefers_read_file_for_known_path():
    catalog = [
        {"id": "skill.node-inspect-debugger", "kind": "skill", "name": "node-inspect-debugger", "description": "Inspect Node debugger sessions"},
        {"id": "tool.read_file", "kind": "tool", "name": "read_file", "description": "Read file contents"},
    ]

    hints = build_candidate_hints("I need to understand what core/protocol.md says.", catalog, limit=5)

    assert "1. tool.read_file" in hints
    assert "select read_file" in hints
    assert "SKILL.md content" not in hints


def test_file_read_request_returns_grouped_reasoned_tool_set():
    catalog = [
        {"id": "tool.search_files", "kind": "tool", "name": "search_files", "description": "Search names and content"},
        {"id": "tool.read_file", "kind": "tool", "name": "read_file", "description": "Read file contents"},
        {"id": "skill.design-md", "kind": "skill", "name": "design-md", "description": "Review README and SKILL.md design docs"},
    ]

    hints = build_candidate_hints("读一下 README.md", catalog, limit=5)

    assert "Group 1: local file inspection" in hints
    assert "reason: user asked to read a local file." in hints
    assert "select: tool_backpack select read_file,search_files" in hints
    assert hints.index("tool.read_file") < hints.index("tool.search_files")
    assert "skill.design-md" not in hints


def test_file_search_request_returns_search_first_tool_set():
    catalog = [
        {"id": "tool.read_file", "kind": "tool", "name": "read_file", "description": "Read file contents"},
        {"id": "tool.search_files", "kind": "tool", "name": "search_files", "description": "Search names and content"},
    ]

    hints = build_candidate_hints("搜索 README.md", catalog, limit=5)

    assert "Group 1: local file inspection" in hints
    assert "reason: user asked to search local files or repository content." in hints
    assert "select: tool_backpack select search_files,read_file" in hints
    assert hints.index("tool.search_files") < hints.index("tool.read_file")


def test_debug_failure_request_returns_debugging_group_and_repo_tools():
    catalog = [
        {"id": "tool.search_files", "kind": "tool", "name": "search_files", "description": "Search names and content"},
        {"id": "tool.read_file", "kind": "tool", "name": "read_file", "description": "Read file contents"},
        {"id": "tool.terminal", "kind": "tool", "name": "terminal", "description": "Run shell commands and tests"},
        {"id": "skill.systematic-debugging", "kind": "skill", "name": "systematic-debugging", "description": "Diagnose failures and root causes"},
    ]

    hints = build_candidate_hints("这个测试失败怎么办", catalog, limit=5)

    assert "Group 1: debugging workflow" in hints
    assert "reason: user described a failing test or error." in hints
    assert "select: skill_backpack select systematic-debugging" in hints
    assert "Group 2: repo diagnosis tools" in hints
    assert "select: tool_backpack select search_files,read_file,terminal" in hints


def test_english_traceback_request_returns_debugging_group():
    catalog = [
        {"id": "tool.search_files", "kind": "tool", "name": "search_files", "description": "Search names and content"},
        {"id": "skill.systematic-debugging", "kind": "skill", "name": "systematic-debugging", "description": "Diagnose failures and root causes"},
    ]

    hints = build_candidate_hints("Traceback in parser", catalog, limit=5)

    assert "Group 1: debugging workflow" in hints
    assert "skill.systematic-debugging" in hints


def test_grouped_hints_respect_candidate_limit():
    catalog = [
        {"id": "tool.search_files", "kind": "tool", "name": "search_files", "description": "Search names and content"},
        {"id": "tool.read_file", "kind": "tool", "name": "read_file", "description": "Read file contents"},
    ]

    hints = build_candidate_hints("读一下 README.md", catalog, limit=1)

    assert "tool.read_file" in hints
    assert "tool.search_files" not in hints
    assert "select: tool_backpack select read_file" in hints


def test_implementation_request_returns_workflow_and_repo_edit_groups():
    catalog = [
        {"id": "tool.search_files", "kind": "tool", "name": "search_files", "description": "Search names and content"},
        {"id": "tool.read_file", "kind": "tool", "name": "read_file", "description": "Read file contents"},
        {"id": "tool.patch", "kind": "tool", "name": "patch", "description": "Apply file patch"},
        {"id": "tool.terminal", "kind": "tool", "name": "terminal", "description": "Run shell commands and tests"},
        {"id": "skill.test-driven-development", "kind": "skill", "name": "test-driven-development", "description": "Write tests before code changes"},
    ]

    hints = build_candidate_hints("实现这个功能", catalog, limit=5)

    assert "Group 1: implementation workflow" in hints
    assert "select: skill_backpack select test-driven-development" in hints
    assert "Group 2: repo edit tools" in hints
    assert "select: tool_backpack select search_files,read_file,patch,terminal" in hints


def test_url_request_returns_web_lookup_group():
    catalog = [
        {"id": "tool.web_search", "kind": "tool", "name": "web_search", "description": "Search web"},
        {"id": "tool.web_extract", "kind": "tool", "name": "web_extract", "description": "Extract web page"},
        {"id": "tool.browser_navigate", "kind": "tool", "name": "browser_navigate", "description": "Open page"},
    ]

    hints = build_candidate_hints("看一下 https://example.com", catalog, limit=5)

    assert "Group 1: web lookup" in hints
    assert "reason: user provided a URL or asked for current web information." in hints
    assert "select: tool_backpack select web_extract,web_search" in hints
    assert "tool.browser_navigate" not in hints


def test_url_token_without_lookup_intent_does_not_return_web_lookup_group():
    catalog = [
        {"id": "tool.web_search", "kind": "tool", "name": "web_search", "description": "Search web"},
        {"id": "tool.web_extract", "kind": "tool", "name": "web_extract", "description": "Extract web page"},
    ]

    hints = build_candidate_hints("fix URL parser", catalog, limit=5)

    assert "Group 1: web lookup" not in hints


def test_ranked_fallback_candidates_are_still_grouped():
    catalog = [
        {"id": "tool.session_search", "kind": "tool", "name": "session_search", "description": "Search previous sessions"},
        {"id": "skill.backpack-manager", "kind": "skill", "name": "backpack-manager", "description": "Manage Backpack routing"},
    ]

    hints = build_candidate_hints("之前我们怎么处理 Backpack 的", catalog, limit=5)

    assert "Group 1: fallback candidates" in hints
    assert "reason: no specialized group matched; these are the highest-ranked candidates." in hints
    assert "select: tool_backpack select session_search" in hints
    assert "select: skill_backpack select backpack-manager" in hints
    assert "1. tool.session_search - use tool_backpack select session_search" not in hints


def test_build_candidate_hints_limits_output_to_five_candidates():
    catalog = [
        {"id": f"tool.tool_{index}", "kind": "tool", "name": f"tool_{index}", "description": "Search files and read content"}
        for index in range(8)
    ]

    hints = build_candidate_hints("Search files and read content.", catalog, limit=5)

    assert "Group 1: fallback candidates" in hints
    assert "5. tool.tool_" in hints
    assert "6. tool.tool_" not in hints
