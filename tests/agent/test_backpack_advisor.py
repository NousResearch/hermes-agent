import pytest

from agent import backpack_advisor
from agent.backpack_advisor import build_backpack_candidate_hints, build_candidate_hints, should_build_backpack_candidate_hints


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


def test_build_candidate_hints_limits_output_to_five_candidates():
    catalog = [
        {"id": f"tool.tool_{index}", "kind": "tool", "name": f"tool_{index}", "description": "Search files and read content"}
        for index in range(8)
    ]

    hints = build_candidate_hints("Search files and read content.", catalog, limit=5)

    assert hints.count("\n") < 8
    assert "5. tool.tool_" in hints
    assert "6. tool.tool_" not in hints
