"""Red-first tests for work_depth decay fix.

Expectations derived INDEPENDENTLY from the intended semantics:
- keyword hits in OLD user messages must not keep work_depth > 0 forever
- tool activity (role=tool / assistant tool_calls) is real work history and stays
- keyword-only histories decay once the task has been over for a full turn
"""
import importlib.util
import os

_PLUGIN = os.path.join(
    os.path.dirname(__file__), "..", "..", "plugins", "adaptive-reasoning", "__init__.py"
)
_spec = importlib.util.spec_from_file_location("ar_red", _PLUGIN)
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)


def test_old_keyword_messages_alone_do_not_hold_work_depth():
    # Hard task asked long ago, answered, done. History carries only its
    # keyword-bearing user message. New simple question must NOT see work_depth>0.
    msgs = [
        {"role": "user", "content": "Help me debug this complicated race condition"},
        {"role": "assistant", "content": "Fixed: added a lock."},
        {"role": "user", "content": "今天天气怎样"},
    ]
    # 期望: 0 — 任务已完成一轮问答,旧关键词不再计入历史工作信号(推导:仅关键词无工具活动=话题线索,非进行中工作)
    assert ar._session_work_depth(msgs) == 0  # 期望: 0


def test_recent_tool_activity_keeps_depth():
    msgs = [
        {"role": "user", "content": "debug the failing test"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "ok"},
    ]
    # 期望: >=2 — 工具调用+工具结果是真实工作历史,永不衰减(推导:工具活动=客观进行中证据)
    assert ar._session_work_depth(msgs) >= 2  # 期望: >=2


def test_keyword_only_history_after_full_turn_decays():
    # A past task whose user message carried keywords, followed by a full
    # unrelated exchange (question + answer) — the work signal must be gone.
    msgs = [
        {"role": "user", "content": "optimize the ETL pipeline performance"},
        {"role": "assistant", "content": "Done, index added."},
        {"role": "user", "content": "午饭吃什么好"},
        {"role": "assistant", "content": "推荐拉面。"},
    ]
    # 期望: 0 — 完整无关一轮问答后,旧任务关键词信号必须已衰减(推导:这正是kvnloo点名的sticky complexity形态)
    assert ar._session_work_depth(msgs) == 0  # 期望: 0
