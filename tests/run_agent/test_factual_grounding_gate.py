from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "test-model"
    agent.session_id = "test-session"
    agent._factual_grounding_gate = True
    agent._strict_verbatim_evidence = True
    agent._strict_verbatim_exact_only = True
    return agent


def test_factual_grounding_blocks_missing_file_claim():
    agent = _bare_agent()
    messages = [
        {"role": "tool", "content": "ls: cannot access '/opt/pytools/json/v38_filter_analysis.json': No such file or directory"},
        {"role": "assistant", "content": "文件已确认存在：/opt/pytools/json/v38_filter_analysis.json"},
    ]
    out = AIAgent._apply_factual_grounding_gate_if_needed(
        agent,
        original_user_message="这个文件存在吗",
        final_response="文件已确认存在：/opt/pytools/json/v38_filter_analysis.json",
        interrupted=False,
        messages=messages,
    )
    assert out is not None
    assert "不能把该结论当作已验证事实输出" in out


def test_factual_grounding_allows_verified_file_claim_with_citation():
    agent = _bare_agent()
    messages = [
        {"role": "tool", "content": "-rw-r--r-- 1 root root 2416 May 1 11:00 /opt/pytools/json/v38_filter_analysis.json"},
        {"role": "assistant", "content": "文件已确认存在：/opt/pytools/json/v38_filter_analysis.json"},
    ]
    out = AIAgent._apply_factual_grounding_gate_if_needed(
        agent,
        original_user_message="这个文件存在吗",
        final_response="文件已确认存在：/opt/pytools/json/v38_filter_analysis.json",
        interrupted=False,
        messages=messages,
    )
    assert out is not None
    assert "证据引用（工具实际输出）" in out
    assert "/opt/pytools/json/v38_filter_analysis.json" in out


def test_factual_grounding_blocks_unverified_statistical_claims():
    agent = _bare_agent()
    messages = [
        {"role": "tool", "content": "scan completed, but no numeric summary emitted"},
        {"role": "assistant", "content": "最终通过率：0.29%（26/8880）"},
    ]
    out = AIAgent._apply_factual_grounding_gate_if_needed(
        agent,
        original_user_message="帮我统计结果",
        final_response="最终通过率：0.29%（26/8880）",
        interrupted=False,
        messages=messages,
    )
    assert out is not None
    assert "不能输出这些统计数字结论" in out


def test_factual_grounding_allows_verified_statistical_claims():
    agent = _bare_agent()
    messages = [
        {"role": "tool", "content": "总候选 8,880 通过 26 最终通过率 0.29%"},
        {"role": "assistant", "content": "最终通过率：0.29%（26/8880）"},
    ]
    out = AIAgent._apply_factual_grounding_gate_if_needed(
        agent,
        original_user_message="帮我统计结果",
        final_response="最终通过率：0.29%（26/8880）",
        interrupted=False,
        messages=messages,
    )
    assert out is not None
    assert "不能输出这些统计数字结论" not in out
    assert "证据引用（工具实际输出）" in out


def test_strict_verbatim_appends_raw_tool_snippet_when_user_requests_raw():
    agent = _bare_agent()
    messages = [
        {"role": "tool", "content": "total 4\n-rw-r--r-- 1 root root 2416 v38_filter_analysis.json"},
        {"role": "assistant", "content": "文件已确认存在"},
    ]
    out = AIAgent._apply_factual_grounding_gate_if_needed(
        agent,
        original_user_message="把结果原样发给我，不要总结",
        final_response="文件已确认存在",
        interrupted=False,
        messages=messages,
    )
    assert out is not None
    assert "原样工具输出" in out
    assert "v38_filter_analysis.json" in out
