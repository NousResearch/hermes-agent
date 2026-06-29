"""Probe 4: 链接稳定性 (错误率 / 重试 / 限流 / 长连接)

检测目标:
  - 连续 N 次请求的错误率
  - 限流 (429) 触发频率
  - 重试是否恢复 (检测"假故障 + 重试可用"模式)
  - 长会话是否丢上下文 (10 轮多轮对话)
  - max_tokens 边界: 请求超出 max 是否正确报错
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from llm_api_probe.probes.client import ProbeClient
from llm_api_probe.probes.models import Provider, ProbeResult


PING_PROMPT = [{"role": "user", "content": "ping"}]


def _short_dialog() -> list[dict]:
    """构造 10 轮多轮对话 (让模型在第 8 轮回忆第 1 轮的内容)。"""
    msgs = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "记住这个数字: 84291567"},
        {"role": "assistant", "content": "好的, 已记住 84291567。"},
        {"role": "user", "content": "再加 1000 呢?"},
        {"role": "assistant", "content": "是 84292567。"},
        {"role": "user", "content": "再减 50000?"},
        {"role": "assistant", "content": "是 84242567。"},
        {"role": "user", "content": "现在这个数是多少?"},
        {"role": "assistant", "content": "是 84242567。"},
        {"role": "user", "content": "最开始那个数还记得吗?"},
    ]
    return msgs


def run(
    provider: Provider,
    model: str,
    *,
    verbose: bool = False,
    burst: int = 10,
    test_long_session: bool = True,
    test_max_tokens_boundary: bool = True,
) -> ProbeResult:
    r = ProbeResult(probe="stability", provider=provider.name)
    client = ProbeClient(provider)

    # 1) burst N 次 ping
    outcomes = []
    error_types = Counter()
    retries_recovered = 0
    retries_failed = 0
    for i in range(burst):
        cr = client.call(model, PING_PROMPT, stream=False, max_tokens=8, temperature=0.0, verbose=verbose)
        if cr.ok:
            outcomes.append("ok")
        else:
            outcomes.append(cr.error_type or "unknown")
            error_types[cr.error_type or "unknown"] += 1
            # 重试一次 (检测"假故障"模式)
            retry = client.call(model, PING_PROMPT, stream=False, max_tokens=8, temperature=0.0, verbose=verbose)
            if retry.ok:
                retries_recovered += 1
            else:
                retries_failed += 1

    ok_count = outcomes.count("ok")
    err_count = burst - ok_count
    r.add("burst_total", burst, "")
    r.add("burst_ok", ok_count, "")
    r.add("burst_error_rate", round(err_count / burst * 100, 1), "%")
    r.add("retries_recovered", retries_recovered, "")
    r.add("retries_failed", retries_failed, "")
    if error_types:
        r.add("error_breakdown", dict(error_types), "")
    r.raw["outcomes"] = outcomes

    if err_count == 0:
        r.find(f"burst {burst} 次全部成功")
    else:
        r.warn(f"burst {burst} 次中有 {err_count} 次失败: {dict(error_types)}")
    if retries_recovered > 0:
        r.warn(f"初次失败但重试恢复 {retries_recovered} 次 — 接口不太稳定")

    # 2) 长会话 (10 轮)
    if test_long_session:
        dialog = _short_dialog()
        cr = client.call(model, dialog, stream=False, max_tokens=128, temperature=0.0, verbose=verbose)
        if cr.ok:
            r.add("long_session_pass", "84291567" in cr.content, "")
            r.raw["long_session_response"] = cr.content[:200]
            if "84291567" not in cr.content:
                r.warn(
                    "10 轮多轮对话无法回忆第 1 轮的关键数字 — "
                    "可能多轮上下文被截断或 KV cache 不稳定"
                )
            else:
                r.find("10 轮多轮对话成功回忆开头信息")
        else:
            r.warn(f"长会话测试失败: {cr.error_type}")

    # 3) max_tokens 边界: 请求超出 max_tokens
    if test_max_tokens_boundary:
        cr = client.call(
                    model,
                    [{"role": "user", "content": "用 5000 字介绍量子力学"}],
                    stream=False,
                    max_tokens=100_000,  # 故意极大
                    temperature=0.0,
                    verbose=verbose,
                )
        if cr.ok:
            # 应当要么成功但只返回 max_tokens, 要么返回 max_tokens 限制错误
            if cr.completion_tokens < 1000:
                r.find(f"max_tokens 边界生效: 请求 100k, 实际返回 {cr.completion_tokens}")
            else:
                r.warn(f"max_tokens=100000 没被正确限制, 返回 {cr.completion_tokens} tokens")
        else:
            # 报错也算合理
            r.find(f"max_tokens 边界正确报错: {cr.error_type}")

    client.close()
    return r