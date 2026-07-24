"""Probe 2: 上下文窗口探测 (中转站截断检测)

检测目标:
  - 二分查找 API key 实际可用的最大 context length
  - 用"针线包"测试 (needle-in-haystack) 检测是否偷偷截断中间内容
  - 输出 limit tokens vs claimed tokens

策略:
  1. 二分探测错误边界: 每次把 prompt 放大, 直到返回 context_length_exceeded
  2. 在 ~80% limit 处插入一个唯一 token, 让模型回忆, 看是否真的收到了全部内容
"""
from __future__ import annotations

import random
import string
from typing import Optional

from llm_api_probe.probes.client import ProbeClient
from llm_api_probe.probes.models import Provider, ProbeResult


def _rand_token(n: int = 12) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def _make_filler(target_tokens: int, marker: str, marker_pos_ratio: float) -> tuple[str, str]:
    """生成填充文本, 在 marker_pos_ratio 位置插入 marker。

    估算: 中文 ~1.5 chars/token, 英文 ~4 chars/token。粗略按平均 3 chars/token。
    """
    chars = target_tokens * 3
    filler = ("The quick brown fox jumps over the lazy dog. " * (chars // 45 + 1))[:chars]
    # 插入 marker
    pos = int(len(filler) * marker_pos_ratio)
    text = filler[:pos] + f" [NEEDLE:{marker}] " + filler[pos:]
    return text, marker


def _estimate_tokens(s: str) -> int:
    """粗略估算 (不依赖 tiktoken, 兼容性更好)。"""
    # 中文 1.5 chars/token, 英文 4 chars/token → 取均值 2.5
    return max(1, len(s) // 3)


def _probe_limit(client: ProbeClient, model: str, lo: int, hi: int, verbose: bool = False) -> Optional[int]:
    """二分查找最大可用输入 token 数。返回最大成功的 token 数。"""
    last_ok = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        target_chars = mid * 3
        filler = "X" * target_chars
        msgs = [{"role": "user", "content": f"请只回 OK: {filler}"}]
        cr = client.call(model, msgs, stream=False, max_tokens=16, temperature=0.0, verbose=verbose)
        if cr.ok:
            last_ok = mid
            lo = mid + 1
        else:
            if cr.is_context_overflow or "context_length" in (cr.error or "").lower():
                hi = mid - 1
            else:
                # 别的错误 (网络/限流), 中止
                return None
    return last_ok


def run(
    provider: Provider,
    model: str,
    *,
    verbose: bool = False,
    claimed_limit: Optional[int] = None,
    max_probe: int = 200_000,
    test_needle: bool = True,
) -> ProbeResult:
    """claimed_limit: 厂商宣称的最大 context, 用于对比。max_probe: 上限探测上限。"""
    r = ProbeResult(probe="context", provider=provider.name)
    client = ProbeClient(provider)

    # 1) 二分探测实际可用上限
    upper = min(max_probe, claimed_limit) if claimed_limit else max_probe
    actual = _probe_limit(client, model, lo=1024, hi=upper, verbose=verbose)
    if actual is None:
        r.ok = False
        r.error = "无法确定 context 上限 (可能限流或网络不稳)"
        client.close()
        return r

    r.add("actual_max_input_tokens", actual, "tokens")
    if claimed_limit:
        r.add("claimed_limit_tokens", claimed_limit, "tokens")
        diff_pct = round((1 - actual / claimed_limit) * 100, 1) if claimed_limit else 0
        r.add("shrink_pct", diff_pct, "%")
        if actual < claimed_limit * 0.9:
            r.warn(
                f"实际可用 {actual} 比宣称 {claimed_limit} 少 {diff_pct}% — "
                "中转站可能偷换模型或限速版"
            )

    # 2) Needle-in-haystack: 在 ~80% 处插入唯一 token, 让模型回忆
    if test_needle and actual >= 4096:
        target = int(actual * 0.8)
        marker = _rand_token()
        text, _ = _make_filler(target, marker, marker_pos_ratio=0.5)
        needle_msg = [
            {
                "role": "user",
                "content": (
                    f"以下文本中有一个标记 [NEEDLE:{marker}]。"
                    f"请原样输出这个标记 (只要标记本身, 不要其他内容):\n\n{text}"
                ),
            }
        ]
        nr = client.call(model, needle_msg, stream=False, max_tokens=64, temperature=0.0, verbose=verbose)
        if nr.ok:
            hit = marker in nr.content
            r.add("needle_test_pass", hit, "")
            r.raw["needle_marker"] = marker
            r.raw["needle_response"] = nr.content[:200]
            if not hit:
                # 可能截断, 也可能模型太弱 — 提示但不判定
                r.warn(
                    f"needle 测试失败: 文本 ~{target} tokens 中插入的标记 '{marker}' 未被回忆, "
                    f"模型回复: '{nr.content.strip()[:80]}' — 可能被截断或能力不足"
                )
        else:
            r.warn(f"needle 测试本身失败: {nr.error_type}")

    r.find(f"实测可用 context ≈ {actual} tokens" + (f" (宣称 {claimed_limit})" if claimed_limit else ""))
    client.close()
    return r