"""Probe 3: 速度基准 (TTFT / 输出 tokens/s / 并发吞吐)

检测目标:
  - TTFT (time-to-first-token): 单次流式首 token 延迟
  - 输出吞吐: tokens / 秒
  - 并发: 在 N 个并发下能否稳定 + 总吞吐量
  - 输出上限: max_tokens 限制是否生效

策略:
  - 流式单请求: 测 TTFT + 单流输出速度
  - 多轮 (3 次): 取中位数, 排除冷启动
  - 并发 (4 路): 测总吞吐
"""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_api_probe.probes.client import ProbeClient
from llm_api_probe.probes.models import Provider, ProbeResult


SPEED_PROMPT = (
    "用 800 字左右, 详细解释 TCP 三次握手的过程, 包括每一步的状态变化和原因。"
    "请尽量技术化但保持可读。"
)


def _bench_once(client: ProbeClient, model: str, verbose: bool = False) -> dict:
    cr = client.call(
        model,
        [{"role": "user", "content": SPEED_PROMPT}],
        stream=True,
        max_tokens=1024,
        temperature=0.0,
        verbose=verbose,
    )
    return {
        "ok": cr.ok,
        "ttft_ms": cr.ttft_ms,
        "total_ms": cr.total_ms,
        "out_tps": cr.output_tokens_per_s,
        "out_tokens": cr.completion_tokens,
        "error_type": cr.error_type if not cr.ok else None,
        "error": cr.error if not cr.ok else None,
    }


def run(
    provider: Provider,
    model: str,
    *,
    verbose: bool = False,
    rounds: int = 3,
    concurrency: int = 4,
) -> ProbeResult:
    r = ProbeResult(probe="speed", provider=provider.name)
    client = ProbeClient(provider)

    # 1) 顺序 rounds 轮, 测 TTFT / 单流吞吐
    seq_results: list[dict] = []
    for i in range(rounds):
        rec = _bench_once(client, model, verbose=verbose)
        seq_results.append(rec)
        if not rec["ok"]:
            r.warn(f"第 {i + 1} 轮失败: {rec['error_type']} — {rec['error'][:100]}")

    oks = [x for x in seq_results if x["ok"]]
    if not oks:
        r.ok = False
        r.error = "所有顺序请求都失败"
        client.close()
        return r

    ttfts = [x["ttft_ms"] for x in oks]
    tps = [x["out_tps"] for x in oks]
    totals = [x["total_ms"] for x in oks]
    out_tokens = [x["out_tokens"] for x in oks]

    r.add("ttft_median_ms", round(statistics.median(ttfts), 1), "ms")
    r.add("ttft_min_ms", round(min(ttfts), 1), "ms")
    r.add("ttft_max_ms", round(max(ttfts), 1), "ms")
    r.add("output_tps_median", round(statistics.median(tps), 2), " tokens/s")
    r.add("output_tps_max", round(max(tps), 2), " tokens/s")
    r.add("e2e_total_ms_median", round(statistics.median(totals), 1), "ms")
    r.add("output_tokens_median", int(statistics.median(out_tokens)), "tokens")
    r.add("seq_success_rate", f"{len(oks)}/{rounds}", "")
    r.raw["seq_results"] = seq_results

    # 2) 并发测试
    if concurrency > 1:
        conc_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_bench_once, client, model, verbose) for _ in range(concurrency)]
            conc_results = [f.result() for f in as_completed(futures)]
        conc_elapsed = (time.perf_counter() - conc_start) * 1000.0
        conc_oks = [x for x in conc_results if x["ok"]]
        conc_total_out = sum(x["out_tokens"] for x in conc_oks)
        conc_tps_total = conc_total_out / (conc_elapsed / 1000.0) if conc_elapsed > 0 else 0

        r.add("concurrency", concurrency, " 路")
        r.add("concurrency_total_ms", round(conc_elapsed, 1), "ms")
        r.add("concurrency_success_rate", f"{len(conc_oks)}/{concurrency}", "")
        r.add("concurrency_total_out_tps", round(conc_tps_total, 2), " tokens/s")
        r.raw["concurrency_results"] = conc_results

        if len(conc_oks) < concurrency:
            r.warn(f"并发 {concurrency} 路只成功 {len(conc_oks)} 路")

    r.find(
        f"顺序 TTFT 中位 {statistics.median(ttfts):.0f}ms, "
        f"输出速度中位 {statistics.median(tps):.1f} tok/s"
    )
    client.close()
    return r