"""Probe 1: 基础连通性 + 模型列表探测

检测目标:
  - API key 是否有效 (401/403?)
  - /v1/models 端点是否可用
  - 配置中的模型是否真的存在 (检测中转站"模型列表撒谎")
  - 一次最简单的 hello 往返耗时 (网络延迟基线)
"""
from __future__ import annotations

from llm_api_probe.probes.client import ProbeClient
from llm_api_probe.probes.models import Provider, ProbeResult


HELLO_PROMPTS = [
    {"role": "user", "content": "用一句话自我介绍, 不要超过 20 个字。"},
]


def run(provider: Provider, model: str, *, verbose: bool = False) -> ProbeResult:
    r = ProbeResult(probe="connectivity", provider=provider.name)
    client = ProbeClient(provider)

    # 1) 拉模型列表
    lm = client.list_models(verbose=verbose)
    if lm.ok:
        ids = [x.strip() for x in lm.content.split(",") if x.strip()]
        r.add("models_endpoint_ms", round(lm.total_ms, 1), "ms")
        r.add("available_models", len(ids), "个")
        r.raw["models"] = ids
        if model not in ids:
            # 列表存在但请求的模型不在 — 中转站很常见的"偷换"信号
            r.warn(f"配置模型 '{model}' 不在 /v1/models 返回列表中 (列表={ids[:5]}{'...' if len(ids) > 5 else ''})")
    else:
        r.warn(f"/v1/models 不可用: {lm.error_type} — {lm.error[:120]}")
        if lm.is_auth_error:
            r.ok = False
            r.error = "鉴权失败, 检查 api_key"
            client.close()
            return r

    # 2) hello call — 用第一个配置的模型
    hc = client.call(model, HELLO_PROMPTS, stream=False, max_tokens=64, temperature=0.0, verbose=verbose)
    if not hc.ok:
        r.ok = False
        r.error = f"hello 调用失败: {hc.error_type} — {hc.error[:200]}"
        r.add("hello_latency_ms", round(hc.total_ms, 1), "ms")
        r.raw["hello_error"] = hc.error_type
        client.close()
        return r

    r.add("hello_latency_ms", round(hc.total_ms, 1), "ms")
    r.add("hello_output_tokens", hc.completion_tokens, "tokens")
    r.add("hello_prompt_tokens", hc.prompt_tokens, "tokens")
    r.raw["hello_response"] = hc.content[:200]
    r.raw["hello_model_returned"] = hc.model_returned
    r.find(f"连通正常, 模型 '{model}' 返回: {hc.content.strip()[:80]}")

    if hc.model_returned and hc.model_returned != model:
        r.warn(
            f"模型名被改写: 请求 '{model}', 实际返回 '{hc.model_returned}' — "
            "可能走了其他模型或被代理覆盖"
        )

    client.close()
    return r