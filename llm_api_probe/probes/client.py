"""OpenAI 兼容客户端封装 (使用官方 openai SDK)。

所有 probe 模块都通过这个客户端发起请求, 统一超时/重试/统计。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

from .models import Provider


@dataclass
class CallResult:
    """一次 API 调用的完整结果 (含耗时/首 token/用量)。"""
    ok: bool
    error: Optional[str] = None
    error_type: Optional[str] = None     # auth / rate_limit / context / timeout / conn / unknown
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0.0                  # streaming 才有意义, 非流式 ≈ 总耗时
    total_ms: float = 0.0
    output_tokens_per_s: float = 0.0
    model_returned: str = ""              # 实际返回的模型名 (检测掉包)
    raw: dict[str, Any] = None            # type: ignore

    @property
    def is_rate_limit(self) -> bool:
        return self.error_type == "rate_limit"

    @property
    def is_context_overflow(self) -> bool:
        return self.error_type == "context"

    @property
    def is_auth_error(self) -> bool:
        return self.error_type == "auth"


def _classify_error(e: Exception) -> str:
    """把异常归类。"""
    if isinstance(e, RateLimitError):
        return "rate_limit"
    if isinstance(e, APITimeoutError):
        return "timeout"
    if isinstance(e, APIConnectionError):
        return "conn"
    if isinstance(e, APIError):
        body = (str(e) or "").lower()
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        if code in (401, 403) or "auth" in body or "api key" in body:
            return "auth"
        if code == 400 and ("context" in body or "token" in body or "length" in body):
            return "context"
        if code == 404:
            return "not_found"
        if code in (429,):
            return "rate_limit"
        return f"api_error_{code}"
    return "unknown"


class ProbeClient:
    """单个 provider 的客户端封装。"""

    def __init__(self, provider: Provider):
        self.provider = provider
        self._client = OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            default_headers=provider.headers or None,
            timeout=provider.timeout,
            max_retries=0,         # probe 不需要 SDK 自动重试, 我们自己控制
        )

    def call(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.0,
        extra: Optional[dict[str, Any]] = None,
        record_ttft: bool = True,
        verbose: bool = False,
    ) -> CallResult:
        """发起一次 chat.completions 调用, 返回统计结果。verbose=True 时实时输出进度。"""
        t0 = time.perf_counter()
        if verbose:
            preview = (messages[-1].get("content") or "")[:50].replace("\n", " ")
            print(f"    → request: model={model} stream={stream} msg='{preview}...'", flush=True)
        first_token_at: Optional[float] = None
        content_parts: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        model_returned = ""
        extra = extra or {}

        try:
            if stream:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra,
                )
                for chunk in resp:
                    if first_token_at is None and record_ttft:
                        first_token_at = time.perf_counter()
                    try:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content_parts.append(delta.content)
                    except (IndexError, AttributeError):
                        pass
                    # usage chunk
                    if getattr(chunk, "usage", None):
                        usage["prompt_tokens"] = chunk.usage.prompt_tokens or 0
                        usage["completion_tokens"] = chunk.usage.completion_tokens or 0
                        usage["total_tokens"] = chunk.usage.total_tokens or 0
                    if getattr(chunk, "model", None):
                        model_returned = chunk.model
            else:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra,
                )
                first_token_at = time.perf_counter()
                if resp.choices:
                    content_parts.append(resp.choices[0].message.content or "")
                if resp.usage:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens or 0,
                        "completion_tokens": resp.usage.completion_tokens or 0,
                        "total_tokens": resp.usage.total_tokens or 0,
                    }
                model_returned = getattr(resp, "model", "") or model

            total_ms = (time.perf_counter() - t0) * 1000.0
            ttft_ms = ((first_token_at - t0) * 1000.0) if first_token_at else total_ms
            content = "".join(content_parts)
            out_tokens = usage["completion_tokens"]
            gen_ms = total_ms - ttft_ms
            out_tps = (out_tokens / (gen_ms / 1000.0)) if gen_ms > 0 and out_tokens > 0 else 0.0

            return CallResult(
                ok=True,
                content=content,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=out_tokens,
                total_tokens=usage["total_tokens"],
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                output_tokens_per_s=out_tps,
                model_returned=model_returned or model,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            return CallResult(
                ok=False,
                error=str(e)[:500],
                error_type=_classify_error(e),
                total_ms=total_ms,
                model_returned=model,
            )

    def list_models(self, verbose: bool = False) -> CallResult:
        """GET /v1/models — 探测可用模型列表。"""
        t0 = time.perf_counter()
        if verbose:
            print(f"    → GET {self.provider.base_url}/models", flush=True)
        try:
            resp = self._client.models.list()
            ids = [m.id for m in resp.data] if getattr(resp, "data", None) else []
            return CallResult(
                ok=True,
                content=",".join(ids),
                total_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as e:
            return CallResult(
                ok=False,
                error=str(e)[:500],
                error_type=_classify_error(e),
                total_ms=(time.perf_counter() - t0) * 1000.0,
            )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass