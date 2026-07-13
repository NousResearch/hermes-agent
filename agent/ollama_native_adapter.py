"""Native Ollama (/api/chat) adapter for the Hermes Agent.

Ollama's OpenAI-compatible /v1 endpoint IGNORES per-request `num_ctx` (verified
through Ollama v0.30.11 / main), so a Hermes agent talking to Ollama via /v1 always
loads models at Ollama's 4096 default. Ollama's NATIVE /api/chat endpoint honors
`options.num_ctx` and returns correct streaming tool-call deltas.

This module provides a drop-in, OpenAI-SDK-shaped client (`OllamaNativeClient`)
that POSTs to /api/chat and translates requests/responses/streams back to the
OpenAI shape the agent already consumes — modeled directly on the sibling
`agent/gemini_native_adapter.py`, so `api_mode` stays "chat_completions" and the
rest of the agent loop is unchanged.

Selected in `agent_runtime_helpers.create_openai_client` for a `custom` provider
whose `base_url` is positively identified as Ollama (gated on the
`HERMES_OLLAMA_NATIVE` env var). Self-contained: only depends on httpx + stdlib,
so it can be unit-tested without the rest of the agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import httpx

logger = logging.getLogger("hermes_ollama_native")

# Feature flag — native mode only engages when explicitly enabled. Off → callers
# fall through to the stock OpenAI client (the default /v1 behavior).
FLAG_ENV = "HERMES_OLLAMA_NATIVE"

# Cache of base_url(root) -> (is_ollama, expires_at_monotonic_or_None). Positive
# results never expire (an Ollama endpoint stays Ollama); negatives expire after a
# short TTL so a transient blip (Ollama restart) recovers, while a persistent
# misconfiguration (non-Ollama endpoint) isn't re-probed on every client build.
_OLLAMA_PROBE_CACHE: Dict[str, Tuple[bool, Optional[float]]] = {}
_NEG_PROBE_TTL = 60.0  # seconds


def _flag_enabled() -> bool:
    return str(os.environ.get(FLAG_ENV, "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def native_root(base_url: str) -> str:
    """Native Ollama root for a base URL — strips a trailing /v1 (and /) so we can
    POST to {root}/api/chat regardless of whether the caller passed the /v1 form."""
    b = (base_url or "").strip().rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return b


def _probe_is_ollama(root: str, *, timeout: float = 2.0) -> bool:
    """Positively identify Ollama via GET {root}/api/version. Cached per root.

    This is what keeps us from mis-routing other `provider="custom"` endpoints
    (vLLM / llama.cpp / LM Studio) — which have no /api/chat — to native mode.

    Positive results are memoized for the process (an Ollama endpoint stays Ollama).
    Negative results are cached only for a short TTL (`_NEG_PROBE_TTL`): a transient
    blip (Ollama restart) recovers on the next probe after the TTL, while a persistent
    misconfiguration (non-Ollama endpoint, or Ollama down) isn't re-probed on every
    client construction — bounding the network round-trips on the hot path.
    """
    entry = _OLLAMA_PROBE_CACHE.get(root)
    if entry is not None:
        result, expires_at = entry
        if expires_at is None or time.monotonic() < expires_at:
            return result
    ok = False
    try:
        resp = httpx.get(f"{root}/api/version", timeout=timeout)
        # Validate the JSON shape ({"version": "<str>"}) rather than a substring
        # match, so a non-Ollama server returning 200 text containing the word
        # "version" can't false-positive into native routing.
        ok = resp.status_code == 200 and isinstance(resp.json().get("version"), str)
    except Exception as exc:  # network/timeout/non-ollama/non-JSON
        logger.debug("Ollama /api/version probe failed for %s: %s", root, exc)
        ok = False
    _OLLAMA_PROBE_CACHE[root] = (
        (True, None) if ok else (False, time.monotonic() + _NEG_PROBE_TTL)
    )
    return ok


def is_native_ollama_base_url(base_url: str) -> bool:
    """True when we should route this endpoint through native /api/chat.

    Gated on (a) the HERMES_OLLAMA_NATIVE flag and (b) a positive Ollama
    identification via /api/version. Works whether the URL carries /v1 or not.
    """
    if not _flag_enabled():
        return False
    if not (base_url or "").strip():
        return False
    return _probe_is_ollama(native_root(base_url))


# Cache of (root, model) -> (supports_thinking, expires_at_monotonic_or_None).
# Definitive answers (any 200 from /api/show) are memoized for the process; probe
# failures are cached with the short `_NEG_PROBE_TTL` (same pattern as the version
# probe above) so a flaky /api/show can't add a round trip to every chat call,
# while a recovered server is re-probed shortly after.
_THINKING_CAP_CACHE: Dict[Tuple[str, str], Tuple[bool, Optional[float]]] = {}


# ── Request construction ────────────────────────────────────────────────────


def _coerce_text(content: Any) -> Tuple[str, List[str]]:
    """Return (text, images[]) from OpenAI message content (str or parts list).

    Images are returned as raw base64 (Ollama's `images` field), best-effort.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        texts: List[str] = []
        images: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif ptype == "image_url":
                url = (
                    ((part.get("image_url") or {}).get("url"))
                    if isinstance(part.get("image_url"), dict)
                    else None
                )
                if isinstance(url, str) and url.startswith("data:") and "," in url:
                    images.append(url.split(",", 1)[1])  # strip data:...;base64,
        return "".join(texts), images
    return str(content), []


def _to_ollama_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-format messages to Ollama /api/chat format.

    Key differences handled:
      - assistant tool_calls: arguments are a JSON *string* in OpenAI, an *object*
        in Ollama → parse.
      - tool results (role="tool"): Ollama uses {role:"tool", content[, tool_name]};
        OpenAI's tool_call_id is dropped (Ollama matches positionally).
      - multimodal content list → text + images[].
    """
    out: List[Dict[str, Any]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        text, images = _coerce_text(msg.get("content"))
        om: Dict[str, Any] = {"role": role, "content": text}
        if images:
            om["images"] = images

        if role == "tool":
            name = msg.get("name") or msg.get("tool_name")
            if name:
                om["tool_name"] = name

        tcs = msg.get("tool_calls")
        if role == "assistant" and isinstance(tcs, list) and tcs:
            conv: List[Dict[str, Any]] = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except (TypeError, ValueError):
                        args = {}
                # Coerce anything that isn't an object (bad parse, or a JSON null /
                # array / scalar) to {} — Ollama expects tool-call args as an object.
                if not isinstance(args, dict):
                    args = {}
                conv.append({
                    "function": {"name": fn.get("name") or "", "arguments": args}
                })
            if conv:
                om["tool_calls"] = conv
        out.append(om)
    return out


def _to_ollama_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    """OpenAI tool defs are already Ollama-compatible ({type:function, function:{...}})."""
    if not tools or not isinstance(tools, list):
        return None
    cleaned = [
        t
        for t in tools
        if isinstance(t, dict) and t.get("type") == "function" and t.get("function")
    ]
    return cleaned or None


def build_ollama_request(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Any = None,
    stream: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop: Any = None,
    seed: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
    think: Any = None,
) -> Dict[str, Any]:
    """Assemble the native /api/chat payload. `options` (incl. num_ctx) come from
    the caller's extra_body.options; sampling params merge in without clobbering it."""
    opts: Dict[str, Any] = dict(options or {})  # num_ctx, etc. (authoritative)
    if temperature is not None and "temperature" not in opts:
        opts["temperature"] = temperature
    if top_p is not None and "top_p" not in opts:
        opts["top_p"] = top_p
    if max_tokens is not None and "num_predict" not in opts:
        opts["num_predict"] = max_tokens
    if seed is not None and "seed" not in opts:
        opts["seed"] = seed
    if stop is not None and "stop" not in opts:
        opts["stop"] = [stop] if isinstance(stop, str) else stop

    payload: Dict[str, Any] = {
        "model": model,
        "messages": _to_ollama_messages(messages),
        "stream": bool(stream),
    }
    tl = _to_ollama_tools(tools)
    if tl:
        payload["tools"] = tl
    if opts:
        payload["options"] = opts
    if think is not None:
        payload["think"] = think
    return payload


# ── Response / stream translation (output shapes mirror gemini_native_adapter) ─


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _translate_tool_calls(raw: Any) -> Optional[List[SimpleNamespace]]:
    if not isinstance(raw, list) or not raw:
        return None
    calls: List[SimpleNamespace] = []
    for index, tc in enumerate(raw):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, (dict, list)):
            try:
                args_str = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args_str = "{}"
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = "{}"
        calls.append(
            SimpleNamespace(
                id=tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                index=index,
                function=SimpleNamespace(
                    name=str(fn.get("name") or ""), arguments=args_str
                ),
            )
        )
    return calls or None


def _usage_from(payload: Dict[str, Any]) -> SimpleNamespace:
    prompt = int(payload.get("prompt_eval_count") or 0)
    completion = int(payload.get("eval_count") or 0)
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )


def _map_finish_reason(done_reason: Any, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    dr = str(done_reason or "stop")
    if dr == "length":
        return "length"
    return "stop"


def translate_ollama_response(payload: Dict[str, Any], model: str) -> SimpleNamespace:
    msg = payload.get("message") or {}
    tool_calls = _translate_tool_calls(msg.get("tool_calls"))
    content = msg.get("content")
    thinking = msg.get("thinking") or None  # Ollama returns reasoning under "thinking"
    message = SimpleNamespace(
        role=msg.get("role") or "assistant",
        content=(
            content if (content not in (None, "")) else (None if tool_calls else "")
        ),
        tool_calls=tool_calls,
        reasoning=thinking,
        reasoning_content=thinking,
        reasoning_details=None,
    )
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason=_map_finish_reason(payload.get("done_reason"), bool(tool_calls)),
    )
    return SimpleNamespace(
        id=_new_id(),
        object="chat.completion",
        created=int(payload.get("created_at_epoch") or time.time()),
        model=payload.get("model") or model,
        choices=[choice],
        usage=_usage_from(payload),
    )


class _OllamaStreamChunk(SimpleNamespace):
    pass


def _make_stream_chunk(
    *,
    model: str,
    content: str = "",
    reasoning: str = "",
    tool_call_delta: Optional[Dict[str, Any]] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[SimpleNamespace] = None,
) -> _OllamaStreamChunk:
    delta_kwargs: Dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
        "tool_calls": None,
        "reasoning": reasoning or None,
        "reasoning_content": reasoning or None,
    }
    if tool_call_delta is not None:
        delta_kwargs["tool_calls"] = [
            SimpleNamespace(
                index=tool_call_delta.get("index", 0),
                id=tool_call_delta.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                function=SimpleNamespace(
                    name=tool_call_delta.get("name") or "",
                    arguments=tool_call_delta.get("arguments") or "",
                ),
            )
        ]
    delta = SimpleNamespace(**delta_kwargs)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return _OllamaStreamChunk(
        id=_new_id(),
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


def translate_stream_line(
    line: Dict[str, Any], model: str, tool_idx: Dict[str, int]
) -> List[_OllamaStreamChunk]:
    """Translate one Ollama NDJSON object into zero+ OpenAI-style chunks."""
    chunks: List[_OllamaStreamChunk] = []
    msg = line.get("message") or {}

    reasoning = msg.get("thinking")
    if isinstance(reasoning, str) and reasoning:
        chunks.append(_make_stream_chunk(model=model, reasoning=reasoning))

    content = msg.get("content")
    if isinstance(content, str) and content:
        chunks.append(_make_stream_chunk(model=model, content=content))

    raw_tcs = msg.get("tool_calls")
    if isinstance(raw_tcs, list):
        for tc in raw_tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            args = fn.get("arguments")
            if isinstance(args, (dict, list)):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except (TypeError, ValueError):
                    args = "{}"
            elif not isinstance(args, str):
                args = "{}"
            # Ollama streams a whole tool call per object; give each a fresh index.
            idx = tool_idx.get("n", 0)
            tool_idx["n"] = idx + 1
            chunks.append(
                _make_stream_chunk(
                    model=model,
                    tool_call_delta={
                        "index": idx,
                        "id": tc.get("id"),
                        "name": name,
                        "arguments": args,
                    },
                )
            )

    if line.get("done") is True:
        has_tc = isinstance(raw_tcs, list) and len(raw_tcs) > 0
        chunks.append(
            _make_stream_chunk(
                model=model,
                finish_reason=_map_finish_reason(
                    line.get("done_reason"), has_tc or tool_idx.get("n", 0) > 0
                ),
                usage=_usage_from(line),
            )
        )
    return chunks


# ── Embeddings translation ──────────────────────────────────────────────────


def _translate_embeddings(payload: Dict[str, Any], model: str) -> SimpleNamespace:
    vectors = payload.get("embeddings") or []
    data = [
        SimpleNamespace(object="embedding", index=i, embedding=v)
        for i, v in enumerate(vectors)
    ]
    tokens = int(payload.get("prompt_eval_count") or 0)
    return SimpleNamespace(
        object="list",
        data=data,
        model=payload.get("model") or model,
        usage=SimpleNamespace(prompt_tokens=tokens, total_tokens=tokens),
    )


# ── Client facade (OpenAI-SDK-shaped, drop-in) ───────────────────────────────


class OllamaAPIError(Exception):
    def __init__(
        self, message: str, *, status_code: Optional[int] = None, response: Any = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


def _http_error(response: httpx.Response) -> OllamaAPIError:
    # Ollama's native API reports failures as {"error": "<str>"} — surface that
    # message directly instead of the raw JSON blob; fall back to raw body text.
    body = "<unreadable>"
    try:
        body = response.text[:500]
        parsed = response.json()
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
            body = parsed["error"][:500]
    except Exception:
        pass  # non-JSON body: keep the raw text captured above
    return OllamaAPIError(
        f"Ollama native API error {response.status_code}: {body}",
        status_code=response.status_code,
        response=response,
    )


class _ChatCompletions:
    def __init__(self, client: "OllamaNativeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ChatNamespace:
    def __init__(self, client: "OllamaNativeClient"):
        self.completions = _ChatCompletions(client)


class _Embeddings:
    def __init__(self, client: "OllamaNativeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_embeddings(**kwargs)


class OllamaNativeClient:
    """Minimal OpenAI-SDK-compatible facade over Ollama's native REST API."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Any = None,
        http_client: Optional[httpx.Client] = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or "ollama"
        self.base_url = native_root(base_url or "http://127.0.0.1:11434")
        self._default_headers = dict(default_headers or {})
        self.chat = _ChatNamespace(self)
        self.embeddings = _Embeddings(self)
        self.is_closed = False
        # Per-request default timeout. Applied both to a self-made client AND, below,
        # as the explicit per-request timeout — so an injected http_client that only
        # carries httpx's short 5s default never governs a long local-model call.
        self._default_timeout = (
            timeout
            if timeout is not None
            else httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0)
        )
        self._http = http_client or httpx.Client(timeout=self._default_timeout)
        # Also expose the httpx client as `_client`: the agent's socket-abort path
        # (force_close_tcp_sockets -> _iter_pool_sockets) resolves the pool via
        # getattr(client, "_client"), the OpenAI-SDK convention. Without this alias
        # interrupt / stale-kill can't shut a long native stream's socket.
        self._client = self._http

    # lifecycle / SDK-compat surface
    def close(self) -> None:
        self.is_closed = True
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _headers(self) -> Dict[str, Union[str, bytes]]:
        h: Dict[str, Union[str, bytes]] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "hermes-agent (ollama-native)",
        }
        # Ollama ignores auth locally, but a reverse proxy may want it.
        if self.api_key and self.api_key != "ollama":
            h["Authorization"] = f"Bearer {self.api_key}"
        # Only forward headers with real str/bytes values. default_headers copied
        # from an OpenAI SDK client (the aux Hook-2 path mirrors the client's
        # timeout + default_headers) can contain the SDK's `Omit()` / `NotGiven`
        # sentinels for UNSET headers (e.g. OpenAI-Organization / OpenAI-Project).
        # httpx rejects those ("Header value must be str or bytes"), which would
        # fail every aux side-task (title/compression/vision). Drop non-string values.
        for k, v in (self._default_headers or {}).items():
            if isinstance(v, (str, bytes)):
                h[k] = v
        return h

    def _timeout_kwargs(self, timeout: Any) -> Dict[str, Any]:
        """httpx request kwargs. Always pass a concrete timeout — the caller's when
        given, else our long default — so a request never silently falls back to
        httpx's 5s default (which applies when an external http_client was injected
        without one). Passing ``timeout=None`` to httpx DISABLES the timeout, which
        we never want."""
        return {"timeout": timeout if timeout is not None else self._default_timeout}

    def _model_supports_thinking(self, model: str) -> bool:
        """Whether the model advertises the "thinking" capability via /api/show.

        Returns True on any uncertainty — a missing capabilities list (older
        Ollama accepts `think` for every model) or a failed probe — so the gate
        only ever suppresses `think` on a positive "not supported" signal.
        Cached per (base_url, model) for the process.
        """
        key = (self.base_url, model)
        entry = _THINKING_CAP_CACHE.get(key)
        if entry is not None:
            cached_supports, expires_at = entry
            if expires_at is None or time.monotonic() < expires_at:
                return cached_supports
        supports = True
        definitive = False
        try:
            # Send the same headers as real requests — behind an authenticated
            # reverse proxy an unauthenticated probe would 401 and the gate
            # would silently never engage.
            resp = self._http.post(
                f"{self.base_url}/api/show",
                json={"model": model},
                headers=self._headers(),
                timeout=2.0,
            )
            if resp.status_code == 200:
                caps = resp.json().get("capabilities")
                if isinstance(caps, list):
                    supports = "thinking" in caps
                # A 200 is definitive (capabilities absent = pre-capability
                # server, which accepts `think` for every model).
                definitive = True
            else:
                logger.debug(
                    "thinking-capability probe got HTTP %s for %s", resp.status_code, model
                )
        except Exception as exc:
            logger.debug("thinking-capability probe failed for %s: %s", model, exc)
        # Definitive answers are memoized; failures forward `think` (fail-open)
        # and are TTL-cached so a persistently failing /api/show doesn't add a
        # round trip to every chat call.
        _THINKING_CAP_CACHE[key] = (
            (supports, None) if definitive else (supports, time.monotonic() + _NEG_PROBE_TTL)
        )
        return supports

    def _create_chat_completion(
        self,
        *,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        tools: Any = None,
        tool_choice: Any = None,  # Ollama has no tool_choice; accepted + ignored
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Any = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        options = None
        think = None
        if isinstance(extra_body, dict):
            options = extra_body.get("options")
            think = extra_body.get("think")

        # Newer Ollama 400s on a truthy `think` (true or a "low"/"high" level)
        # for models without the "thinking" capability; `think: false` is
        # tolerated everywhere. Drop `think` entirely for non-thinking models
        # (mirrors Ollama's own relax_thinking path, and a false/None think is
        # equivalent for a model that cannot think anyway). Unknown capability
        # (older Ollama, probe failure) forwards `think` unchanged.
        if think is not None and not self._model_supports_thinking(model):
            logger.debug("model %s lacks 'thinking' capability; dropping think=%r", model, think)
            think = None

        request = build_ollama_request(
            model=model,
            messages=messages or [],
            tools=tools,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            options=options if isinstance(options, dict) else None,
            think=think,
        )

        if stream:
            return self._stream_completion(
                model=model, request=request, timeout=timeout
            )

        url = f"{self.base_url}/api/chat"
        resp = self._http.post(
            url, json=request, headers=self._headers(), **self._timeout_kwargs(timeout)
        )
        if resp.status_code != 200:
            raise _http_error(resp)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise OllamaAPIError(
                f"Invalid JSON from Ollama /api/chat: {exc}",
                status_code=resp.status_code,
            ) from exc
        return translate_ollama_response(payload, model=model)

    def _stream_completion(
        self, *, model: str, request: Dict[str, Any], timeout: Any = None
    ) -> Iterator[_OllamaStreamChunk]:
        url = f"{self.base_url}/api/chat"

        def _generator() -> Iterator[_OllamaStreamChunk]:
            tool_idx: Dict[str, int] = {"n": 0}
            try:
                with self._http.stream(
                    "POST",
                    url,
                    json=request,
                    headers=self._headers(),
                    **self._timeout_kwargs(timeout),
                ) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        raise _http_error(resp)
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        try:
                            obj = json.loads(raw_line)
                        except (json.JSONDecodeError, TypeError):
                            logger.debug(
                                "Non-JSON Ollama stream line: %s", str(raw_line)[:200]
                            )
                            continue
                        if isinstance(obj, dict):
                            for chunk in translate_stream_line(obj, model, tool_idx):
                                yield chunk
            except httpx.HTTPError as exc:
                raise OllamaAPIError(f"Ollama streaming request failed: {exc}") from exc

        return _generator()

    def _create_embeddings(
        self, *, model: str, input: Any = None, timeout: Any = None, **_: Any
    ) -> Any:
        url = f"{self.base_url}/api/embed"
        resp = self._http.post(
            url,
            json={"model": model, "input": input},
            headers=self._headers(),
            **self._timeout_kwargs(timeout),
        )
        if resp.status_code != 200:
            raise _http_error(resp)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise OllamaAPIError(
                f"Invalid JSON from Ollama /api/embed: {exc}",
                status_code=resp.status_code,
            ) from exc
        return _translate_embeddings(payload, model=model)


class _AsyncChatCompletions:
    def __init__(self, client: "AsyncOllamaNativeClient"):
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        return await self._client._create_chat_completion(**kwargs)


class _AsyncChatNamespace:
    def __init__(self, client: "AsyncOllamaNativeClient"):
        self.completions = _AsyncChatCompletions(client)


class AsyncOllamaNativeClient:
    """Async wrapper (used by auxiliary_client) over a sync OllamaNativeClient."""

    def __init__(self, sync_client: OllamaNativeClient):
        self._sync = sync_client
        self.api_key = sync_client.api_key
        self.base_url = sync_client.base_url
        self.chat = _AsyncChatNamespace(self)
        self._real_client = sync_client  # for the aux cache's leaf-client eviction

    async def _create_chat_completion(self, **kwargs: Any) -> Any:
        stream = bool(kwargs.get("stream"))
        result = await asyncio.to_thread(self._sync.chat.completions.create, **kwargs)
        if not stream:
            return result

        # Consume the sync streaming generator on ONE dedicated thread (preserving the
        # httpx streaming response's thread affinity) and forward chunks to the async
        # consumer via an asyncio.Queue. Advancing the generator with `asyncio.to_thread`
        # per chunk could resume it on different pool threads, which is not safe for a
        # live httpx sync stream.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _producer() -> None:
            try:
                for chunk in result:
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
            except Exception as exc:  # surface to the async consumer
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        threading.Thread(
            target=_producer, name="hermes-ollama-aux-stream", daemon=True
        ).start()

        async def _async_stream() -> Any:
            while True:
                kind, payload = await queue.get()
                if kind == "chunk":
                    yield payload
                elif kind == "error":
                    raise payload
                else:
                    break

        return _async_stream()

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)
