from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DONE = object()


def _resolve_api_base(domain_name: str) -> str:
    domain = (domain_name or "feishu").strip().lower()
    if domain == "lark":
        return "https://open.larksuite.com/open-apis"
    return "https://open.feishu.cn/open-apis"


def _merge_streaming_text(previous_text: str | None, next_text: str | None) -> str:
    previous = previous_text or ""
    next_value = next_text or ""
    if not next_value:
        return previous
    if not previous or next_value == previous:
        return next_value
    if next_value.startswith(previous):
        return next_value
    if previous.startswith(next_value):
        return previous
    if next_value in previous:
        return previous
    if previous in next_value:
        return next_value

    max_overlap = min(len(previous), len(next_value))
    for overlap in range(max_overlap, 0, -1):
        if previous[-overlap:] == next_value[:overlap]:
            return previous + next_value[overlap:]
    return previous + next_value


def _truncate_summary(text: str, max_chars: int = 50) -> str:
    clean = (text or "").replace("\n", " ").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def _clean_visible_text(adapter: Any, text: Optional[str]) -> str:
    cleaned = text or ""
    if hasattr(adapter, "extract_media"):
        _media_files, cleaned = adapter.extract_media(cleaned)
    cleaned = cleaned.replace("[[audio_as_voice]]", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _format_elapsed(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds < 10:
        return f"{seconds:.1f}s"
    return f"{int(round(seconds))}s"


def _build_footer_markdown(
    *,
    model_label: Optional[str],
    elapsed_seconds: Optional[float],
    footer_text: Optional[str] = None,
) -> str:
    parts = []
    if footer_text:
        parts.append(str(footer_text).strip())
    elif model_label or elapsed_seconds is not None:
        if model_label:
            parts.append(f"模型：{model_label}")
        if elapsed_seconds is not None:
            parts.append(f"耗时：{_format_elapsed(elapsed_seconds)}")
    if not parts:
        return ""
    footer_text = " · ".join(parts)
    return f"\n\n---\n<font color='grey'>{footer_text}</font>"


class FeishuCardTokenCache:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, float]] = {}

    async def get_token(self, adapter: Any) -> str:
        cache_key = f"{getattr(adapter, '_domain_name', 'feishu')}|{getattr(adapter, '_app_id', '')}"
        cached = self._cache.get(cache_key)
        if cached and cached[1] > time.time() + 60:
            return cached[0]

        if not getattr(adapter, "_app_id", "") or not getattr(adapter, "_app_secret", ""):
            raise RuntimeError("Feishu app_id/app_secret missing")

        if getattr(adapter, "_domain_name", "feishu") == "lark":
            url = "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal"
        else:
            url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"

        session = getattr(adapter, "_rest_session", None)
        close_after = False
        if session is None:
            import aiohttp

            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            close_after = True

        try:
            async with session.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"app_id": adapter._app_id, "app_secret": adapter._app_secret},
            ) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Feishu token request failed with HTTP {response.status}")
                data = await response.json()
        finally:
            if close_after:
                await session.close()

        token = str(data.get("tenant_access_token") or "")
        if int(data.get("code", -1)) != 0 or not token:
            raise RuntimeError(f"Feishu token error: {data.get('msg') or 'unknown'}")

        expires_in = int(data.get("expire") or 7200)
        self._cache[cache_key] = (token, time.time() + expires_in)
        return token


_TOKEN_CACHE = FeishuCardTokenCache()


@dataclass
class _CompletionMeta:
    final_text: Optional[str] = None
    model_label: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    footer_text: Optional[str] = None


class _FeishuCardSession:
    def __init__(self, adapter: Any, chat_id: str, metadata: Optional[dict] = None):
        self.adapter = adapter
        self.chat_id = chat_id
        self.metadata = metadata or {}
        self.card_id: Optional[str] = None
        self.message_id: Optional[str] = None
        self.sequence = 1
        self.current_text = ""
        self.closed = False
        self._last_update_at = 0.0
        self._pending_text: Optional[str] = None
        self._throttle_ms = 100

    async def _request(self, method: str, url: str, *, json_body: dict, audit_name: str) -> dict:
        import aiohttp

        token = await _TOKEN_CACHE.get_token(self.adapter)
        session = getattr(self.adapter, "_rest_session", None)
        close_after = False
        if session is None:
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            close_after = True
        try:
            async with session.request(
                method,
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                json=json_body,
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(f"{audit_name} failed with HTTP {response.status}: {body[:500]}")
                data = await response.json()
        finally:
            if close_after:
                await session.close()

        if int(data.get("code", -1)) != 0:
            raise RuntimeError(f"{audit_name} failed: {data.get('msg') or 'unknown'}")
        return data

    async def start(self) -> None:
        if self.card_id:
            return
        api_base = _resolve_api_base(getattr(self.adapter, "_domain_name", "feishu"))
        initial_card = {
            "schema": "2.0",
            "config": {
                "streaming_mode": True,
                "summary": {"content": "[Generating...]"},
                "streaming_config": {
                    "print_frequency_ms": {"default": 50},
                    "print_step": {"default": 1},
                },
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": "⏳ Thinking...",
                        "element_id": "content",
                    }
                ]
            },
        }

        create_data = await self._request(
            "POST",
            f"{api_base}/cardkit/v1/cards",
            json_body={"type": "card_json", "data": json.dumps(initial_card, ensure_ascii=False)},
            audit_name="feishu.streaming-card.create",
        )
        self.card_id = str(((create_data.get("data") or {}).get("card_id") or "")).strip()
        if not self.card_id:
            raise RuntimeError("Feishu streaming card creation returned no card_id")

        card_ref = json.dumps({"type": "card", "data": {"card_id": self.card_id}}, ensure_ascii=False)
        response = await self.adapter._feishu_send_with_retry(
            chat_id=self.chat_id,
            msg_type="interactive",
            payload=card_ref,
            reply_to=None,
            metadata=self.metadata,
        )
        result = self.adapter._finalize_send_result(response, "feishu streaming card send failed")
        if not result.success or not result.message_id:
            raise RuntimeError(result.error or "Feishu streaming card send failed")

        self.message_id = result.message_id
        self.adapter._store_message_text_cache(self.message_id, "⏳ Thinking...")

    async def update(self, text: str) -> None:
        if self.closed:
            return
        if not self.card_id:
            await self.start()

        merged = _merge_streaming_text(self._pending_text or self.current_text, text)
        if not merged or merged == self.current_text:
            return

        now = time.time() * 1000
        if now - self._last_update_at < self._throttle_ms:
            self._pending_text = merged
            return

        self._pending_text = None
        self._last_update_at = now
        self.current_text = _merge_streaming_text(self.current_text, merged)
        self.adapter._store_message_text_cache(self.message_id, self.current_text)
        self.sequence += 1
        api_base = _resolve_api_base(getattr(self.adapter, "_domain_name", "feishu"))
        await self._request(
            "PUT",
            f"{api_base}/cardkit/v1/cards/{self.card_id}/elements/content/content",
            json_body={
                "content": self.current_text,
                "sequence": self.sequence,
                "uuid": f"s_{self.card_id}_{self.sequence}",
            },
            audit_name="feishu.streaming-card.update",
        )

    async def close(self, *, final_text: Optional[str], model_label: Optional[str], elapsed_seconds: Optional[float], footer_text: Optional[str] = None) -> None:
        if self.closed:
            return
        self.closed = True
        if not self.card_id:
            return

        pending_merged = _merge_streaming_text(self.current_text, self._pending_text)
        pending_visible = _clean_visible_text(self.adapter, pending_merged)
        final_visible = _clean_visible_text(self.adapter, final_text)
        display_text = final_visible or pending_visible
        footer = _build_footer_markdown(
            model_label=model_label,
            elapsed_seconds=elapsed_seconds,
            footer_text=footer_text,
        )
        if footer and footer not in display_text:
            display_text = (display_text or "") + footer

        if display_text and display_text != self.current_text:
            self.sequence += 1
            api_base = _resolve_api_base(getattr(self.adapter, "_domain_name", "feishu"))
            await self._request(
                "PUT",
                f"{api_base}/cardkit/v1/cards/{self.card_id}/elements/content/content",
                json_body={
                    "content": display_text,
                    "sequence": self.sequence,
                    "uuid": f"s_{self.card_id}_{self.sequence}",
                },
                audit_name="feishu.streaming-card.final-update",
            )
            self.current_text = display_text

        self.adapter._store_message_text_cache(self.message_id, self.current_text)
        self.sequence += 1
        api_base = _resolve_api_base(getattr(self.adapter, "_domain_name", "feishu"))
        await self._request(
            "PATCH",
            f"{api_base}/cardkit/v1/cards/{self.card_id}/settings",
            json_body={
                "settings": json.dumps(
                    {
                        "config": {
                            "streaming_mode": False,
                            "summary": {"content": _truncate_summary(self.current_text)},
                        }
                    },
                    ensure_ascii=False,
                ),
                "sequence": self.sequence,
                "uuid": f"c_{self.card_id}_{self.sequence}",
            },
            audit_name="feishu.streaming-card.close",
        )


class FeishuCardStreamConsumer:
    """Feishu-specific streaming consumer using interactive cards instead of message edits."""

    def __init__(self, adapter: Any, chat_id: str, metadata: Optional[dict] = None):
        self.adapter = adapter
        self.chat_id = chat_id
        self.metadata = metadata or {}
        self._queue: queue.Queue = queue.Queue()
        self._accumulated = ""
        self._session = _FeishuCardSession(adapter=adapter, chat_id=chat_id, metadata=metadata)
        self._completion = _CompletionMeta()
        self._already_sent = False
        self._final_response_sent = False

    @property
    def already_sent(self) -> bool:
        return self._already_sent

    @property
    def final_response_sent(self) -> bool:
        return self._final_response_sent

    def on_delta(self, text: Optional[str]) -> None:
        if text:
            self._queue.put(text)

    def on_segment_break(self) -> None:
        return None

    def on_commentary(self, text: str) -> None:
        if text:
            self._queue.put(text)

    def set_completion_meta(
        self,
        *,
        final_text: Optional[str] = None,
        model_label: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
        footer_text: Optional[str] = None,
    ) -> None:
        self._completion = _CompletionMeta(
            final_text=final_text,
            model_label=model_label,
            elapsed_seconds=elapsed_seconds,
            footer_text=footer_text,
        )

    def finish(self) -> None:
        self._queue.put(_DONE)

    async def run(self) -> None:
        try:
            while True:
                got_done = False
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is _DONE:
                        got_done = True
                        break
                    self._accumulated = _merge_streaming_text(self._accumulated, str(item))

                if self._accumulated:
                    await self._session.update(self._accumulated)
                    self._already_sent = True

                if got_done:
                    await self._session.close(
                        final_text=self._completion.final_text or self._accumulated,
                        model_label=self._completion.model_label,
                        elapsed_seconds=self._completion.elapsed_seconds,
                        footer_text=self._completion.footer_text,
                    )
                    delivered = bool(
                        getattr(self._session, "message_id", None)
                        or getattr(self._session, "card_id", None)
                    )
                    self._already_sent = self._already_sent or delivered
                    self._final_response_sent = delivered
                    return

                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Feishu interactive card streaming failed: %s", exc, exc_info=True)
