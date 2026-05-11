"""MiniMax OAuth client — sync and async wrappers for the Coding Plan API.

Provides chat completion (via OpenAI SDK delegation) and vision (VLM)
through MiniMax's Coding Plan endpoints.  Both clients expose the
``client.chat.completions.create(...)`` surface expected by auxiliary_client.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

MINIMAX_VLM_MODEL = "MiniMax-VL-01"
_VLM_ENDPOINT = "/v1/coding_plan/vlm"
_VLM_TIMEOUT = 180  # seconds — vision calls can be slow


# ── Response shim ───────────────────────────────────────────────────────────
#
# MiniMax's VLM endpoint returns raw JSON, not an OpenAI-shaped object.
# These dataclasses provide a minimal but type-safe shim so callers that
# expect ``response.choices[0].message.content`` work transparently.


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    finish_reason: str = "stop"


@dataclass
class _ChatCompletion:
    """Minimal OpenAI-compatible response object for VLM results."""

    choices: List[_Choice] = field(default_factory=list)


def _build_vlm_response(text: str) -> _ChatCompletion:
    """Wrap a VLM text result in an OpenAI-shaped response."""
    return _ChatCompletion(choices=[_Choice(message=_Message(content=text))])


# ── Shared helpers ──────────────────────────────────────────────────────────


def _parse_root_url(base_url: str) -> str:
    """Extract the root API URL, stripping any ``/anthropic`` suffix.

    MiniMax OAuth credentials store the inference URL with an ``/anthropic``
    suffix for Anthropic-wire compatibility.  Chat delegation needs the
    ``/anthropic`` path, but VLM and search endpoints sit at the root.
    """
    base = base_url.rstrip("/")
    idx = base.find("/anthropic")
    return base[:idx] if idx != -1 else base


def _extract_vision_payload(
    messages: List[Dict[str, Any]],
) -> tuple[str, str]:
    """Extract ``(prompt, image_data_url)`` from message history.

    Walks messages in reverse, supporting both OpenAI ``image_url`` blocks
    and Anthropic ``image`` blocks (produced by auxiliary_client's
    ``_convert_openai_images_to_anthropic`` before messages reach us).
    """
    prompt = ""
    image_data = ""

    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if not prompt:
                prompt = content
        elif isinstance(content, list):
            for part in content:
                part_type = part.get("type")
                if part_type == "text" and not prompt:
                    prompt = part.get("text", "")
                elif part_type == "image_url":
                    # OpenAI format: {"type": "image_url", "image_url": {"url": "..."}}
                    image_data = part["image_url"]["url"]
                elif part_type == "image":
                    # Anthropic format: {"type": "image", "source": {"type": "base64", ...}}
                    src = part.get("source", {})
                    if src.get("type") == "base64":
                        image_data = (
                            f"data:{src.get('media_type')};base64,{src.get('data')}"
                        )
                    else:
                        image_data = src.get("url", "")
        if image_data:
            break

    # Sanitize: strip whitespace and line breaks from data URIs
    if image_data:
        image_data = image_data.strip().replace("\n", "").replace("\r", "")

    return prompt or "Describe this image", image_data


def _validate_vlm_response(data: Dict[str, Any]) -> str:
    """Extract and validate the VLM content from the API response.

    Raises :class:`RuntimeError` if the MiniMax ``base_resp`` indicates an
    error or the response contains no content.
    """
    base_resp = data.get("base_resp", {})
    if base_resp.get("status_code") != 0:
        error_msg = base_resp.get("status_msg", "Unknown error")
        raise RuntimeError(f"MiniMax VLM error: {error_msg}")
    content = data.get("content", "")
    if not content:
        raise RuntimeError("MiniMax VLM returned empty content")
    return content


# ── Base client ─────────────────────────────────────────────────────────────


class MiniMaxClientBase:
    """Shared state and interface shim for MiniMax clients.

    The ``self.chat = self`` / ``self.completions = self`` trick lets callers
    use ``client.chat.completions.create(...)`` — the same surface that the
    OpenAI SDK exposes — so ``auxiliary_client`` doesn't need special-casing
    beyond the ``isinstance`` checks in ``_to_async_client``.
    """

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.root_url = _parse_root_url(base_url)
        self.anthropic_url = f"{self.root_url}/anthropic"
        # Shim: allows callers to do client.chat.completions.create(...)
        self.chat = self
        self.completions = self

    @property
    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def close(self) -> None:
        """No-op for compatibility with OpenAI client lifecycle."""


# ── Sync client ─────────────────────────────────────────────────────────────


class MiniMaxOAuthClient(MiniMaxClientBase):
    """Synchronous MiniMax OAuth client."""

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> _ChatCompletion:
        if model == MINIMAX_VLM_MODEL:
            return self._call_vlm(messages)
        return self._delegate_chat(model, messages, **kwargs)

    def _call_vlm(self, messages: List[Dict[str, Any]]) -> _ChatCompletion:
        prompt, image = _extract_vision_payload(messages)
        if not image:
            raise ValueError(
                f"{MINIMAX_VLM_MODEL} requires an image in the message history"
            )
        with httpx.Client(timeout=_VLM_TIMEOUT) as client:
            resp = client.post(
                f"{self.root_url}{_VLM_ENDPOINT}",
                headers=self._auth_headers,
                json={"prompt": prompt, "image_url": image},
            )
            resp.raise_for_status()
            return _build_vlm_response(_validate_vlm_response(resp.json()))

    def _delegate_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ):
        from agent.auxiliary_client import _to_openai_base_url

        with OpenAI(
            api_key=self.api_key,
            base_url=_to_openai_base_url(self.anthropic_url),
        ) as client:
            return client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )


# ── Async client ────────────────────────────────────────────────────────────


class AsyncMiniMaxOAuthClient(MiniMaxClientBase):
    """Asynchronous MiniMax OAuth client."""

    def __init__(self, sync_client: MiniMaxOAuthClient):
        super().__init__(sync_client.api_key, sync_client.root_url)

    async def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> _ChatCompletion:
        if model == MINIMAX_VLM_MODEL:
            return await self._call_vlm(messages)
        return await self._delegate_chat(model, messages, **kwargs)

    async def _call_vlm(
        self, messages: List[Dict[str, Any]]
    ) -> _ChatCompletion:
        prompt, image = _extract_vision_payload(messages)
        if not image:
            raise ValueError(
                f"{MINIMAX_VLM_MODEL} requires an image in the message history"
            )
        async with httpx.AsyncClient(timeout=_VLM_TIMEOUT) as client:
            resp = await client.post(
                f"{self.root_url}{_VLM_ENDPOINT}",
                headers=self._auth_headers,
                json={"prompt": prompt, "image_url": image},
            )
            resp.raise_for_status()
            return _build_vlm_response(_validate_vlm_response(resp.json()))

    async def _delegate_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ):
        from agent.auxiliary_client import _to_openai_base_url

        async with AsyncOpenAI(
            api_key=self.api_key,
            base_url=_to_openai_base_url(self.anthropic_url),
        ) as client:
            return await client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )


# ── Factory ─────────────────────────────────────────────────────────────────


def get_minimax_oauth_client() -> Optional[MiniMaxOAuthClient]:
    """Resolve credentials and return a configured sync client, or ``None``."""
    try:
        from hermes_cli.auth import resolve_minimax_oauth_runtime_credentials

        creds = resolve_minimax_oauth_runtime_credentials()
        if creds and creds.get("api_key"):
            return MiniMaxOAuthClient(creds["api_key"], creds["base_url"])
    except Exception as exc:
        logger.debug("Failed to resolve MiniMax OAuth credentials: %s", exc)
    return None
