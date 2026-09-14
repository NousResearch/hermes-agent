"""Image-part handling for ``AIAgent`` API messages.

Vision capability probes, non-vision text fallbacks (cached ``vision_analyze`` descriptions), tool-result
image stripping, and provider quirks (Anthropic dot preservation, Qwen portal message shaping).
"""
import logging
import asyncio
import base64
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from agent.lazy_forward import forward_static as _forward_static
from agent.tool_dispatch_helpers import _is_multimodal_tool_result, _multimodal_text_summary
from utils import base_url_host_matches, base_url_hostname

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")

_IMAGE_PART_TYPES = {"image_url", "input_image"}
_TEXT_PART_TYPES = {"text", "input_text"}
_DATA_URL_SUFFIXES = {
    "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp", "image/jpeg": ".jpg", "image/jpg": ".jpg"
}


def _is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES


def _salvage_text_parts(content: list, *, any_dict_text: bool) -> List[str]:
    """Stripped, non-empty text from string parts and text-typed dict parts (or any dict's
    ``text`` when ``any_dict_text``), in order."""
    texts: List[str] = []
    for part in content:
        if isinstance(part, str):
            text = part.strip()
        elif isinstance(part, dict) and (any_dict_text or part.get("type") in _TEXT_PART_TYPES):
            text = str(part.get("text", "") or "").strip()
        else:
            continue
        if text:
            texts.append(text)
    return texts


def _provider_model_key(agent: Any) -> tuple[str, str]:
    """``(provider.lower(), model)`` as recorded in ``_no_list_tool_content_models``.
    Module-level so ``MagicMock(spec=AIAgent)`` agents in tests don't swallow it."""
    return (
        (getattr(agent, "provider", "") or "").strip().lower(),
        (getattr(agent, "model", "") or "").strip(),
    )


class VisionMessagePrepMixin:
    """Vision probes + image-part fallbacks for outgoing messages (see module docstring)."""

    @staticmethod
    def _content_has_image_parts(content: Any) -> bool:
        return isinstance(content, list) and any(_is_image_part(part) for part in content)

    # 20 MB base64 ≈ 15 MB decoded — prevents OOM from an oversized data: URL in a shared gateway process.
    _MAX_DATA_URL_BASE64_BYTES = 20 * 1024 * 1024

    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        header, _, data = str(image_url or "").partition(",")
        if len(data) > VisionMessagePrepMixin._MAX_DATA_URL_BASE64_BYTES:
            logger.warning("data-URL payload too large (%d bytes), skipping", len(data))
            return "", None
        mime = header[len("data:"):].split(";", 1)[0].strip() if header.startswith("data:") else ""
        suffix = _DATA_URL_SUFFIXES.get(mime if mime.startswith("image/") else "image/jpeg", ".jpg")
        tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
        try:
            with tmp:
                tmp.write(base64.b64decode(data))
        except Exception:
            # delete=False means a corrupt/unsupported data URL would otherwise
            # leak a zero-byte temp file on every failed materialization.
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
        return tmp.name, Path(tmp.name)

    def _describe_image_for_anthropic_fallback(self, image_url: str, role: str) -> str:
        cache_key = hashlib.sha256(str(image_url or "").encode("utf-8")).hexdigest()
        cached = self._anthropic_image_fallback_cache.get(cache_key)
        if cached:
            return cached

        role_label = {"assistant": "assistant", "tool": "tool result"}.get(role, "user")
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, UI, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        vision_source = str(image_url or "")
        is_data_url = vision_source.startswith("data:")
        cleanup_path: Optional[Path] = None
        if is_data_url:
            vision_source, cleanup_path = self._materialize_data_url_for_vision(vision_source)

        try:
            from tools.vision_tools import vision_analyze_tool

            result_json = asyncio.run(vision_analyze_tool(image_url=vision_source, user_prompt=analysis_prompt))
            result = json.loads(result_json) if isinstance(result_json, str) else {}
            description = (result.get("analysis") or "").strip()
        except Exception as e:
            description = f"Image analysis failed: {e}"
        finally:
            if cleanup_path and cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except OSError:
                    pass

        note = f"[The {role_label} attached an image. Here's what it contains:\n{description or 'Image analysis failed.'}]"
        if vision_source and not is_data_url:
            note += f"\n[If you need a closer look, use vision_analyze with image_url: {vision_source}]"

        self._anthropic_image_fallback_cache[cache_key] = note
        return note

    def _model_supports_vision(self) -> bool:
        """True if the active provider+model reports native vision (config override
        > models.dev; see ``image_routing._supports_vision_override``)."""
        try:
            from hermes_cli.config import load_config
            from agent.image_routing import _lookup_supports_vision
            provider = (getattr(self, "provider", "") or "").strip()
            model = (getattr(self, "model", "") or "").strip()
            return _lookup_supports_vision(provider, model, load_config()) is True
        except Exception:
            return False

    def _provider_supports_vision_tool_messages(self) -> bool:
        """True if the active provider accepts list-type tool content (some, e.g. Xiaomi MiMo, take
        multimodal user messages but 400 on list-type tool content; profile ``supports_vision_tool_messages``)."""
        try:
            from providers import get_provider_profile
            profile = get_provider_profile((getattr(self, "provider", "") or "").strip())
            if profile is not None:
                return getattr(profile, "supports_vision_tool_messages", True)
        except Exception:
            pass
        return True  # default: assume compatible

    def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
        if not self._content_has_image_parts(content):
            return content

        image_notes: List[str] = []
        for part in filter(_is_image_part, content):
            image_data = part.get("image_url", {})
            image_url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
            image_notes.append(
                self._describe_image_for_anthropic_fallback(image_url, role) if image_url
                else "[An image was attached but no image source was available.]"
            )
        # Text parts and unknown dict types both contribute their ``text``.
        prefix = "\n\n".join(note for note in image_notes if note).strip()
        suffix = "\n".join(_salvage_text_parts(content, any_dict_text=True)).strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}"
        return prefix or suffix or "[A multimodal message was converted to text for Anthropic compatibility.]"

    def _get_transport(self, api_mode: str = None):
        """Return the cached transport for the given (or current) api_mode (lazy; None if unregistered)."""
        mode = api_mode or self.api_mode
        cache = getattr(self, "_transport_cache", None)
        if cache is None:
            cache = self._transport_cache = {}
        if cache.get(mode) is None:
            from agent.transports import get_transport
            cache[mode] = get_transport(mode)
        return cache[mode]

    def _prepare_messages_for_non_vision_model(self, api_messages: list) -> list:
        """Replace native image parts with cached vision_analyze text when the active model lacks vision;
        vision-capable models pass through unchanged (the provider adapter handles image parts natively)."""
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content")) for msg in api_messages
        ) or self._model_supports_vision():
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if isinstance(msg, dict):
                msg["content"] = self._preprocess_anthropic_content(
                    msg.get("content"), str(msg.get("role", "user") or "user")
                )
        return transformed

    # Same transform for the Anthropic route (callers/tests patch this name independently).
    _prepare_anthropic_messages_for_api = _prepare_messages_for_non_vision_model

    def _tool_result_content_for_active_model(self, tool_name: str, result: Any) -> Any:
        """Tool message content that is safe for the active model. Text-only providers must not receive
        image parts: a rejected tool result becomes canonical history and can break the next user turn."""
        if not _is_multimodal_tool_result(result):
            return result

        content = result.get("content") or []
        if not self._content_has_image_parts(content):
            return content

        if self._model_supports_vision():
            # Vision on paper, but the provider rejects list-type tool content (or we already learned that
            # in-session): short-circuit to a text summary.
            if not self._provider_supports_vision_tool_messages():
                logger.debug(
                    "Tool %s: provider %s does not accept list-type tool "
                    "content — sending text summary",
                    tool_name, getattr(self, "provider", ""),
                )
                return _multimodal_text_summary(result)
            key = _provider_model_key(self)
            if key in (getattr(self, "_no_list_tool_content_models", None) or ()):
                logger.debug(
                    "Tool %s: model %s/%s known to reject list-type tool "
                    "content this session — sending text summary",
                    tool_name, key[0], key[1],
                )
                return _multimodal_text_summary(result)
            return content

        summary = _multimodal_text_summary(result)
        if tool_name == "computer_use":
            return json.dumps({
                "error": (
                    "computer_use returned screenshot/image content, but the active "
                    "model/provider does not support image input. Switch to a "
                    "vision-capable model for desktop computer use, or use browser "
                    "tools for browser tasks."
                ),
                "text_summary": summary,
            })

        logger.warning(
            "Tool %s returned image content for non-vision model %s/%s; "
            "falling back to text summary",
            tool_name, self.provider, self.model,
        )
        return summary

    _try_shrink_image_parts_in_messages = _forward_static("agent.conversation_compression", "try_shrink_image_parts_in_messages")

    def _try_strip_image_parts_from_tool_messages(
        self, api_messages: list, *, remember_model: bool = True
    ) -> bool:
        """Downgrade list-type tool messages to text in place; True if any were downgraded.

        Recovery for providers that 400 on list-type tool content (e.g. MiMo "text is not set"). By default
        records (provider, model) in ``_no_list_tool_content_models`` so later results downgrade without a
        round-trip; 413 recovery passes ``remember_model=False`` (body too large ≠ provider rejects lists).
        """
        if not isinstance(api_messages, list):
            return False

        if remember_model:
            # Record (provider, model) so we don't relearn this lesson.
            key = _provider_model_key(self)
            if not hasattr(self, "_no_list_tool_content_models"):
                self._no_list_tool_content_models = set()
            if key[1]:  # only record when we actually have a model id
                self._no_list_tool_content_models.add(key)

        changed = False
        for msg in api_messages:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content")
            # List content without image parts is left alone; stripping wouldn't reduce ambiguity.
            if not self._content_has_image_parts(content):
                continue

            # Salvage any text parts so the model still sees some signal.
            msg["content"] = "\n\n".join(_salvage_text_parts(content, any_dict_text=False)) or (
                "[image content removed — provider does not accept "
                "list-type tool message content]"
            )
            changed = True

        return changed

    def _anthropic_preserve_dots(self) -> bool:
        """True for anthropic-compatible endpoints that keep dots in model names (DashScope, MiniMax, Xiaomi
        MiMo, OpenCode Go/Zen, ZAI/Zhipu; Bedrock's dotted inference-profile IDs 400 on the hyphenated form).

        Alibaba/DashScope keeps dots (e.g. qwen3.5-plus). OpenCode Go/Zen keeps dots for non-Claude models
        (e.g. minimax-m2.5-free). ``global.anthropic.claude-opus-4-7``,
        ``us.anthropic.claude-sonnet-4-5-20250929-v1:0``) and rejects the hyphenated form with ``HTTP 400
        The provided model identifier is invalid``. Regression for #11976; mirrors the opencode-go fix for
        #5211
        """
        if (getattr(self, "provider", "") or "").lower() in {
            "alibaba", "minimax", "minimax-cn", "opencode-go", "opencode-zen", "zai", "bedrock", "xiaomi", "vertex",
        }:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        host = base_url_hostname(base)
        return (
            "dashscope" in host
            or base_url_host_matches(base, "aliyuncs.com")
            or "minimax" in host
            or (base_url_host_matches(base, "opencode.ai") and "/zen/" in base)
            or base_url_host_matches(base, "bigmodel.cn")
            or base_url_host_matches(base, "xiaomimimo.com")
            # Vertex AI OpenAI-compat endpoint — Gemini model ids keep dots
            # (e.g. google/gemini-3.5-flash); the hyphenated form is wrong.
            or base_url_host_matches(base, "aiplatform.googleapis.com")
            # AWS Bedrock runtime endpoints — defense-in-depth when
            # ``provider`` is unset but ``base_url`` still names Bedrock.
            or host.startswith("bedrock-runtime.")
        )

    def _is_qwen_portal(self) -> bool:
        """Return True when the base URL targets Qwen Portal."""
        return base_url_host_matches(self._base_url_lower, "portal.qwen.ai")

    def _qwen_prepare_chat_messages(self, api_messages: list) -> list:
        """Deep-copy ``api_messages`` and shape them for Qwen Portal (see the in-place variant)."""
        prepared = copy.deepcopy(api_messages)
        self._qwen_prepare_chat_messages_inplace(prepared)
        return prepared

    def _qwen_prepare_chat_messages_inplace(self, messages: list) -> None:
        """Qwen Portal shaping, in place: every content becomes a list of parts (bare strings → text
        dicts, dicts kept), then ``cache_control`` is injected on the last part of the system message."""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_parts = [
                    {"type": "text", "text": part} if isinstance(part, str) else part
                    for part in content if isinstance(part, (str, dict))
                ]
                if normalized_parts:
                    msg["content"] = normalized_parts

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break
