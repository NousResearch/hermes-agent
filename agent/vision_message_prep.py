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

# Request-build relocation of tool-result images (providers that accept
# user-message images but reject list-type tool content; see
# ``ProviderProfile.relocate_tool_result_images``). Constant by design:
# deterministic rebuilds keep prompt-cache prefixes byte-stable.
_TOOL_IMAGE_RELOCATION_MARKER = "Attached media from tool result:"
_TOOL_IMAGE_RELOCATION_PLACEHOLDER = "[image content relocated to the following user message]"


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
            # in-session): short-circuit to a text summary.  Exception: when the provider profile opts
            # into relocation (or the session learned it), keep the image parts — the request-build
            # projection moves them into a user message so the model keeps seeing pixels.
            if (
                not self._provider_supports_vision_tool_messages()
                and not self._should_relocate_tool_result_images()
            ):
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

    @staticmethod
    def _split_tool_content_images(content: list) -> tuple:
        """Split a tool-row content list into (image_parts, residual_content).

        Image parts are preserved verbatim (no re-encode) so rebuilds stay
        byte-stable. The residual keeps unknown part types as a list; when
        only text survives it collapses to a joined string, and when nothing
        survives to ``_TOOL_IMAGE_RELOCATION_PLACEHOLDER``. Callers treat an
        empty image list as "unchanged".
        """
        images: List[Any] = []
        texts: List[str] = []
        rest: List[Any] = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype in _IMAGE_PART_TYPES:
                    images.append(part)
                    continue
                if ptype in _TEXT_PART_TYPES:
                    text = str(part.get("text") or "").strip()
                    if text:
                        texts.append(text)
                    continue
                rest.append(part)
                continue
            if isinstance(part, str):
                if part.strip():
                    texts.append(part.strip())
                continue
            rest.append(part)
        if not images:
            return [], content
        if rest:
            residual: Any = (
                ([{"type": "text", "text": "\n\n".join(texts)}] if texts else []) + rest
            )
        else:
            residual = (
                "\n\n".join(texts) if texts else _TOOL_IMAGE_RELOCATION_PLACEHOLDER
            )
        return images, residual

    @staticmethod
    def _merge_relocated_into_user(user_msg: dict, images: List[Any], marker: str) -> dict:
        """Prepend marker + images to an existing user message (new dict, new
        content list — the input message and its content list stay untouched)."""
        base = user_msg.get("content")
        if isinstance(base, str):
            existing: List[Any] = [{"type": "text", "text": base}] if base else []
        elif isinstance(base, list):
            existing = list(base)
        elif base is None:
            existing = []
        else:
            existing = [{"type": "text", "text": str(base)}]
        return {
            **user_msg,
            "content": [{"type": "text", "text": marker}, *images, *existing],
        }

    def _project_tool_images_for_build(self, api_messages: list) -> list:
        """Deterministic request-build projection: move tool-row images into a
        following user message.

        Produces ``assistant(tool_calls) -> tool(text) -> user([marker,
        images...])``; when a real user message already follows the tool run
        the marker + images are merged into it so role alternation holds.
        Only tool rows whose ``tool_call_id`` matches a preceding assistant
        tool_call are candidates (orphan rows are left alone). Returns the
        input list unchanged when nothing moved.
        """
        marker = _TOOL_IMAGE_RELOCATION_MARKER
        out: List[Any] = []
        known_ids: set = set()
        changed = False
        i = 0
        n = len(api_messages)
        while i < n:
            msg = api_messages[i]
            if not isinstance(msg, dict):
                out.append(msg)
                i += 1
                continue
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        known_ids.add(tc["id"])
                out.append(msg)
                i += 1
                continue
            if role != "tool":
                out.append(msg)
                i += 1
                continue

            run_rows: List[Any] = []
            run_images: List[Any] = []
            j = i
            while j < n:
                row = api_messages[j]
                if not isinstance(row, dict) or row.get("role") != "tool":
                    break
                content = row.get("content")
                if isinstance(content, list) and row.get("tool_call_id") in known_ids:
                    images, residual = self._split_tool_content_images(content)
                    if images:
                        run_images.extend(images)
                        row = {**row, "content": residual}
                run_rows.append(row)
                j += 1

            if run_images:
                changed = True
                next_msg = api_messages[j] if j < n else None
                if isinstance(next_msg, dict) and next_msg.get("role") == "user":
                    out.extend(run_rows)
                    out.append(
                        self._merge_relocated_into_user(next_msg, run_images, marker)
                    )
                    i = j + 1
                    continue
                out.extend(run_rows)
                out.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": marker}, *run_images],
                    }
                )
                i = j
                continue

            out.extend(run_rows)
            i = j
        return out if changed else api_messages

    def _provider_relocates_tool_images(self) -> bool:
        """True when the active provider profile opts into relocation."""
        try:
            from providers import get_provider_profile
            profile = get_provider_profile((getattr(self, "provider", "") or "").strip().lower())
            return profile is not None and bool(
                getattr(profile, "relocate_tool_result_images", False)
            )
        except Exception:
            return False

    def _should_relocate_tool_result_images(self) -> bool:
        """Whether tool-result images must be relocated to a user message at
        request-build time for the active provider.

        True on chat_completions with a vision-capable model when the profile
        opts into relocation (``relocate_tool_result_images``) or the session
        learned the provider rejects tool media. The strip-learned
        ``_no_list_tool_content_models`` memory always wins (convergence).
        """
        try:
            if (getattr(self, "api_mode", "chat_completions") or "chat_completions") != "chat_completions":
                return False
            if not self._model_supports_vision():
                return False
            key = _provider_model_key(self)
            if key in (getattr(self, "_no_list_tool_content_models", None) or ()):
                return False
            if self._provider_relocates_tool_images():
                return True
            relocate_set = getattr(self, "_relocate_tool_images_models", None)
            return bool(relocate_set and key in relocate_set)
        except Exception:
            return False

    def _relocate_tool_result_images_for_api(self, api_messages: list) -> list:
        """Request-build projection entry point (see module constant docs).

        Runs on the per-request ``api_messages`` copy only — persisted history
        is never written. When the session already learned that every image
        shape is rejected (``_no_list_tool_content_models``), historical tool
        rows are stripped to text on the wire instead of relocated, matching
        the insertion-time downgrade for new results.
        """
        if not isinstance(api_messages, list) or not api_messages:
            return api_messages
        try:
            key = _provider_model_key(self)
            if key in (getattr(self, "_no_list_tool_content_models", None) or ()):
                self._try_strip_image_parts_from_tool_messages(api_messages, remember_model=False)
                return api_messages
            if not self._should_relocate_tool_result_images():
                return api_messages
            projected = self._project_tool_images_for_build(api_messages)
            if projected is not api_messages:
                logger.debug(
                    "Relocated tool-result images into a user message for %s/%s "
                    "(chat_completions projection)",
                    key[0], key[1],
                )
            return projected
        except Exception as exc:
            logger.debug("tool-image relocation projection failed: %s", exc)
            return api_messages

    def _try_relocate_image_parts_to_user_message(
        self, api_messages: list, *, remember_model: bool = True
    ) -> bool:
        """Reactive recovery: relocate tool-row images in-place and retry.

        Used by the multimodal-tool-content 400 handler before the existing
        strip fallback. The 400 itself is the evidence of rejection, so this
        helper does not require the proactive predicate — only a
        chat_completions vision model that is not known to accept tool media.
        Records the active (provider, model) in
        ``_relocate_tool_images_models`` so subsequent request builds relocate
        preemptively. Returns True when at least one tool row was relocated.
        """
        if not isinstance(api_messages, list) or not api_messages:
            return False
        try:
            if (getattr(self, "api_mode", "chat_completions") or "chat_completions") != "chat_completions":
                return False
            if not self._model_supports_vision():
                return False
            try:
                from tools.vision_tools import _supports_media_in_tool_results
                provider = (getattr(self, "provider", "") or "").strip()
                model = (getattr(self, "model", "") or "").strip()
                if (
                    _supports_media_in_tool_results(provider, model)
                    and self._provider_supports_vision_tool_messages()
                ):
                    return False
            except Exception:
                pass
            projected = self._project_tool_images_for_build(api_messages)
            if projected is api_messages:
                return False
            api_messages[:] = projected
            if remember_model:
                key = _provider_model_key(self)
                if key[1]:
                    if not hasattr(self, "_relocate_tool_images_models"):
                        self._relocate_tool_images_models = set()
                    self._relocate_tool_images_models.add(key)
            return True
        except Exception:
            return False

    def _try_strip_relocated_images_from_user_messages(
        self, api_messages: list, *, remember_model: bool = True
    ) -> bool:
        """Last-resort recovery: remove relocated images from user messages.

        Fires when a preemptively-relocated request still 400s — the gateway
        rejects user-message images too. Only image parts directly following
        the relocation marker text are removed (user-attached images later in
        the same content list survive; marker-gated so unmarked content is
        never touched). Records ``_no_list_tool_content_models`` so the
        session converges to text-only tool data.
        """
        if not isinstance(api_messages, list):
            return False
        marker = _TOOL_IMAGE_RELOCATION_MARKER
        changed = False
        for msg in api_messages:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            marker_pos = None
            for pos, part in enumerate(content):
                if (
                    isinstance(part, dict)
                    and part.get("type") in _TEXT_PART_TYPES
                    and part.get("text") == marker
                ):
                    marker_pos = pos
                    break
            if marker_pos is None:
                continue
            k = marker_pos + 1
            while k < len(content) and _is_image_part(content[k]):
                k += 1
            if k > marker_pos + 1:
                msg["content"] = content[: marker_pos + 1] + content[k:]
                changed = True
        if changed and remember_model:
            key = _provider_model_key(self)
            if key[1]:
                if not hasattr(self, "_no_list_tool_content_models"):
                    self._no_list_tool_content_models = set()
                self._no_list_tool_content_models.add(key)
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
