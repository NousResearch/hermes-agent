"""MessagePrepMixin -- API message sanitization, repair, and preparation.

Encapsulates message sequence repair, tool call sanitization, Anthropic-
specific message preparation, thinking block handling, reasoning content
extraction, and non-vision model preparation.  Previously inline in AIAgent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import copy
import json
import logging
import re

# Re-export utilities for mixin method access
from agent.utils import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


_VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})


def _get_tool_call_id_static(tc) -> str:
    if isinstance(tc, dict):
        return tc.get("call_id", "") or tc.get("id", "") or ""
    return getattr(tc, "call_id", "") or getattr(tc, "id", "") or ""


def _get_tool_call_name_static(tc) -> str:
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return fn.get("name", "") or ""
        return ""
    fn = getattr(tc, "function", None)
    return getattr(fn, "name", "") or ""


class MessagePrepMixin:
    """Mixin providing API message preparation methods for AIAgent.

    All methods use `self` to access agent state and are designed to be
    mixed into the AIAgent class.
    """

    @staticmethod
    def _sanitize_tool_call_arguments(
        messages: list,
        *,
        logger=None,
        session_id: str = None,
    ) -> int:
        """Repair corrupted assistant tool-call argument JSON in-place."""
        log = logger or logging.getLogger(__name__)
        if not isinstance(messages, list):
            return 0

        repaired = 0
        _CORRUPTION_MARKER = (
            "[hermes-agent: tool call arguments were corrupted in this session and "
            "have been dropped to keep the conversation alive. See issue #15236.]"
        )

        def _prepend__CORRUPTION_MARKER(tool_msg: dict) -> None:
            existing = tool_msg.get("content")
            if isinstance(existing, str):
                if not existing:
                    tool_msg["content"] = _CORRUPTION_MARKER
                elif not existing.startswith(_CORRUPTION_MARKER):
                    tool_msg["content"] = f"{_CORRUPTION_MARKER}\n{existing}"
                return
            if existing is None:
                tool_msg["content"] = _CORRUPTION_MARKER
                return
            try:
                existing_text = json.dumps(existing)
            except TypeError:
                existing_text = str(existing)
            tool_msg["content"] = f"{_CORRUPTION_MARKER}\n{existing_text}"

        message_index = 0
        while message_index < len(messages):
            msg = messages[message_index]
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                message_index += 1
                continue

            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                message_index += 1
                continue

            insert_at = message_index + 1
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue

                arguments = function.get("arguments")
                if arguments is None or arguments == "":
                    function["arguments"] = "{}"
                    continue
                if isinstance(arguments, str) and not arguments.strip():
                    function["arguments"] = "{}"
                    continue
                if not isinstance(arguments, str):
                    continue

                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    tool_call_id = tool_call.get("id")
                    function_name = function.get("name", "?")
                    preview = arguments[:80]
                    log.warning(
                        "Corrupted tool_call arguments repaired before request "
                        "(session=%s, message_index=%s, tool_call_id=%s, function=%s, preview=%r)",
                        session_id or "-",
                        message_index,
                        tool_call_id or "-",
                        function_name,
                        preview,
                    )
                    function["arguments"] = "{}"

                    existing_tool_msg = None
                    scan_index = message_index + 1
                    while scan_index < len(messages):
                        candidate = messages[scan_index]
                        if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                            break
                        if candidate.get("tool_call_id") == tool_call_id:
                            existing_tool_msg = candidate
                            break
                        scan_index += 1

                    if existing_tool_msg is None:
                        messages.insert(
                            insert_at,
                            {
                                "role": "tool",
                                "name": function_name if function_name != "?" else "",
                                "tool_call_id": tool_call_id,
                                "content": _CORRUPTION_MARKER,
                            },
                        )
                        insert_at += 1
                    else:
                        _prepend__CORRUPTION_MARKER(existing_tool_msg)

                    repaired += 1

            message_index += 1

        return repaired


    @staticmethod
    def _sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs before every LLM call.

        Runs unconditionally — not gated on whether the context compressor
        is present — so orphans from session loading or manual message
        manipulation are always caught.
        """
        # --- Role allowlist: drop messages with roles the API won't accept ---
        filtered = []
        for msg in messages:
            role = msg.get("role")
            if role not in _VALID_API_ROLES:
                logger.debug(
                    "Pre-call sanitizer: dropping message with invalid role %r",
                    role,
                )
                continue
            filtered.append(msg)
        messages = filtered

        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = _get_tool_call_id_static(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Drop tool results with no matching assistant call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            logger.debug(
                "Pre-call sanitizer: removed %d orphaned tool result(s)",
                len(orphaned_results),
            )

        # 2. Inject stub results for calls whose result was dropped
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = _get_tool_call_id_static(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "name": _get_tool_call_name_static(tc),
                                "content": "[Result unavailable — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            logger.debug(
                "Pre-call sanitizer: added %d stub tool result(s)",
                len(missing_results),
            )
        return messages


    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
        """Strip Codex Responses API fields from tool_calls for strict providers.

        Providers like Mistral, Fireworks, and other strict OpenAI-compatible APIs
        validate the Chat Completions schema and reject unknown fields (call_id,
        response_item_id) with 400 or 422 errors. These fields are preserved in
        the internal message history — this method only modifies the outgoing
        API copy.

        Creates new tool_call dicts rather than mutating in-place, so the
        original messages list retains call_id/response_item_id for Codex
        Responses API compatibility (e.g. if the session falls back to a
        Codex provider later).

        Fields stripped: call_id, response_item_id
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        _STRIP_KEYS = {"call_id", "response_item_id"}
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
            if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg


    def _repair_message_sequence(self, messages: List[Dict]) -> int:
        """Collapse malformed role-alternation left in the live history.

        Providers (OpenAI, OpenRouter, Anthropic) expect strict alternation:
        after the system message, user/tool alternates with assistant, with
        no two consecutive user messages and no tool-result that doesn't
        follow an assistant-with-tool_calls. Violations cause silent empty
        responses on most providers, which triggers the empty-retry loop.

        This runs right before the API call as a defensive belt — by the
        time it fires, the scaffolding strip should already have prevented
        most shapes, but external callers (gateway multi-queue replay,
        session resume, cron, explicit conversation_history passed in by
        host code) can feed in already-broken histories.

        Repairs applied:
          1. Stray ``tool`` messages whose ``tool_call_id`` doesn't match
             any preceding assistant tool_call — dropped.
          2. Consecutive ``user`` messages — merged with newline separator
             so no user input is lost.

        Deliberately does NOT rewind orphan ``assistant(tool_calls)+tool``
        pairs that precede a user message — that pattern IS valid when the
        previous turn completed normally and the user jumped in to redirect
        before the model got a continuation turn (the ongoing dialog
        pattern). The empty-response scaffolding stripper handles the
        genuinely-broken variant via its flag-gated rewind.

        Returns the number of repairs made (for logging/telemetry).
        """
        if not messages:
            return 0

        repairs = 0

        # Pass 1: drop stray tool messages that don't follow a known
        # assistant tool_call_id. Uses a rolling set of known ids refreshed
        # on each assistant message.
        known_tool_ids: set = set()
        filtered: List[Dict] = []
        for msg in messages:
            if not isinstance(msg, dict):
                filtered.append(msg)
                continue
            role = msg.get("role")
            if role == "assistant":
                known_tool_ids = set()
                for tc in (msg.get("tool_calls") or []):
                    tc_id = tc.get("id") if isinstance(tc, dict) else None
                    if tc_id:
                        known_tool_ids.add(tc_id)
                filtered.append(msg)
            elif role == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id and tc_id in known_tool_ids:
                    filtered.append(msg)
                else:
                    repairs += 1
            else:
                if role == "user":
                    # A user turn closes the tool-result run; subsequent
                    # tool messages without a fresh assistant tool_call
                    # are orphans.
                    known_tool_ids = set()
                filtered.append(msg)

        # Pass 2: merge consecutive user messages. Preserves all user input
        # so nothing the user typed is lost.
        merged: List[Dict] = []
        for msg in filtered:
            if (
                merged
                and isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(merged[-1], dict)
                and merged[-1].get("role") == "user"
            ):
                prev = merged[-1]
                prev_content = prev.get("content", "")
                new_content = msg.get("content", "")
                # Only merge plain-text content; leave multimodal (list)
                # content alone — collapsing image/audio blocks risks
                # mangling the attachment structure.
                if isinstance(prev_content, str) and isinstance(new_content, str):
                    prev["content"] = (
                        (prev_content + "\n\n" + new_content)
                        if prev_content and new_content
                        else (prev_content or new_content)
                    )
                    repairs += 1
                    continue
            merged.append(msg)

        if repairs > 0:
            # Rewrite in place so downstream paths (persistence, return
            # value, session DB flush) see the repaired sequence.
            messages[:] = merged

        return repairs


    def _repair_tool_call(self, tool_name: str) -> str | None:
        """Attempt to repair a mismatched tool name before aborting.

        Models sometimes emit variants of a tool name that differ only
        in casing, separators, or class-like suffixes. Normalize
        aggressively before falling back to fuzzy match:

        1. Lowercase direct match.
        2. Lowercase + hyphens/spaces -> underscores.
        3. CamelCase -> snake_case (TodoTool -> todo_tool).
        4. Strip trailing ``_tool`` / ``-tool`` / ``tool`` suffix that
           Claude-style models sometimes tack on (TodoTool_tool ->
           TodoTool -> Todo -> todo). Applied twice so double-tacked
           suffixes like ``TodoTool_tool`` reduce all the way.
        5. Fuzzy match (difflib, cutoff=0.7).

        See #14784 for the original reports (TodoTool_tool, Patch_tool,
        BrowserClick_tool were all returning "Unknown tool" before).

        Returns the repaired name if found in valid_tool_names, else None.
        """
        import re
        from difflib import get_close_matches

        if not tool_name:
            return None

        def _norm(s: str) -> str:
            return s.lower().replace("-", "_").replace(" ", "_")

        def _camel_snake(s: str) -> str:
            return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

        def _strip_tool_suffix(s: str) -> str | None:
            lc = s.lower()
            for suffix in ("_tool", "-tool", "tool"):
                if lc.endswith(suffix):
                    return s[: -len(suffix)].rstrip("_-")
            return None

        # Cheap fast-paths first — these cover the common case.
        lowered = tool_name.lower()
        if lowered in self.valid_tool_names:
            return lowered
        normalized = _norm(tool_name)
        if normalized in self.valid_tool_names:
            return normalized

        # Build the full candidate set for class-like emissions.
        cands: set[str] = {tool_name, lowered, normalized, _camel_snake(tool_name)}
        # Strip trailing tool-suffix up to twice — TodoTool_tool needs it.
        for _ in range(2):
            extra: set[str] = set()
            for c in cands:
                stripped = _strip_tool_suffix(c)
                if stripped:
                    extra.add(stripped)
                    extra.add(_norm(stripped))
                    extra.add(_camel_snake(stripped))
            cands |= extra

        for c in cands:
            if c and c in self.valid_tool_names:
                return c

        # Fuzzy match as last resort.
        matches = get_close_matches(lowered, self.valid_tool_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        return None


    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """Remove duplicate (tool_name, arguments) pairs within a single turn.

        Only the first occurrence of each unique pair is kept.
        Returns the original list if no duplicates were found.
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            key = (tc.function.name, tc.function.arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls


    def _prepare_messages_for_non_vision_model(self, api_messages: list) -> list:
        """Strip native image parts when the active model lacks vision.

        Runs on the chat.completions / codex_responses paths. Vision-capable
        models pass through unchanged (provider and any downstream translator
        handle the image parts natively). Non-vision models get each image
        replaced by a cached vision_analyze text description so the turn
        doesn't fail with "model does not support image input".
        """
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        if self._model_supports_vision():
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            # Reuse the Anthropic text-fallback preprocessor — the behaviour is
            # identical (walk content parts, replace images with cached
            # descriptions, merge back into a single text or structured
            # content). Naming is historical.
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed


    def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
        # Fast exit when no message carries image content at all.
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        # The Anthropic adapter (agent/anthropic_adapter.py:_convert_content_part_to_anthropic)
        # already translates OpenAI-style image_url/input_image parts into
        # native Anthropic ``{"type": "image", "source": ...}`` blocks. When
        # the active model supports vision we let the adapter do its job and
        # skip this legacy text-fallback preprocessor entirely.
        if self._model_supports_vision():
            return api_messages

        # Non-vision Anthropic model (rare today, but keep the fallback for
        # compat): replace each image part with a vision_analyze text note.
        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed


    def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
        if not self._content_has_image_parts(content):
            return content

        text_parts: List[str] = []
        image_notes: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue

            ptype = part.get("type")
            if ptype in {"text", "input_text"}:
                text = str(part.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
                continue

            if ptype in {"image_url", "input_image"}:
                image_data = part.get("image_url", {})
                image_url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
                if image_url:
                    image_notes.append(self._describe_image_for_anthropic_fallback(image_url, role))
                else:
                    image_notes.append("[An image was attached but no image source was available.]")
                continue

            text = str(part.get("text", "") or "").strip()
            if text:
                text_parts.append(text)

        prefix = "\n\n".join(note for note in image_notes if note).strip()
        suffix = "\n".join(text for text in text_parts if text).strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}"
        if prefix:
            return prefix
        if suffix:
            return suffix
        return "[A multimodal message was converted to text for Anthropic compatibility.]"


    def _anthropic_prompt_cache_policy(
        self,
        *,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
        model: Optional[str] = None,
    ) -> tuple[bool, bool]:
        """Decide whether to apply Anthropic prompt caching and which layout to use.

        Returns ``(should_cache, use_native_layout)``:
          * ``should_cache`` — inject ``cache_control`` breakpoints for this
            request (applies to OpenRouter Claude, native Anthropic, and
            third-party gateways that speak the native Anthropic protocol).
          * ``use_native_layout`` — place _CORRUPTION_MARKERs on the *inner* content
            blocks (native Anthropic accepts and requires this layout);
            when False _CORRUPTION_MARKERs go on the message envelope (OpenRouter and
            OpenAI-wire proxies expect the looser layout).

        Third-party providers using the native Anthropic transport
        (``api_mode == 'anthropic_messages'`` + Claude-named model) get
        caching with the native layout so they benefit from the same
        cost reduction as direct Anthropic callers, provided their
        gateway implements the Anthropic cache_control contract
        (MiniMax, Zhipu GLM, LiteLLM's Anthropic proxy mode all do).

        Qwen / Alibaba-family models on OpenCode, OpenCode Go, and direct
        Alibaba (DashScope) also honour Anthropic-style ``cache_control``
        _CORRUPTION_MARKERs on OpenAI-wire chat completions. Upstream pi-mono #3392 /
        pi #3393 documented this for opencode-go Qwen. Without _CORRUPTION_MARKERs
        these providers serve zero cache hits, re-billing the full prompt
        on every turn.
        """
        eff_provider = (provider if provider is not None else self.provider) or ""
        eff_base_url = base_url if base_url is not None else (self.base_url or "")
        eff_api_mode = api_mode if api_mode is not None else (self.api_mode or "")
        eff_model = (model if model is not None else self.model) or ""

        model_lower = eff_model.lower()
        provider_lower = eff_provider.lower()
        is_claude = "claude" in model_lower
        is_openrouter = base_url_host_matches(eff_base_url, "openrouter.ai")
        is_anthropic_wire = eff_api_mode == "anthropic_messages"
        is_native_anthropic = (
            is_anthropic_wire
            and (eff_provider == "anthropic" or base_url_hostname(eff_base_url) == "api.anthropic.com")
        )

        if is_native_anthropic:
            return True, True
        if is_openrouter and is_claude:
            return True, False
        if is_anthropic_wire and is_claude:
            # Third-party Anthropic-compatible gateway.
            return True, True

        # MiniMax on its Anthropic-compatible endpoint serves its own
        # model family (MiniMax-M2.7, M2.5, M2.1, M2) with documented
        # cache_control support (0.1× read pricing, 5-minute TTL).  The
        # blanket is_claude gate above excludes these — opt them in
        # explicitly via provider id or host match so users on
        # provider=minimax / minimax-cn (or custom endpoints pointing at
        # api.minimax.io/anthropic / api.minimaxi.com/anthropic) get the
        # same cost reduction as Claude traffic.
        # Docs: https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache
        if is_anthropic_wire:
            is_minimax_provider = provider_lower in {"minimax", "minimax-cn"}
            is_minimax_host = (
                base_url_host_matches(eff_base_url, "api.minimax.io")
                or base_url_host_matches(eff_base_url, "api.minimaxi.com")
            )
            if is_minimax_provider or is_minimax_host:
                return True, True

        # Qwen/Alibaba on OpenCode (Zen/Go) and native DashScope: OpenAI-wire
        # transport that accepts Anthropic-style cache_control _CORRUPTION_MARKERs and
        # rewards them with real cache hits.  Without this branch
        # qwen3.6-plus on opencode-go reports 0% cached tokens and burns
        # through the subscription on every turn.
        model_is_qwen = "qwen" in model_lower
        provider_is_alibaba_family = provider_lower in {
            "opencode", "opencode-zen", "opencode-go", "alibaba",
        }
        if provider_is_alibaba_family and model_is_qwen:
            # Envelope layout (native_anthropic=False): _CORRUPTION_MARKERs on inner
            # content parts, not top-level tool messages.  Matches
            # pi-mono's "alibaba" cacheControlFormat.
            return True, False

        return False, False


    def _anthropic_preserve_dots(self) -> bool:
        """True when using an anthropic-compatible endpoint that preserves dots in model names.
        Alibaba/DashScope keeps dots (e.g. qwen3.5-plus).
        MiniMax keeps dots (e.g. MiniMax-M2.7).
        Xiaomi MiMo keeps dots (e.g. mimo-v2.5, mimo-v2.5-pro).
        OpenCode Go/Zen keeps dots for non-Claude models (e.g. minimax-m2.5-free).
        ZAI/Zhipu keeps dots (e.g. glm-4.7, glm-5.1).
        AWS Bedrock uses dotted inference-profile IDs
        (e.g. ``global.anthropic.claude-opus-4-7``,
        ``us.anthropic.claude-sonnet-4-5-20250929-v1:0``) and rejects
        the hyphenated form with
        ``HTTP 400 The provided model identifier is invalid``.
        Regression for #11976; mirrors the opencode-go fix for #5211
        (commit f77be22c), which extended this same allowlist."""
        if (getattr(self, "provider", "") or "").lower() in {
            "alibaba", "minimax", "minimax-cn",
            "opencode-go", "opencode-zen",
            "zai", "bedrock",
            "xiaomi",
        }:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        return (
            "dashscope" in base
            or "aliyuncs" in base
            or "minimax" in base
            or "opencode.ai/zen/" in base
            or "bigmodel.cn" in base
            or "xiaomimimo.com" in base
            # AWS Bedrock runtime endpoints — defense-in-depth when
            # ``provider`` is unset but ``base_url`` still names Bedrock.
            or "bedrock-runtime." in base
        )


    @staticmethod
    def _drop_thinking_only_and_merge_users( 
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Drop thinking-only assistant turns; merge any adjacent user messages left behind.

        Runs on the per-call ``api_messages`` copy only. The stored
        conversation history (``self.messages``) is never mutated, so the
        user still sees the thinking block in the CLI/gateway transcript and
        session persistence keeps the full trace. Only the wire copy sent to
        the provider is cleaned.

        Why drop-and-merge rather than inject stub text:
        - Fabricating ``"."`` / ``"(continued)"`` text lies in the history
          and makes future turns see model output the model didn't emit.
        - Dropping the turn preserves honesty; merging adjacent user messages
          preserves the provider's role-alternation invariant.
        - This is the pattern used by Claude Code's ``normalizeMessagesForAPI``
          (filterOrphanedThinkingOnlyMessages + mergeAdjacentUserMessages).
        """
        if not messages:
            return messages

        # Pass 1: drop thinking-only assistant turns.
        kept = [m for m in messages if not MessagePrepMixin._is_thinking_only_assistant(m)]
        dropped = len(messages) - len(kept)
        if dropped == 0:
            return messages

        # Pass 2: merge any newly-adjacent user messages.
        merged: List[Dict[str, Any]] = []
        merges = 0
        for m in kept:
            prev = merged[-1] if merged else None
            if (
                prev is not None
                and prev.get("role") == "user"
                and m.get("role") == "user"
            ):
                prev_content = prev.get("content", "")
                cur_content = m.get("content", "")
                # Work on a copy of ``prev`` so the caller's input dicts are
                # never mutated. ``_sanitize_api_messages`` upstream already
                # hands us per-call copies, but staying pure here means we
                # can be called safely from anywhere (tests, other loops).
                prev_copy = dict(prev)
                # Only string-content merge is meaningful for role-alternation
                # purposes. If either side is a list (multimodal), append as a
                # separate block rather than collapsing.
                if isinstance(prev_content, str) and isinstance(cur_content, str):
                    sep = "\n\n" if prev_content and cur_content else ""
                    prev_copy["content"] = prev_content + sep + cur_content
                elif isinstance(prev_content, list) and isinstance(cur_content, list):
                    prev_copy["content"] = list(prev_content) + list(cur_content)
                elif isinstance(prev_content, list) and isinstance(cur_content, str):
                    if cur_content:
                        prev_copy["content"] = list(prev_content) + [
                            {"type": "text", "text": cur_content}
                        ]
                    else:
                        prev_copy["content"] = list(prev_content)
                elif isinstance(prev_content, str) and isinstance(cur_content, list):
                    new_blocks: List[Dict[str, Any]] = []
                    if prev_content:
                        new_blocks.append({"type": "text", "text": prev_content})
                    new_blocks.extend(cur_content)
                    prev_copy["content"] = new_blocks
                else:
                    # Unknown content shape — fall back to appending separately
                    # (violates alternation, but safer than raising in a hot path).
                    merged.append(m)
                    continue
                merged[-1] = prev_copy
                merges += 1
            else:
                merged.append(m)

        logger.debug(
            "Pre-call sanitizer: dropped %d thinking-only assistant turn(s), "
            "merged %d adjacent user message(s)",
            dropped,
            merges,
        )
        return merged


    def _strip_think_blocks(self, content: str) -> str:
        """Remove reasoning/thinking blocks from content, returning only visible text.

        Handles four cases:
          1. Closed tag pairs (``<think>…</think>``) — the common path when
             the provider emits complete reasoning blocks.
          2. Unterminated open tag at a block boundary (start of text or
             after a newline) — e.g. MiniMax M2.7 / NIM endpoints where the
             closing tag is dropped.  Everything from the open tag to end
             of string is stripped.  The block-boundary check mirrors
             ``gateway/stream_consumer.py``'s filter so models that mention
             ``<think>`` in prose aren't over-stripped.
          3. Stray orphan open/close tags that slip through.
          4. Tag variants: ``<think>``, ``<thinking>``, ``<reasoning>``,
             ``<REASONING_SCRATCHPAD>``, ``<thought>`` (Gemma 4), all
             case-insensitive.

        Additionally strips standalone tool-call XML blocks that some open
        models (notably Gemma variants on OpenRouter) emit inside assistant
        content instead of via the structured ``tool_calls`` field:
          * ``<tool_call>…</tool_call>``
          * ``<tool_calls>…</tool_calls>``
          * ``<tool_result>…</tool_result>``
          * ``<function_call>…</function_call>``
          * ``<function_calls>…</function_calls>``
          * ``<function name="…">…</function>`` (Gemma style)
        Ported from openclaw/openclaw#67318. The ``<function>`` variant is
        boundary-gated (only strips when the tag sits at start-of-line or
        after punctuation and carries a ``name="..."`` attribute) so prose
        mentions like "Use <function> in JavaScript" are preserved.
        """
        if not content:
            return ""
        # 1. Closed tag pairs — case-insensitive for all variants so
        #    mixed-case tags (<THINK>, <Thinking>) don't slip through to
        #    the unterminated-tag pass and take trailing content with them.
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # 1b. Tool-call XML blocks (openclaw/openclaw#67318). Handle the
        #     generic tag names first — they have no attribute gating since
        #     a literal <tool_call> in prose is already vanishingly rare.
        for _tc_name in ("tool_call", "tool_calls", "tool_result",
                          "function_call", "function_calls"):
            content = re.sub(
                rf'<{_tc_name}\b[^>]*>.*?</{_tc_name}>',
                '',
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )
        # 1c. <function name="...">...</function> — Gemma-style standalone
        #     tool call. Only strip when the tag sits at a block boundary
        #     (start of text, after a newline, or after sentence-ending
        #     punctuation) AND carries a name="..." attribute. This keeps
        #     prose mentions like "Use <function> to declare" safe.
        content = re.sub(
            r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
            r'<function\b[^>]*\bname\s*=[^>]*>'
            r'(?:(?:(?!</function>).)*)</function>',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # 2. Unterminated reasoning block — open tag at a block boundary
        #    (start of text, or after a newline) with no matching close.
        #    Strip from the tag to end of string.  Fixes #8878 / #9568
        #    (MiniMax M2.7 leaking raw reasoning into assistant content).
        content = re.sub(
            r'(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*$',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # 3. Stray orphan open/close tags that slipped through.
        content = re.sub(
            r'</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*',
            '',
            content,
            flags=re.IGNORECASE,
        )
        # 3b. Stray tool-call closers. (We do NOT strip bare <function> or
        #     unterminated <function name="..."> because a truncated tail
        #     during streaming may still be valuable to the user; matches
        #     OpenClaw's intentional asymmetry.)
        content = re.sub(
            r'</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*',
            '',
            content,
            flags=re.IGNORECASE,
        )
        return content


    def _has_content_after_think_block(self, content: str) -> bool:
        """
        Check if content has actual text after any reasoning/thinking blocks.

        This detects cases where the model only outputs reasoning but no actual
        response, which indicates an incomplete generation that should be retried.
        Must stay in sync with _strip_think_blocks() tag variants.

        Args:
            content: The assistant message content to check

        Returns:
            True if there's meaningful content after think blocks, False otherwise
        """
        if not content:
            return False

        # Remove all reasoning tag variants (must match _strip_think_blocks)
        cleaned = self._strip_think_blocks(content)

        # Check if there's any non-whitespace content remaining
        return bool(cleaned.strip())


    def _copy_reasoning_content_for_api(self, source_msg: dict, api_msg: dict) -> None:
        """Copy provider-facing reasoning fields onto an API replay message."""
        if source_msg.get("role") != "assistant":
            return

        # 1. Explicit reasoning_content already set — preserve it verbatim
        # (includes DeepSeek/Kimi's own space-placeholder written at creation
        # time, and any valid reasoning content from the same provider).
        #
        # Exception: sessions persisted BEFORE #17341 have empty-string
        # placeholders pinned at creation time. DeepSeek V4 Pro rejects
        # those with HTTP 400. When the active provider enforces the
        # thinking-mode echo, upgrade "" → " " on replay so stale history
        # doesn't 400 the user on the next turn.
        existing = source_msg.get("reasoning_content")
        if isinstance(existing, str):
            if existing == "" and self._needs_thinking_reasoning_pad():
                api_msg["reasoning_content"] = " "
            else:
                api_msg["reasoning_content"] = existing
            return

        needs_thinking_pad = self._needs_thinking_reasoning_pad()

        # 2. Cross-provider poisoned history (#15748): on DeepSeek/Kimi,
        # if the source turn has tool_calls AND a 'reasoning' field but no
        # 'reasoning_content' key, the 'reasoning' text was written by a
        # prior provider (e.g. MiniMax) — DeepSeek's own _build_assistant_message
        # pins reasoning_content at creation time for tool-call turns, so the
        # shape (reasoning set, reasoning_content absent, tool_calls present)
        # is unreachable from same-provider DeepSeek history after this fix.
        # Inject a single space to satisfy the API without leaking another
        # provider's chain of thought to DeepSeek/Kimi. Space (not "")
        # because DeepSeek V4 Pro rejects empty-string reasoning_content
        # in thinking mode (refs #17341).
        normalized_reasoning = source_msg.get("reasoning")
        if (
            needs_thinking_pad
            and source_msg.get("tool_calls")
            and isinstance(normalized_reasoning, str)
            and normalized_reasoning
        ):
            api_msg["reasoning_content"] = " "
            return

        # 3. Healthy session: promote 'reasoning' field to 'reasoning_content'
        # for providers that use the internal 'reasoning' key.
        # This must happen before the unconditional empty-string fallback so
        # genuine reasoning content is not overwritten (#15812 regression in
        # PR #15478).
        if isinstance(normalized_reasoning, str) and normalized_reasoning:
            api_msg["reasoning_content"] = normalized_reasoning
            return

        # 4. DeepSeek / Kimi thinking mode: all assistant messages need
        # reasoning_content. Inject a single space to satisfy the provider's
        # requirement when no explicit reasoning content is present. Covers
        # both tool-call turns (already-poisoned history with no reasoning
        # at all) and plain text turns. Space (not "") because DeepSeek V4
        # Pro tightened validation and rejects empty string with HTTP 400
        # ("The reasoning content in the thinking mode must be passed back
        # to the API"). Refs #17341.
        if needs_thinking_pad:
            api_msg["reasoning_content"] = " "
            return

        # 5. reasoning_content was present but not a string (e.g. None after
        # context compaction).  Don't pass null to the API.
        api_msg.pop("reasoning_content", None)


    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """
        Extract reasoning/thinking content from an assistant message.
        
        OpenRouter and various providers can return reasoning in multiple formats:
        1. message.reasoning - Direct reasoning field (DeepSeek, Qwen, etc.)
        2. message.reasoning_content - Alternative field (Moonshot AI, Novita, etc.)
        3. message.reasoning_details - Array of {type, summary, ...} objects (OpenRouter unified)
        
        Args:
            assistant_message: The assistant message object from the API response
            
        Returns:
            Combined reasoning text, or None if no reasoning found
        """
        reasoning_parts = []
        
        # Check direct reasoning field
        if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
            reasoning_parts.append(assistant_message.reasoning)
        
        # Check reasoning_content field (alternative name used by some providers)
        if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
            # Don't duplicate if same as reasoning
            if assistant_message.reasoning_content not in reasoning_parts:
                reasoning_parts.append(assistant_message.reasoning_content)
        
        # Check reasoning_details array (OpenRouter unified format)
        # Format: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            for detail in assistant_message.reasoning_details:
                if isinstance(detail, dict):
                    # Extract summary from reasoning detail object
                    summary = (
                        detail.get('summary')
                        or detail.get('thinking')
                        or detail.get('content')
                        or detail.get('text')
                    )
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)

        # Some providers embed reasoning directly inside assistant content
        # instead of returning structured reasoning fields.  Only fall back
        # to inline extraction when no structured reasoning was found.
        content = getattr(assistant_message, "content", None)
        if not reasoning_parts and isinstance(content, list):
            # DeepSeek V4 Pro (and compatible providers) return content as a
            # list of typed blocks, e.g.:
            #   [{"type": "thinking", "thinking": "..."}, {"type": "output", ...}]
            # Without this branch the thinking text is silently dropped and the
            # next turn fails with HTTP 400 ("thinking must be passed back").
            # Refs #21944.
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_text = block.get("thinking") or block.get("text") or ""
                    thinking_text = thinking_text.strip()
                    if thinking_text and thinking_text not in reasoning_parts:
                        reasoning_parts.append(thinking_text)
        if not reasoning_parts and isinstance(content, str) and content:
            inline_patterns = (
                r"<think>(.*?)</think>",
                r"<thinking>(.*?)</thinking>",
                r"<thought>(.*?)</thought>",
                r"<reasoning>(.*?)</reasoning>",
                r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
            )
            for pattern in inline_patterns:
                flags = re.DOTALL | re.IGNORECASE
                for block in re.findall(pattern, content, flags=flags):
                    cleaned = block.strip()
                    if cleaned and cleaned not in reasoning_parts:
                        reasoning_parts.append(cleaned)
        
        # Combine all reasoning parts
        if reasoning_parts:
            return "\n\n".join(reasoning_parts)
        
        return None


    @staticmethod
    def _is_thinking_only_assistant(msg: Dict[str, Any]) -> bool:
        """Return True if ``msg`` is an assistant turn whose only payload is reasoning.

        "Thinking-only" means the model emitted reasoning (``reasoning`` or
        ``reasoning_content``) but no visible text and no tool_calls. When sent
        back to providers that convert reasoning into thinking blocks (native
        Anthropic, OpenRouter Anthropic, third-party Anthropic-compatible
        gateways), the resulting message has only thinking blocks — which
        Anthropic rejects with HTTP 400 "The final block in an assistant
        message cannot be `thinking`."

        Symmetric with Claude Code's ``filterOrphanedThinkingOnlyMessages``
        (src/utils/messages.ts). We drop the whole turn from the API copy
        rather than fabricating stub text — the message log (UI transcript)
        keeps the reasoning block; only the wire copy is cleaned.
        """
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            return False
        if msg.get("tool_calls"):
            return False
        # Does it have any actual output?
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                return False
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    if block:  # non-empty non-dict string etc.
                        return False
                    continue
                btype = block.get("type")
                if btype in ("thinking", "redacted_thinking"):
                    continue
                if btype == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text.strip():
                        return False
                    continue
                # tool_use, image, document, etc. — real payload
                return False
        elif content is not None and content != "":
            return False
        # Content is empty-ish. Is there reasoning to make it thinking-only?
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return True
        # reasoning_details list form
        rd = msg.get("reasoning_details")
        if isinstance(rd, list) and rd:
            return True
        return False


    def _drop_trailing_empty_response_scaffolding(self, messages: List[Dict]) -> None:
        """Remove private empty-response retry/failure scaffolding from transcript tails.

        Also rewinds past any trailing tool-result / assistant(tool_calls) pair
        that the failed iteration left hanging. Without this, the tail ends at
        a raw ``tool`` message and the next user turn lands as
        ``...tool, user, user`` — a protocol-invalid sequence that most
        providers silently reject (returns empty content), causing the
        empty-retry loop to fire forever. See #<TBD>.
        """
        # Pass 1: strip the flagged scaffolding messages themselves.
        dropped_scaffolding = False
        while (
            messages
            and isinstance(messages[-1], dict)
            and (
                messages[-1].get("_empty_recovery_synthetic")
                or messages[-1].get("_empty_terminal_sentinel")
            )
        ):
            messages.pop()
            dropped_scaffolding = True

        # Pass 2: if we stripped scaffolding, rewind through any trailing
        # tool-result messages plus the assistant(tool_calls) message that
        # produced them. This preserves role alternation so the next user
        # message follows a user or assistant message, not an orphan tool
        # result. Only runs when scaffolding was actually present — normal
        # conversation tails (real tool loops mid-progress) are untouched.
        if not dropped_scaffolding:
            return

        # Drop any trailing tool-result messages
        while (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "tool"
        ):
            messages.pop()

        # Drop the assistant message that issued the tool calls, if the tail
        # now ends in an assistant-with-tool_calls (the pair that owned the
        # just-popped tool results). Without this, the tail is
        # ``assistant(tool_calls=...)`` with no tool answers, which some
        # providers also reject.
        if (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "assistant"
            and messages[-1].get("tool_calls")
        ):
            messages.pop()
