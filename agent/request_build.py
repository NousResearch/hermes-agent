"""RequestBuildMixin -- API request building and lifecycle management.

Encapsulates system prompt building, API kwargs construction, assistant
message building, tool formatting, debug dumping, background review
spawning/summarization, client cleanup, error summarization, message
context helpers, and base_url property. Previously inline in AIAgent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging
import threading

# Re-export utilities for mixin method access
from agent.utils import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


class RequestBuildMixin:
    """Mixin providing request building and lifecycle methods for AIAgent.

    All methods use `self` to access agent state and are designed to be
    mixed into the AIAgent class.
    """

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """Build the keyword arguments dict for the active API mode."""
        if self.api_mode == "anthropic_messages":
            _transport = self._get_transport()
            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            ctx_len = getattr(self, "context_compressor", None)
            ctx_len = ctx_len.context_length if ctx_len else None
            ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
            if ephemeral_out is not None:
                self._ephemeral_max_output_tokens = None  # consume immediately
            return _transport.build_kwargs(
                model=self.model,
                messages=anthropic_messages,
                tools=self.tools,
                max_tokens=ephemeral_out if ephemeral_out is not None else self.max_tokens,
                reasoning_config=self.reasoning_config,
                is_oauth=self._is_anthropic_oauth,
                preserve_dots=self._anthropic_preserve_dots(),
                context_length=ctx_len,
                base_url=getattr(self, "_anthropic_base_url", None),
                fast_mode=(self.request_overrides or {}).get("speed") == "fast",
                drop_context_1m_beta=bool(getattr(self, "_oauth_1m_beta_disabled", False)),
            )

        # AWS Bedrock native Converse API — bypasses the OpenAI client entirely.
        # The adapter handles message/tool conversion and boto3 calls directly.
        if self.api_mode == "bedrock_converse":
            _bt = self._get_transport()
            region = getattr(self, "_bedrock_region", None) or "us-east-1"
            guardrail = getattr(self, "_bedrock_guardrail_config", None)
            return _bt.build_kwargs(
                model=self.model,
                messages=api_messages,
                tools=self.tools,
                max_tokens=self.max_tokens or 4096,
                region=region,
                guardrail_config=guardrail,
            )

        if self.api_mode == "codex_responses":
            _ct = self._get_transport()
            is_github_responses = (
                base_url_host_matches(self.base_url, "models.github.ai")
                or base_url_host_matches(self.base_url, "api.githubcopilot.com")
            )
            is_codex_backend = (
                self.provider == "openai-codex"
                or (
                    self._base_url_hostname == "chatgpt.com"
                    and "/backend-api/codex" in self._base_url_lower
                )
            )
            is_xai_responses = self.provider == "xai" or self._base_url_hostname == "api.x.ai"
            _msgs_for_codex = self._prepare_messages_for_non_vision_model(api_messages)
            return _ct.build_kwargs(
                model=self.model,
                messages=_msgs_for_codex,
                tools=self.tools,
                reasoning_config=self.reasoning_config,
                session_id=getattr(self, "session_id", None),
                max_tokens=self.max_tokens,
                request_overrides=self.request_overrides,
                is_github_responses=is_github_responses,
                is_codex_backend=is_codex_backend,
                is_xai_responses=is_xai_responses,
                github_reasoning_extra=self._github_models_reasoning_extra_body() if is_github_responses else None,
            )

        # ── chat_completions (default) ─────────────────────────────────────
        _ct = self._get_transport()

        # Provider detection flags
        _is_qwen = self._is_qwen_portal()
        _is_or = self._is_openrouter_url()
        _is_gh = (
            base_url_host_matches(self._base_url_lower, "models.github.ai")
            or base_url_host_matches(self._base_url_lower, "api.githubcopilot.com")
        )
        _is_nous = "nousresearch" in self._base_url_lower
        _is_nvidia = "integrate.api.nvidia.com" in self._base_url_lower
        _is_kimi = (
            base_url_host_matches(self.base_url, "api.kimi.com")
            or base_url_host_matches(self.base_url, "moonshot.ai")
            or base_url_host_matches(self.base_url, "moonshot.cn")
        )
        _is_tokenhub = base_url_host_matches(self._base_url_lower, "tokenhub.tencentmaas.com")
        _is_lmstudio = (self.provider or "").strip().lower() == "lmstudio"

        # Temperature: _fixed_temperature_for_model may return OMIT_TEMPERATURE
        # sentinel (temperature omitted entirely), a numeric override, or None.
        try:
            from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
            _ft = _fixed_temperature_for_model(self.model, self.base_url)
            _omit_temp = _ft is OMIT_TEMPERATURE
            _fixed_temp = _ft if not _omit_temp else None
        except Exception:
            _omit_temp = False
            _fixed_temp = None

        # Provider preferences (OpenRouter-style)
        _prefs: Dict[str, Any] = {}
        if self.providers_allowed:
            _prefs["only"] = self.providers_allowed
        if self.providers_ignored:
            _prefs["ignore"] = self.providers_ignored
        if self.providers_order:
            _prefs["order"] = self.providers_order
        if self.provider_sort:
            _prefs["sort"] = self.provider_sort
        if self.provider_require_parameters:
            _prefs["require_parameters"] = True
        if self.provider_data_collection:
            _prefs["data_collection"] = self.provider_data_collection

        # Claude max-output override on aggregators
        _ant_max = None
        if (_is_or or _is_nous) and "claude" in (self.model or "").lower():
            try:
                from agent.anthropic_adapter import _get_anthropic_max_output
                _ant_max = _get_anthropic_max_output(self.model)
            except Exception:
                pass

        # Qwen session metadata
        _qwen_meta = None
        if _is_qwen:
            _qwen_meta = {
                "sessionId": self.session_id or "hermes",
                "promptId": str(uuid.uuid4()),
            }

        # ── Provider profile path (registered providers) ───────────────────
        # Profiles handle per-provider quirks via hooks. When a profile is
        # found, delegate fully; otherwise fall through to the legacy flag path.
        try:
            from providers import get_provider_profile
            _profile = get_provider_profile(self.provider)
        except Exception:
            _profile = None

        if _profile:
            _ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
            if _ephemeral_out is not None:
                self._ephemeral_max_output_tokens = None

            return _ct.build_kwargs(
                model=self.model,
                messages=api_messages,
                tools=self.tools,
                base_url=self.base_url,
                timeout=self._resolved_api_call_timeout(),
                max_tokens=self.max_tokens,
                ephemeral_max_output_tokens=_ephemeral_out,
                max_tokens_param_fn=self._max_tokens_param,
                reasoning_config=self.reasoning_config,
                request_overrides=self.request_overrides,
                session_id=getattr(self, "session_id", None),
                provider_profile=_profile,
                ollama_num_ctx=self._ollama_num_ctx,
                # Context forwarded to profile hooks:
                provider_preferences=_prefs or None,
                openrouter_min_coding_score=self.openrouter_min_coding_score,
                anthropic_max_output=_ant_max,
                supports_reasoning=self._supports_reasoning_extra_body(),
                qwen_session_metadata=_qwen_meta,
            )

        # ── Legacy flag path ────────────────────────────────────────────
        # Reached only when get_provider_profile() returns None — i.e. a
        # completely unknown provider not in providers/ registry.
        _ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
        if _ephemeral_out is not None:
            self._ephemeral_max_output_tokens = None

        # Strip image parts for non-vision models (no-op when vision-capable).
        _msgs_for_chat = self._prepare_messages_for_non_vision_model(api_messages)

        return _ct.build_kwargs(
            model=self.model,
            messages=_msgs_for_chat,
            tools=self.tools,
            base_url=self.base_url,
            timeout=self._resolved_api_call_timeout(),
            max_tokens=self.max_tokens,
            ephemeral_max_output_tokens=_ephemeral_out,
            max_tokens_param_fn=self._max_tokens_param,
            reasoning_config=self.reasoning_config,
            request_overrides=self.request_overrides,
            session_id=getattr(self, "session_id", None),
            model_lower=(self.model or "").lower(),
            is_openrouter=_is_or,
            is_nous=_is_nous,
            is_qwen_portal=_is_qwen,
            is_github_models=_is_gh,
            is_nvidia_nim=_is_nvidia,
            is_kimi=_is_kimi,
            is_tokenhub=_is_tokenhub,
            is_lmstudio=_is_lmstudio,
            is_custom_provider=self.provider == "custom",
            ollama_num_ctx=self._ollama_num_ctx,
            provider_preferences=_prefs or None,
            openrouter_min_coding_score=self.openrouter_min_coding_score,
            qwen_prepare_fn=self._qwen_prepare_chat_messages if _is_qwen else None,
            qwen_prepare_inplace_fn=self._qwen_prepare_chat_messages_inplace if _is_qwen else None,
            qwen_session_metadata=_qwen_meta,
            fixed_temperature=_fixed_temp,
            omit_temperature=_omit_temp,
            supports_reasoning=self._supports_reasoning_extra_body(),
            github_reasoning_extra=self._github_models_reasoning_extra_body() if _is_gh else None,
            lmstudio_reasoning_options=self._lmstudio_reasoning_options_cached() if _is_lmstudio else None,
            anthropic_max_output=_ant_max,
            provider_name=self.provider,
        )


    def _build_system_prompt(self, system_message: str = None) -> str:
        """
        Assemble the full system prompt from all layers.
        
        Called once per session (cached on self._cached_system_prompt) and only
        rebuilt after context compression events. This ensures the system prompt
        is stable across all turns in a session, maximizing prefix cache hits.
        """
        # Layers (in order):
        #   1. Agent identity — SOUL.md when available, else DEFAULT_AGENT_IDENTITY
        #   2. User / gateway system prompt (if provided)
        #   3. Persistent memory (frozen snapshot)
        #   4. Skills guidance (if skills tools are loaded)
        #   5. Context files (AGENTS.md, .cursorrules — SOUL.md excluded here when used as identity)
        #   6. Current date & time (frozen at build time)
        #   7. Platform-specific formatting hint

        # Try SOUL.md as primary identity unless the caller explicitly skipped it.
        # Some execution modes (cron) still want HERMES_HOME persona while keeping
        # cwd project instructions disabled.
        _soul_loaded = False
        if self.load_soul_identity or not self.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                prompt_parts = [_soul_content]
                _soul_loaded = True

        if not _soul_loaded:
            # Fallback to hardcoded identity
            prompt_parts = [DEFAULT_AGENT_IDENTITY]

        # Pointer to the hermes-agent skill + docs for user questions about Hermes itself.
        prompt_parts.append(HERMES_AGENT_HELP_GUIDANCE)

        # Tool-aware behavioral guidance: only inject when the tools are loaded
        tool_guidance = []
        if "memory" in self.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in self.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in self.valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        # Kanban worker/orchestrator lifecycle — only present when the
        # dispatcher spawned this process (kanban_show check_fn gates on
        # HERMES_KANBAN_TASK env var). Normal chat sessions never see
        # this block.
        if "kanban_show" in self.valid_tool_names:
            tool_guidance.append(KANBAN_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        # Computer-use (macOS) — goes in as its own block rather than being
        # merged into tool_guidance because the content is multi-paragraph.
        if "computer_use" in self.valid_tool_names:
            from agent.prompt_builder import COMPUTER_USE_GUIDANCE
            prompt_parts.append(COMPUTER_USE_GUIDANCE)

        nous_subscription_prompt = build_nous_subscription_prompt(self.valid_tool_names)
        if nous_subscription_prompt:
            prompt_parts.append(nous_subscription_prompt)
        # Tool-use enforcement: tells the model to actually call tools instead
        # of describing intended actions.  Controlled by config.yaml
        # agent.tool_use_enforcement:
        #   "auto" (default) — matches TOOL_USE_ENFORCEMENT_MODELS
        #   true  — always inject (all models)
        #   false — never inject
        #   list  — custom model-name substrings to match
        if self.valid_tool_names:
            _enforce = self._tool_use_enforcement
            _inject = False
            if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in ("true", "always", "yes", "on")):
                _inject = True
            elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in ("false", "never", "no", "off")):
                _inject = False
            elif isinstance(_enforce, list):
                model_lower = (self.model or "").lower()
                _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
            else:
                # "auto" or any unrecognised value — use hardcoded defaults
                model_lower = (self.model or "").lower()
                _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
            if _inject:
                prompt_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
                _model_lower = (self.model or "").lower()
                # Google model operational guidance (conciseness, absolute
                # paths, parallel tool calls, verify-before-edit, etc.)
                if "gemini" in _model_lower or "gemma" in _model_lower:
                    prompt_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                # OpenAI GPT/Codex execution discipline (tool persistence,
                # prerequisite checks, verification, anti-hallucination).
                if "gpt" in _model_lower or "codex" in _model_lower:
                    prompt_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

        # so it can refer the user to them rather than reinventing answers.

        # Note: ephemeral_system_prompt is NOT included here. It's injected at
        # API-call time only so it stays out of the cached/stored system prompt.
        if system_message is not None:
            prompt_parts.append(system_message)

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            # USER.md is always included when enabled.
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        # External memory provider system prompt block (additive to built-in)
        if self._memory_manager:
            try:
                _ext_mem_block = self._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    prompt_parts.append(_ext_mem_block)
            except Exception:
                pass

        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            avail_toolsets = {
                toolset
                for toolset in (
                    get_toolset_for_tool(tool_name) for tool_name in self.valid_tool_names
                )
                if toolset
            }
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        if not self.skip_context_files:
            # Use TERMINAL_CWD for context file discovery when set (gateway
            # mode).  The gateway process runs from the hermes-agent install
            # dir, so os.getcwd() would pick up the repo's AGENTS.md and
            # other dev files — inflating token usage by ~10k for no benefit.
            _context_cwd = os.getenv("TERMINAL_CWD") or None
            context_files_prompt = build_context_files_prompt(
                cwd=_context_cwd, skip_soul=_soul_loaded)
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\nSession ID: {self.session_id}"
        if self.model:
            timestamp_line += f"\nModel: {self.model}"
        if self.provider:
            timestamp_line += f"\nProvider: {self.provider}"
        prompt_parts.append(timestamp_line)

        # Alibaba Coding Plan API always returns "glm-4.7" as model name regardless
        # of the requested model. Inject explicit model identity into the system prompt
        # so the agent can correctly report which model it is (workaround for API bug).
        if self.provider == "alibaba":
            _model_short = self.model.split("/")[-1] if "/" in self.model else self.model
            prompt_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {self.model}. "
                f"When asked what model you are, always answer based on this information, "
                f"not on any model name returned by the API."
            )

        # Environment hints (WSL, Termux, etc.) — tell the agent about the
        # execution environment so it can translate paths and adapt behavior.
        _env_hints = build_environment_hints()
        if _env_hints:
            prompt_parts.append(_env_hints)

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])
        elif platform_key:
            # Check plugin registry for platform-specific LLM guidance
            try:
                from gateway.platform_registry import platform_registry
                _entry = platform_registry.get(platform_key)
                if _entry and _entry.platform_hint:
                    prompt_parts.append(_entry.platform_hint)
            except Exception:
                pass

        return "\n\n".join(p.strip() for p in prompt_parts if p.strip())

    # =========================================================================
    # Pre/post-call guardrails (inspired by PR #1321 — @alireza78a)
    # =========================================================================


    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """Build a normalized assistant message dict from an API response message.

        Handles reasoning extraction, reasoning_details, and optional tool_calls
        so both the tool-call path and the final-response path share one builder.
        """
        assistant_tool_calls = getattr(assistant_message, "tool_calls", None)
        reasoning_text = self._extract_reasoning(assistant_message)
        _from_structured = bool(reasoning_text)

        # Fallback: extract inline <think> blocks from content when no structured
        # reasoning fields are present (some models/providers embed thinking
        # directly in the content rather than returning separate API fields).
        if not reasoning_text:
            content = assistant_message.content or ""
            think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
            if think_blocks:
                combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
                reasoning_text = combined or None

        if reasoning_text and self.verbose_logging:
            logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

        if reasoning_text and self.reasoning_callback:
            # Skip callback when streaming is active — reasoning was already
            # displayed during the stream via one of two paths:
            #   (a) _fire_reasoning_delta (structured reasoning_content deltas)
            #   (b) _stream_delta tag extraction (<think>/<REASONING_SCRATCHPAD>)
            # When streaming is NOT active, always fire so non-streaming modes
            # (gateway, batch, quiet) still get reasoning.
            # Any reasoning that wasn't shown during streaming is caught by the
            # CLI post-response display fallback (cli.py _reasoning_shown_this_turn).
            if not self.stream_delta_callback and not self._stream_callback:
                try:
                    self.reasoning_callback(reasoning_text)
                except Exception:
                    pass

        # Sanitize surrogates from API response — some models (e.g. Kimi/GLM via Ollama)
        # can return invalid surrogate code points that crash json.dumps() on persist.
        _raw_content = assistant_message.content or ""
        _san_content = _sanitize_surrogates(_raw_content)
        if reasoning_text:
            reasoning_text = _sanitize_surrogates(reasoning_text)

        # Strip inline reasoning tags (<think>…</think> etc.) from the stored
        # assistant content.  Reasoning was already captured into
        # ``reasoning_text`` above (either from structured fields or the
        # inline-block fallback), so the raw tags in content are redundant.
        # Leaving them in place caused reasoning to leak to messaging
        # platforms (#8878, #9568), inflate context on subsequent turns
        # (#9306 observed 16% content-size reduction on a real MiniMax
        # session), and pollute generated session titles.  One strip at the
        # storage boundary cleans content for every downstream consumer:
        # API replay, session transcript, gateway delivery, CLI display,
        # compression, title generation.
        if isinstance(_san_content, str) and _san_content:
            _san_content = self._strip_think_blocks(_san_content).strip()

        msg = {
            "role": "assistant",
            "content": _san_content,
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

        raw_reasoning_content = getattr(assistant_message, "reasoning_content", None)
        if raw_reasoning_content is None and hasattr(assistant_message, "model_extra"):
            model_extra = getattr(assistant_message, "model_extra", None) or {}
            if isinstance(model_extra, dict) and "reasoning_content" in model_extra:
                raw_reasoning_content = model_extra["reasoning_content"]
        if raw_reasoning_content is not None:
            msg["reasoning_content"] = _sanitize_surrogates(raw_reasoning_content)
        elif assistant_tool_calls and self._needs_thinking_reasoning_pad():
            # DeepSeek v4 thinking mode and Kimi / Moonshot thinking mode
            # both require reasoning_content on every assistant tool-call
            # message. Without it, replaying the persisted message causes
            # HTTP 400 ("The reasoning_content in the thinking mode must
            # be passed back to the API"). Include streamed reasoning
            # text when captured; otherwise pad with a single space —
            # DeepSeek V4 Pro tightened validation and rejects empty
            # string ("The reasoning content in the thinking mode must
            # be passed back to the API"). A space satisfies non-empty
            # checks everywhere without leaking fabricated reasoning.
            # Refs #15250, #17400, #17341.
            msg["reasoning_content"] = reasoning_text or " "

        # Additive fallback (refs #16844, #16884). Streaming-only providers
        # (glm, MiniMax, gpt-5.x via aigw, Anthropic via openai-compat shims)
        # accumulate reasoning through ``delta.reasoning_content`` chunks
        # but never land it on the message object as a top-level attribute,
        # so neither branch above fires and the chain-of-thought is stored
        # only under the internal ``reasoning`` key. When the user later
        # replays that history through a DeepSeek-v4 / Kimi thinking model,
        # the missing ``reasoning_content`` causes HTTP 400 ("The
        # reasoning_content in the thinking mode must be passed back to the
        # API.").
        #
        # Promote the already-sanitized streamed ``reasoning_text`` to
        # ``reasoning_content`` at write time, but ONLY when no prior branch
        # already set it AND we actually captured reasoning text. This
        # preserves every existing behavior:
        #   - SDK-exposed ``reasoning_content`` (OpenAI/Moonshot/DeepSeek SDK)
        #     still wins.
        #   - DeepSeek tool-call ""-pad (#15250) still fires.
        #   - Non-thinking turns with no reasoning leave the field absent,
        #     so ``_copy_reasoning_content_for_api``'s cross-provider leak
        #     guard (#15748) and ``reasoning``→``reasoning_content``
        #     promotion tiers still apply at replay time.
        if "reasoning_content" not in msg and reasoning_text:
            msg["reasoning_content"] = reasoning_text

        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            # Pass reasoning_details back unmodified so providers (OpenRouter,
            # Anthropic, OpenAI) can maintain reasoning continuity across turns.
            # Each provider may include opaque fields (signature, encrypted_content)
            # that must be preserved exactly.
            raw_details = assistant_message.reasoning_details
            preserved = []
            for d in raw_details:
                if isinstance(d, dict):
                    preserved.append(d)
                elif hasattr(d, "__dict__"):
                    preserved.append(d.__dict__)
                elif hasattr(d, "model_dump"):
                    preserved.append(d.model_dump())
            if preserved:
                msg["reasoning_details"] = preserved

        # Codex Responses API: preserve encrypted reasoning items for
        # multi-turn continuity. These get replayed as input on the next turn.
        codex_items = getattr(assistant_message, "codex_reasoning_items", None)
        if codex_items:
            msg["codex_reasoning_items"] = codex_items

        # Codex Responses API: preserve exact assistant message items (with
        # id/phase) so follow-up turns can replay structured items instead of
        # flattening to plain text. This is required for prefix cache hits.
        codex_message_items = getattr(assistant_message, "codex_message_items", None)
        if codex_message_items:
            msg["codex_message_items"] = codex_message_items

        if assistant_tool_calls:
            tool_calls = []
            for tool_call in assistant_tool_calls:
                raw_id = getattr(tool_call, "id", None)
                call_id = getattr(tool_call, "call_id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    embedded_call_id, _ = self._split_responses_tool_id(raw_id)
                    call_id = embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_id, str) and raw_id.strip():
                        call_id = raw_id.strip()
                    else:
                        _fn = getattr(tool_call, "function", None)
                        _fn_name = getattr(_fn, "name", "") if _fn else ""
                        _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                        call_id = self._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
                call_id = call_id.strip()

                response_item_id = getattr(tool_call, "response_item_id", None)
                if not isinstance(response_item_id, str) or not response_item_id.strip():
                    _, embedded_response_item_id = self._split_responses_tool_id(raw_id)
                    response_item_id = embedded_response_item_id

                response_item_id = self._derive_responses_function_call_id(
                    call_id,
                    response_item_id if isinstance(response_item_id, str) else None,
                )

                tc_dict = {
                    "id": call_id,
                    "call_id": call_id,
                    "response_item_id": response_item_id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    },
                }
                # Preserve extra_content (e.g. Gemini thought_signature) so it
                # is sent back on subsequent API calls.  Without this, Gemini 3
                # thinking models reject the request with a 400 error.
                extra = getattr(tool_call, "extra_content", None)
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        extra = extra.model_dump()
                    tc_dict["extra_content"] = extra
                tool_calls.append(tc_dict)
            msg["tool_calls"] = tool_calls

        return msg


    def _format_tools_for_system_message(self) -> str:
        """
        Format tool definitions for the system message in the trajectory format.
        
        Returns:
            str: JSON string representation of tool definitions
        """
        if not self.tools:
            return "[]"
        
        # Convert tool definitions to the format expected in trajectories
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None  # Match the format in the example
            }
            formatted_tools.append(formatted_tool)
        
        return json.dumps(formatted_tools, ensure_ascii=False)


    def _dump_api_request_debug(
        self,
        api_kwargs: Dict[str, Any],
        *,
        reason: str,
        error: Optional[Exception] = None,
    ) -> Optional[Path]:
        """
        Dump a debug-friendly HTTP request record for the active inference API.

        Captures the request body from api_kwargs (excluding transport-only keys
        like timeout). Intended for debugging provider-side 4xx failures where
        retries are not useful.
        """
        try:
            body = copy.deepcopy(api_kwargs)
            body.pop("timeout", None)
            body = {k: v for k, v in body.items() if v is not None}

            api_key = None
            try:
                api_key = getattr(self.client, "api_key", None)
            except Exception as e:
                logger.debug("Could not extract API key for debug dump: %s", e)

            dump_payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "reason": reason,
                "request": {
                    "method": "POST",
                    "url": f"{self.base_url.rstrip('/')}{'/responses' if self.api_mode == 'codex_responses' else '/chat/completions'}",
                    "headers": {
                        "Authorization": f"Bearer {self._mask_api_key_for_logs(api_key)}",
                        "Content-Type": "application/json",
                    },
                    "body": body,
                },
            }

            if error is not None:
                error_info: Dict[str, Any] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for attr_name in ("status_code", "request_id", "code", "param", "type"):
                    attr_value = getattr(error, attr_name, None)
                    if attr_value is not None:
                        error_info[attr_name] = attr_value

                body_attr = getattr(error, "body", None)
                if body_attr is not None:
                    error_info["body"] = body_attr

                response_obj = getattr(error, "response", None)
                if response_obj is not None:
                    try:
                        error_info["response_status"] = getattr(response_obj, "status_code", None)
                        error_info["response_text"] = response_obj.text
                    except Exception as e:
                        logger.debug("Could not extract error response details: %s", e)

                dump_payload["error"] = error_info

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dump_file = self.logs_dir / f"request_dump_{self.session_id}_{timestamp}.json"
            dump_file.write_text(
                json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            self._vprint(f"{self.log_prefix}🧾 Request debug dump written to: {dump_file}")

            if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
                print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

            return dump_file
        except Exception as dump_error:
            if self.verbose_logging:
                logging.warning(f"Failed to dump API request debug payload: {dump_error}")
            return None


    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """Spawn a background thread to review the conversation for memory/skill saves.

        Creates a full AIAgent fork with the same model, tools, and context as the
        main session. The review prompt is appended as the next user turn in the
        forked conversation. Writes directly to the shared memory/skill stores.
        Never modifies the main conversation history or produces user-visible output.
        """
        import threading

        # Pick the right prompt based on which triggers fired
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW_PROMPT
        elif review_memory:
            prompt = self._MEMORY_REVIEW_PROMPT
        else:
            prompt = self._SKILL_REVIEW_PROMPT

        def _run_review():
            import contextlib
            # Install a non-interactive approval callback on this worker
            # thread so any dangerous-command guard the review agent trips
            # resolves to "deny" instead of falling back to input() -- which
            # deadlocks against the parent's prompt_toolkit TUI (#15216).
            # Same pattern as _subagent_auto_deny in tools/delegate_tool.py.
            def _bg_review_auto_deny(command, description, **kwargs):
                logger.warning(
                    "Background review auto-denied dangerous command: %s (%s)",
                    command, description,
                )
                return "deny"
            try:
                _set_approval_callback(_bg_review_auto_deny)
            except Exception:
                pass
            review_agent = None
            try:
                with open(os.devnull, "w", encoding="utf-8") as _devnull, \
                     contextlib.redirect_stdout(_devnull), \
                     contextlib.redirect_stderr(_devnull):
                    # Inherit the parent agent's live runtime (provider, model,
                    # base_url, api_key, api_mode) so the fork uses the exact
                    # same credentials the main turn is using.  Without this,
                    # AIAgent.__init__ re-runs auto-resolution from env vars,
                    # which fails for OAuth-only providers, session-scoped
                    # creds, or credential-pool setups where the resolver can't
                    # reconstruct auth from scratch -- producing the spurious
                    # "No LLM provider configured" warning at end of turn.
                    _parent_runtime = self._current_main_runtime()
                    review_agent = AIAgent(
                        model=self.model,
                        max_iterations=16,
                        quiet_mode=True,
                        platform=self.platform,
                        provider=self.provider,
                        api_mode=_parent_runtime.get("api_mode") or None,
                        base_url=_parent_runtime.get("base_url") or None,
                        api_key=_parent_runtime.get("api_key") or None,
                        credential_pool=getattr(self, "_credential_pool", None),
                        parent_session_id=self.session_id,
                        enabled_toolsets=["memory", "skills"],
                    )
                    review_agent._memory_write_origin = "background_review"
                    review_agent._memory_write_context = "background_review"
                    review_agent._memory_store = self._memory_store
                    review_agent._memory_enabled = self._memory_enabled
                    review_agent._user_profile_enabled = self._user_profile_enabled
                    review_agent._memory_nudge_interval = 0
                    review_agent._skill_nudge_interval = 0
                    # Suppress all status/warning emits from the fork so the
                    # user only sees the final successful-action summary.
                    # Without this, mid-review "Iteration budget exhausted",
                    # rate-limit retries, compression warnings, and other
                    # lifecycle messages bubble up through _emit_status ->
                    # _vprint and leak past the stdout redirect (they go via
                    # _print_fn/status_callback, which bypass sys.stdout).
                    review_agent.suppress_status_output = True

                    review_agent.run_conversation(
                        user_message=prompt,
                        conversation_history=messages_snapshot,
                    )

                # Scan the review agent's messages for successful tool actions
                # and surface a compact summary to the user. Tool messages
                # already present in messages_snapshot must be skipped, since
                # the review agent inherits that history and would otherwise
                # re-surface stale "created"/"updated" messages from the prior
                # conversation as if they just happened (issue #14944).
                actions = self._summarize_background_review_actions(
                    getattr(review_agent, "_session_messages", []),
                    messages_snapshot,
                )

                if actions:
                    summary = " · ".join(dict.fromkeys(actions))
                    self._safe_print(
                        f"  💾 Self-improvement review: {summary}"
                    )
                    _bg_cb = self.background_review_callback
                    if _bg_cb:
                        try:
                            _bg_cb(
                                f"💾 Self-improvement review: {summary}"
                            )
                        except Exception:
                            pass

            except Exception as e:
                logger.warning("Background memory/skill review failed: %s", e)
                self._emit_auxiliary_failure("background review", e)
            finally:
                # Background review agents can initialize memory providers
                # (for example Hindsight) that own their own network clients.
                # Explicitly stop those providers before closing the agent so
                # their aiohttp sessions do not leak until GC/process exit.
                # Then close all remaining resources (httpx client,
                # subprocesses, etc.) so GC doesn't try to clean them up on a
                # dead asyncio event loop (which produces "Event loop is
                # closed" errors).
                if review_agent is not None:
                    try:
                        review_agent.shutdown_memory_provider()
                    except Exception:
                        pass
                    try:
                        review_agent.close()
                    except Exception:
                        pass
                # Clear the approval callback on this bg-review thread so a
                # recycled thread-id doesn't inherit a stale reference.
                try:
                    _set_approval_callback(None)
                except Exception:
                    pass

        t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
        t.start()


    def _summarize_background_review_actions(
        review_messages: List[Dict],
        prior_snapshot: List[Dict],
    ) -> List[str]:
        """Build the human-facing action summary for a background review pass.

        Walks the review agent's session messages and collects "successful tool
        action" descriptions to surface to the user (e.g. "Memory updated").
        Tool messages already present in ``prior_snapshot`` are skipped so we
        don't re-surface stale results from the prior conversation that the
        review agent inherited via ``conversation_history`` (issue #14944).

        Matching is by ``tool_call_id`` when available, with a content-equality
        fallback for tool messages that lack one.
        """
        existing_tool_call_ids = set()
        existing_tool_contents = set()
        for prior in prior_snapshot or []:
            if not isinstance(prior, dict) or prior.get("role") != "tool":
                continue
            tcid = prior.get("tool_call_id")
            if tcid:
                existing_tool_call_ids.add(tcid)
            else:
                content = prior.get("content")
                if isinstance(content, str):
                    existing_tool_contents.add(content)

        actions: List[str] = []
        for msg in review_messages or []:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            tcid = msg.get("tool_call_id")
            if tcid and tcid in existing_tool_call_ids:
                continue
            if not tcid:
                content_str = msg.get("content")
                if isinstance(content_str, str) and content_str in existing_tool_contents:
                    continue
            try:
                data = json.loads(msg.get("content", "{}"))
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict) or not data.get("success"):
                continue
            message = data.get("message", "")
            target = data.get("target", "")
            if "created" in message.lower():
                actions.append(message)
            elif "updated" in message.lower():
                actions.append(message)
            elif "added" in message.lower() or (target and "add" in message.lower()):
                label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                actions.append(f"{label} updated")
            elif "Entry added" in message:
                label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                actions.append(f"{label} updated")
            elif "removed" in message.lower() or "replaced" in message.lower():
                label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                actions.append(f"{label} updated")
        return actions


    def close(self) -> None:
        """Release all resources held by this agent instance.

        Cleans up subprocess resources that would otherwise become orphans:
        - Background processes tracked in ProcessRegistry
        - Terminal sandbox environments
        - Browser daemon sessions
        - Active child agents (subagent delegation)
        - OpenAI/httpx client connections

        Safe to call multiple times (idempotent).  Each cleanup step is
        independently guarded so a failure in one does not prevent the rest.
        """
        task_id = getattr(self, "session_id", None) or ""

        # 1. Kill background processes for this task
        try:
            from tools.process_registry import process_registry
            process_registry.kill_all(task_id=task_id)
        except Exception:
            pass

        # 2. Clean terminal sandbox environments
        try:
            cleanup_vm(task_id)
        except Exception:
            pass

        # 3. Clean browser daemon sessions
        try:
            cleanup_browser(task_id)
        except Exception:
            pass

        # 4. Close active child agents
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.close()
                except Exception:
                    pass
        except Exception:
            pass

        # 5. Close the OpenAI/httpx client
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="agent_close", shared=True)
                self.client = None
        except Exception:
            pass


    def release_clients(self) -> None:
        """Release LLM client resources WITHOUT tearing down session tool state.

        Used by the gateway when evicting this agent from _agent_cache for
        memory-management reasons (LRU cap or idle TTL) — the session may
        resume at any time with a freshly-built AIAgent that reuses the
        same task_id / session_id, so we must NOT kill:
          - process_registry entries for task_id (user's bg shells)
          - terminal sandbox for task_id (cwd, env, shell state)
          - browser daemon for task_id (open tabs, cookies)
          - memory provider (has its own lifecycle; keeps running)

        We DO close:
          - OpenAI/httpx client pool (big chunk of held memory + sockets;
            the rebuilt agent gets a fresh client anyway)
          - Active child subagents (per-turn artefacts; safe to drop)

        Safe to call multiple times.  Distinct from close() — which is the
        hard teardown for actual session boundaries (/new, /reset, session
        expiry).
        """
        # Close active child agents (per-turn; no cross-turn persistence).
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.release_clients()
                except Exception:
                    # Fall back to full close on children; they're per-turn.
                    try:
                        child.close()
                    except Exception:
                        pass
        except Exception:
            pass

        # Close the OpenAI/httpx client to release sockets immediately.
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="cache_evict", shared=True)
                self.client = None
        except Exception:
            pass


    def _looks_like_codex_intermediate_ack(
        self,
        user_message: str,
        assistant_content: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """Detect a planning/ack message that should continue instead of ending the turn."""
        if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
            return False

        assistant_text = self._strip_think_blocks(assistant_content or "").strip().lower()
        if not assistant_text:
            return False
        if len(assistant_text) > 1200:
            return False

        has_future_ack = bool(
            re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
        )
        if not has_future_ack:
            return False

        action_markers = (
            "look into",
            "look at",
            "inspect",
            "scan",
            "check",
            "analyz",
            "review",
            "explore",
            "read",
            "open",
            "run",
            "test",
            "fix",
            "debug",
            "search",
            "find",
            "walkthrough",
            "report back",
            "summarize",
        )
        workspace_markers = (
            "directory",
            "current directory",
            "current dir",
            "cwd",
            "repo",
            "repository",
            "codebase",
            "project",
            "folder",
            "filesystem",
            "file tree",
            "files",
            "path",
        )

        user_text = (user_message or "").strip().lower()
        user_targets_workspace = (
            any(marker in user_text for marker in workspace_markers)
            or "~/" in user_text
            or "/" in user_text
        )
        assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
        assistant_targets_workspace = any(
            marker in assistant_text for marker in workspace_markers
        )
        return (user_targets_workspace or assistant_targets_workspace) and assistant_mentions_action


    def _summarize_api_error(error: Exception) -> str:
        """Extract a human-readable one-liner from an API error.

        Handles Cloudflare HTML error pages (502, 503, etc.) by pulling the
        <title> tag instead of dumping raw HTML.  Falls back to a truncated
        str(error) for everything else.
        """
        raw = str(error)

        # Cloudflare / proxy HTML pages: grab the <title> for a clean summary
        if "<!DOCTYPE" in raw or "<html" in raw:
            m = re.search(r"<title[^>]*>([^<]+)</title>", raw, re.IGNORECASE)
            title = m.group(1).strip() if m else "HTML error page (title not found)"
            # Also grab Cloudflare Ray ID if present
            ray = re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", raw)
            ray_id = ray.group(1).strip() if ray else None
            status_code = getattr(error, "status_code", None)
            parts = []
            if status_code:
                parts.append(f"HTTP {status_code}")
            parts.append(title)
            if ray_id:
                parts.append(f"Ray {ray_id}")
            return " — ".join(parts)

        # JSON body errors from OpenAI/Anthropic SDKs
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
            if msg:
                status_code = getattr(error, "status_code", None)
                prefix = f"HTTP {status_code}: " if status_code else ""
                return f"{prefix}{msg[:300]}"

        # Fallback: truncate the raw string but give more room than 200 chars
        status_code = getattr(error, "status_code", None)
        prefix = f"HTTP {status_code}: " if status_code else ""
        return f"{prefix}{raw[:500]}"


    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        Get messages up to (but not including) the last assistant turn.
        
        This is used when we need to "roll back" to the last successful point
        in the conversation, typically when the final assistant message is
        incomplete or malformed.
        
        Args:
            messages: Full message list
            
        Returns:
            Messages up to the last complete assistant turn (ending with user/tool message)
        """
        if not messages:
            return []
        
        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        
        if last_assistant_idx is None:
            # No assistant message found, return all messages
            return messages.copy()
        
        # Return everything up to (not including) the last assistant message
        return messages[:last_assistant_idx]


    def _api_kwargs_have_image_parts(api_kwargs: dict) -> bool:
        """Return True when the outbound request still contains native image parts."""
        if not isinstance(api_kwargs, dict):
            return False
        candidates = []
        messages = api_kwargs.get("messages")
        if isinstance(messages, list):
            candidates.extend(messages)
        # Responses API payloads use `input`; after conversion, image parts can
        # still be present there instead of in `messages`.
        response_input = api_kwargs.get("input")
        if isinstance(response_input, list):
            candidates.extend(response_input)

        def _contains_image(value: Any) -> bool:
            if isinstance(value, dict):
                ptype = value.get("type")
                if ptype in {"image_url", "input_image"}:
                    return True
                return any(_contains_image(v) for v in value.values())
            if isinstance(value, list):
                return any(_contains_image(v) for v in value)
            return False

        return any(_contains_image(item) for item in candidates)


    def _invalidate_system_prompt(self):
        """
        Invalidate the cached system prompt, forcing a rebuild on the next turn.
        
        Called after context compression events. Also reloads memory from disk
        so the rebuilt prompt captures any writes from this session.
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()


    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""
        self._base_url_hostname = base_url_hostname(value)
