"""Google Gemini (AI Studio) provider profile.

Reports api_mode="chat_completions" but runs on GeminiNativeClient; this
profile carries auth/endpoint metadata and the thinking_config translation hook.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class GeminiProfile(ProviderProfile):
    """Gemini — translate reasoning_config to thinking_config in extra_body."""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Avoid false 429 RESOURCE_EXHAUSTED from huge systemInstruction on Gemini/Antigravity (#76783)."""
        if not messages or not isinstance(messages, list):
            return messages
        first = messages[0]
        if isinstance(first, dict) and first.get("role") == "system":
            system_text = str(first.get("content") or "")
            # If system prompt is large (> 400 chars), keep identity in system and move instructions to first user turn
            if len(system_text) > 400:
                new_messages = [{"role": "system", "content": "You are Hermes Agent. Follow the instructions provided."}]
                rest = messages[1:]
                if rest and isinstance(rest[0], dict) and rest[0].get("role") == "user":
                    first_user = rest[0]
                    user_content = first_user.get("content")
                    if isinstance(user_content, str):
                        new_content = f"<hermes_instructions>\n{system_text}\n</hermes_instructions>\n\n{user_content}"
                        new_messages.append({**first_user, "content": new_content})
                        new_messages.extend(rest[1:])
                        return new_messages
                    elif isinstance(user_content, list):
                        new_content = [{"type": "text", "text": f"<hermes_instructions>\n{system_text}\n</hermes_instructions>\n\n"}] + list(user_content)
                        new_messages.append({**first_user, "content": new_content})
                        new_messages.extend(rest[1:])
                        return new_messages
                else:
                    new_messages.append({"role": "user", "content": f"<hermes_instructions>\n{system_text}\n</hermes_instructions>"})
                    new_messages.extend(rest)
                    return new_messages
        return messages

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Native: ``thinking_config``; OpenAI-compat /openai subpath:
        ``extra_body.google.thinking_config`` (snake_case)."""
        from agent.transports.chat_completions import (
            _build_gemini_thinking_config,
            _is_gemini_openai_compat_base_url,
            _snake_case_gemini_thinking_config,
        )

        raw = _build_gemini_thinking_config(context.get("model") or "", context.get("reasoning_config"))
        if not raw:
            return {}
        if self.name == "gemini" and _is_gemini_openai_compat_base_url(context.get("base_url") or self.base_url):
            thinking_config = _snake_case_gemini_thinking_config(raw)
            return {"extra_body": {"google": {"thinking_config": thinking_config}}} if thinking_config else {}
        return {"thinking_config": raw}


gemini = GeminiProfile(
    name="gemini", aliases=("google", "google-gemini", "google-ai-studio"), api_mode="chat_completions",
    env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta", auth_type="api_key",
    default_aux_model="gemini-3.6-flash",
)

register_provider(gemini)
