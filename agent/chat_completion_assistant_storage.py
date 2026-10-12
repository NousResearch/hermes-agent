"""The assistant-row storage projection, split out of ``chat_completion_helpers`` so the
facade can stay under its file-line cap (moved code keeps its cap; see root AGENTS.md)."""

from agent.message_content import flatten_message_text
from agent.message_sanitization import _sanitize_surrogates


def _assistant_content_for_storage(agent, assistant_message):
    # Sanitize surrogates (Kimi/GLM via Ollama emit code points that crash json.dumps),
    # strip inline <think> tags at the storage boundary (they leaked to platforms and
    # polluted titles), then redact inlined credentials before the message enters
    # history / state.db / gateway delivery (no-op with HERMES_REDACT_SECRETS off).
    content = _sanitize_surrogates(flatten_message_text(getattr(assistant_message, "content", None)))
    if isinstance(content, str) and content:
        content = agent._strip_think_blocks(content).strip()
        if content:
            from agent.redact import redact_sensitive_text
            content = redact_sensitive_text(content)
    return content
