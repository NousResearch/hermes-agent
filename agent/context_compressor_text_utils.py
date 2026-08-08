"""Text helpers extracted from context_compressor (leaf util, epic #78647).

Pure module-level helpers used across skill-prune, summary serialize, and
identity paths. Extracted first so skill-prune can import without a cycle
back into the god file.

Part of #78645 + #78647.
"""
from __future__ import annotations

from typing import Any

from agent.redact import redact_sensitive_text


def _redact_compaction_text(text: Any) -> str:
    """Redact text that crosses a compaction summary boundary.

    Compaction summaries persist across sessions and are re-injected into
    every subsequent summarizer prompt, so this boundary uses strict mode:

    - ``force=True`` — deliberately overrides ``security.redact_secrets:
      false``. That opt-out targets *live tool output* (e.g. working on the
      redactor itself); a summary is a persistence boundary where a leaked
      credential keeps re-entering prompts indefinitely.
    - ``redact_url_credentials=True`` — OAuth callback codes, magic-link
      tokens, and URL userinfo never need to survive summarization the way
      they must survive live navigation flows.
    """
    return redact_sensitive_text(
        text or "",
        force=True,
        redact_url_credentials=True,
    )

def _content_text_for_contains(content: Any) -> str:
    """Return a best-effort text view of message content.

    Used only for substring checks when we need to know whether we've already
    appended a note to a message. Keeps multimodal lists intact elsewhere.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)
