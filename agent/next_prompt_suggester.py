"""Guess the user's most likely next prompt after a settled turn, as ghost text for the composer.

One auxiliary call (``task="next_prompt"``) on a clipped tail of the last exchange. It never touches
the conversation, system prompt or toolsets, so the main prompt cache is unaffected. ``None`` means
"show nothing": the model answered NONE, the output was unusable, or the call failed."""

import logging
from typing import Any, Optional

from agent.auxiliary_client import call_llm
from agent.message_content import flatten_message_text

logger = logging.getLogger(__name__)

# Tail budget per message: the end of an assistant reply is what the next prompt follows from.
MAX_USER_CHARS = 1000
MAX_ASSISTANT_CHARS = 2000
NEXT_PROMPT_MAX_TOKENS = 200

_PROMPT = (
    "You predict what the user will type next in a chat with an AI assistant. Given the last "
    "exchange, write the single most likely next message from the USER.\n\n"
    "Rules:\n"
    "- Write as the user, in their voice and language, under 15 words.\n"
    "- If the assistant asked a question or made an offer ('Want me to...?', 'Tell me if you want "
    "X'), answer it with the user's most likely short reply, e.g. 'yes, go ahead' or "
    "'yes, brief the worker'.\n"
    "- If the assistant asked the user to do something and report back ('tell me when it's "
    "done'), write 'done' or a similarly short report.\n"
    "- Otherwise write the natural next request, e.g. 'run the full suite and diff against the "
    "baseline'.\n"
    "- Never suggest thanks, acknowledgements, 'got it', 'sounds good', or waiting/passive "
    "messages; reply NONE instead.\n"
    "- Reply NONE only when nothing useful and plausible fits.\n"
    "- Never describe, explain or refer to 'the user' or 'the assistant'; if you can't write the "
    "message itself, reply NONE.\n"
    "- Output only the message: no quotes, no labels, no explanation."
)

# A rambling answer is a model slip (usually the assistant's voice), better shown as nothing.
MAX_SUGGESTION_WORDS = 20
MAX_SUGGESTION_CHARS = 120  # covers languages written without spaces

_QUOTE_PAIRS = {'"': '"', "'": "'", "`": "`", "“": "”", "‘": "’"}


def clean_suggestion(raw: Any) -> Optional[str]:
    """First non-empty line, quotes stripped; ``None`` for NONE, empty or over-long output."""
    text = flatten_message_text(raw)
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    while len(line) >= 2 and _QUOTE_PAIRS.get(line[0]) == line[-1]:
        line = line[1:-1].strip()
    if not line or line.rstrip(".").upper() == "NONE":
        return None
    if len(line.split()) > MAX_SUGGESTION_WORDS or len(line) > MAX_SUGGESTION_CHARS:
        return None
    return line


def last_exchange(history: list) -> tuple[str, str]:
    """(user, assistant) text of the latest assistant message and the user message before it;
    empty when that latest assistant message carries no text (never reaches back to an older reply)."""
    messages = list(history or [])
    idx = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "assistant"), None)
    if idx is None:
        return "", ""
    assistant = flatten_message_text(messages[idx].get("content"))
    user = next((flatten_message_text(m.get("content")) for m in reversed(messages[:idx])
                 if m.get("role") == "user"), "")
    return user, assistant


def generate_next_prompt(
    user_message: str, assistant_message: str, timeout: Optional[float] = None,
    main_runtime: dict | None = None,
) -> Optional[str]:
    """Suggest the user's next prompt, or ``None``. Never raises."""
    user_tail = (user_message or "").strip()[-MAX_USER_CHARS:]
    assistant_tail = (assistant_message or "").strip()[-MAX_ASSISTANT_CHARS:]
    if not user_tail or not assistant_tail:
        return None
    try:
        response = call_llm(
            task="next_prompt",
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": f"User:\n{user_tail}\n\nAssistant:\n{assistant_tail}"},
            ],
            max_tokens=NEXT_PROMPT_MAX_TOKENS, temperature=None, timeout=timeout, main_runtime=main_runtime,
            reasoning_config={"enabled": False},
        )
        return clean_suggestion(response.choices[0].message.content)
    except Exception as exc:  # a missing hint is a non-event
        logger.debug("Next-prompt suggestion failed: %s", exc, exc_info=True)
        return None
