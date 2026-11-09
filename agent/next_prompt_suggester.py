"""Guess the user's most likely next prompt after a settled turn, as ghost text for the composer.

One auxiliary call (``task="next_prompt"``) on a clipped tail of the last exchange. It never touches
the conversation, system prompt or toolsets, so the main prompt cache is unaffected. ``None`` means
"show nothing": the model answered NONE, the output was unusable, or the call failed."""

import json
import logging
import re
from typing import Any, Optional

from agent.auxiliary_client import call_llm
from agent.message_content import flatten_message_text

logger = logging.getLogger(__name__)

# Tail budget per message: the end of an assistant reply is what the next prompt follows from.
MAX_USER_CHARS = 1000
MAX_ASSISTANT_CHARS = 2000
MAX_CLARIFY_CHARS = 600  # whole block of answered clarify questions
MAX_CLARIFY_FIELD_CHARS = 200
NEXT_PROMPT_MAX_TOKENS = 200

_PROMPT = (
    "You predict what the user will type next in a chat with an AI assistant. Given the last "
    "exchange, write the single most likely next message from the USER.\n\n"
    "Rules:\n"
    "- Lines like 'Asked: ... / User answered: ...' are choices the user already made this turn. "
    "Treat them as decided: never suggest reversing, repeating or re-asking them, and never "
    "suggest an action the user just declined.\n"
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
    "- Write the literal words the user would type (e.g. 'nothing yet'), not a description of "
    "what they will do or see.\n"
    "- Output only the message: no quotes, no labels, no explanation."
)

# A rambling answer is a model slip (usually the assistant's voice), better shown as nothing.
MAX_SUGGESTION_WORDS = 20
MAX_SUGGESTION_CHARS = 120  # covers languages written without spaces

# Narration rule: "the assistant"/"the user" is a party reference unless it modifies a noun
# ("the assistant module", "the user's table"). "the AI/model/bot" are also ordinary objects
# ("switch the model", "restart the bot"), so they count only as a clause subject.
_PARTY_NOUNS = (
    r"module|model|class|table|id|service|config|name|file|record|object|schema|profile|"
    r"account|session|prompt|message|interface|api|endpoint|type|field|settings?|"
    r"permissions?|data|database|code|test|tests|agent|plugin|handler|function|view|page"
)
_PARTY_REF = re.compile(
    rf"\bthe\s+(?:assistant|user)\b(?!['’]s)(?!\s+(?:{_PARTY_NOUNS})\b)"
    rf"|(?:^|[,;:.]\s*|\b(?:so|and|then|that|because)\s+)the\s+(?:ai|model|bot)\b"
    rf"(?!['’]s)(?!\s+(?:{_PARTY_NOUNS})\b)",
    re.IGNORECASE,
)

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
    if _PARTY_REF.search(line):  # narrating about the parties instead of writing as the user
        return None
    return line


def _tool_names_by_call_id(messages: list) -> dict[str, str]:
    """tool_call_id -> tool name from assistant ``tool_calls``, for results that carry no name."""
    names: dict[str, str] = {}
    for msg in messages:
        for call in (msg.get("tool_calls") or []) if msg.get("role") == "assistant" else []:
            if isinstance(call, dict) and call.get("id"):
                names[call["id"]] = (call.get("function") or {}).get("name") or ""
    return names


def _clarify_pairs(result_messages: list, call_names: dict[str, str]) -> list[str]:
    """``Asked: q / User answered: a`` for each answered clarify response in the tool results."""
    pairs: list[str] = []
    for msg in result_messages:
        if msg.get("role") != "tool":
            continue
        name = msg.get("tool_name") or msg.get("name") or call_names.get(msg.get("tool_call_id") or "")
        if name != "clarify":
            continue
        try:
            responses = json.loads(flatten_message_text(msg.get("content")))["responses"]
            for r in responses:
                if r["status"] != "answered":
                    continue
                question = str(r["question"]).strip()[-MAX_CLARIFY_FIELD_CHARS:]
                answer = r["user_response"]
                if isinstance(answer, list):  # multi-select
                    answer = ", ".join(str(a).strip() for a in answer)
                answer = str(answer).strip()[-MAX_CLARIFY_FIELD_CHARS:]
                if question and answer:
                    pairs.append(f"Asked: {question} / User answered: {answer}")
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    return pairs


def _clip_pairs(pairs: list[str]) -> str:
    """Newest pairs that fit ``MAX_CLARIFY_CHARS``, in original order."""
    kept: list[str] = []
    used = 0
    for pair in reversed(pairs):
        used += len(pair) + 1
        if used > MAX_CLARIFY_CHARS and kept:
            break
        kept.append(pair)
    return "\n".join(reversed(kept))[-MAX_CLARIFY_CHARS:]


def last_exchange(history: list) -> tuple[str, str]:
    """(user, assistant) text of the latest assistant message and the user message before it;
    empty when that latest assistant message carries no text (never reaches back to an older reply).

    Answered ``clarify`` results between the two ride on the user side, since they are the user's
    words that the transcript otherwise records only as tool rows."""
    messages = list(history or [])
    idx = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "assistant"), None)
    if idx is None:
        return "", ""
    assistant = flatten_message_text(messages[idx].get("content"))
    user_idx = next((i for i in range(idx - 1, -1, -1) if messages[i].get("role") == "user"), None)
    if user_idx is None:
        return "", assistant
    user = flatten_message_text(messages[user_idx].get("content")).strip()[-MAX_USER_CHARS:]
    pairs = _clarify_pairs(messages[user_idx + 1:idx], _tool_names_by_call_id(messages[user_idx + 1:idx]))
    if pairs:
        user = f"{user}\n\n{_clip_pairs(pairs)}"
    return user, assistant


def generate_next_prompt(
    user_message: str, assistant_message: str, timeout: Optional[float] = None,
    main_runtime: dict | None = None,
) -> Optional[str]:
    """Suggest the user's next prompt, or ``None``. Never raises."""
    user_tail = (user_message or "").strip()[-(MAX_USER_CHARS + MAX_CLARIFY_CHARS + 2):]
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
