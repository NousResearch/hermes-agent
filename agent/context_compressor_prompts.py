"""Summary prompt text for context compaction (byte-pinned wording)."""

from __future__ import annotations

_NO_USER_TASK_SENTINEL = "None. This session contains no user-authored turns."


# Per-section summarizer instructions, keyed by "the transcript has a real user turn". Wording
# is deliberately plain: Azure/OpenAI content filters have flagged stronger "injection" /
# "do not respond" framing. Prompt text is byte-pinned — restructure code around it only.
_SECTION_INSTRUCTIONS: dict[bool, dict[str, str]] = {
    True: {
        "language": (
            "Write the summary in the same language the user was using in the "
            "conversation — do not translate or switch to English. "
        ),
        "historical_task": """[THE SINGLE MOST IMPORTANT FIELD. Identify the user's most recent unfulfilled
input precisely, but summarize it in your own words rather than copying long
passages from the transcript. The compressor inserts a bounded, redacted
snapshot of the real latest user turn after generation, so the model must not
reproduce it.
This includes:
- Explicit task assignments ("<specific user task>")
- Questions awaiting an answer ("<specific user question>")
- Decisions awaiting input ("<option A or B?>")
- Ongoing discussions where the assistant owes the next substantive reply
A conversation where the user just asked a question IS an active task — the
task is "answer that question with full context". Do NOT write "None" merely
because the user did not issue an imperative command; reserve "None" for the
rare case where the last exchange was fully resolved and the user said
something like "thanks, that's all".
If multiple items are outstanding, list only the ones NOT yet completed.
This historical snapshot must identify the latest unresolved user input precisely. Examples:
"User asked for <specific task and constraints>"
"User asked <specific question> — needs investigation + answer"
"User chose <option>; awaiting implementation of <specific next step>"
If the user's most recent message was a reverse signal (stop, undo, roll
back, never mind, just verify, change of topic) that supersedes earlier
work, describe the reverse signal accurately and DO NOT carry forward the
cancelled task.
Example: "User asked to stop the prior task — earlier work is cancelled."
If no outstanding task exists, write "None."]""",
        "goal": "[What the user is trying to accomplish overall]",
        "constraints": (
            "[User preferences, coding style, constraints, important decisions. Any security or safety constraint "
            "the user stated (files/data to avoid, operations that must not be performed, credential-handling rules) "
            "MUST be quoted VERBATIM here so it continues to apply after compaction — never paraphrase those.]"
        ),
        "resolved_questions": (
            "[Questions the user asked that were ALREADY answered — include the answer so it is not repeated]"
        ),
    },
    False: {
        "language": (
            "This session contains no user-authored turns. Write the summary in the dominant language of the "
            "source turns; if they are mixed, use the language of the most recent natural-language assistant "
            "turn. Do not translate, invent a user, or attribute any request to a user. "
        ),
        "historical_task": f"""[NO user-authored turn exists in this session. Write exactly:
{_NO_USER_TASK_SENTINEL}
Do not write "User asked:" or any translated equivalent anywhere in the summary.
Describe agent/tool work only as completed actions, state, or historical work.]""",
        "goal": (
            "[Historical cron/agent objective inferred only from assistant and "
            "tool activity. Never call it a user goal.]"
        ),
        "constraints": (
            "[Runtime, configuration, and technical constraints only. Do not invent user preferences.]"
        ),
        "resolved_questions": "[Write exactly: None. No user-authored questions exist.]",
    },
}
