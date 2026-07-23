"""Shared one-off LLM requests for non-conversational helpers.

A "one-shot" is a single, stateless model call that runs *outside* any
conversation: it never touches a session's history, never breaks prompt
caching, and returns plain text. UI surfaces use it for small generative
chores — a commit message from a diff, a rename suggestion, a summary —
where spinning up an agent turn would be wrong (it would pollute the thread)
and hand-rolling an LLM call at every call site would be worse.

Two ways to call it:

  * ``run_oneshot(instructions=..., user_input=...)`` — caller supplies the
    full prompt.
  * ``run_oneshot(template="commit_message", variables={...})`` — caller
    names a registered template and passes its variables; the template owns
    the prompt engineering so it stays consistent across CLI/TUI/desktop.

Model selection rides the same auxiliary plumbing as title generation
(:func:`agent.auxiliary_client.call_llm`): pass ``main_runtime`` to inherit
the live session's provider/model, otherwise the configured ``task`` (default
``title_generation``) resolves a cheap/fast backend.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from agent.auxiliary_client import call_llm, extract_content_or_reasoning

logger = logging.getLogger(__name__)

# A template turns a variables dict into a (instructions, user_input) pair.
# Templates are plain callables (not str.format) so diff/code payloads with
# literal "{" / "}" pass through untouched.
PromptTemplate = Callable[[Dict[str, Any]], Tuple[str, str]]


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n…(truncated)"


_COMMIT_INSTRUCTIONS = (
    "You write git commit messages. Given a diff of staged changes, write ONE "
    "concise Conventional Commits message describing what the change does and why.\n"
    "Rules:\n"
    "- Subject line: type(scope): summary — imperative mood, lower-case, no "
    "trailing period, ≤ 72 characters. Types: feat, fix, refactor, perf, docs, "
    "test, build, chore, style, ci.\n"
    "- Omit the scope if it isn't obvious.\n"
    "- Add a short body (wrapped at ~72 cols) ONLY when the change needs "
    "explanation; skip it for small/obvious changes.\n"
    "- Describe the actual change, never restate the diff line-by-line.\n"
    "- Return ONLY the commit message text — no quotes, no markdown fences, no "
    "preamble."
)


def _commit_message_template(variables: Dict[str, Any]) -> Tuple[str, str]:
    diff = _truncate(str(variables.get("diff") or ""), 12000)
    recent = _truncate(str(variables.get("recent_commits") or ""), 1500)

    parts = []
    if recent.strip():
        parts.append(
            "Recent commit subjects from this repo (match their style/conventions):\n"
            f"{recent}"
        )
    parts.append("Diff to describe:\n" + (diff or "(no textual diff available)"))

    # "Regenerate" must yield something new even on models that decode greedily
    # / pin temperature server-side. A trailing nonce isn't enough, so we hand
    # back the previous message and require a genuinely different one.
    avoid = _truncate(str(variables.get("avoid") or "").strip(), 1000)
    if avoid:
        parts.append(
            "You already proposed the message below and the user wants a "
            "different one. Write a NEW message with different wording (and, if "
            "reasonable, a different emphasis or scope framing) — do not repeat "
            f"it:\n{avoid}"
        )

    return _COMMIT_INSTRUCTIONS, "\n\n".join(parts)


# Registry of named templates. Add an entry here to give a new surface a
# consistent, reusable prompt without teaching every caller the prompt text.
PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "commit_message": _commit_message_template,
}


def render_template(name: str, variables: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Resolve a registered template into (instructions, user_input).

    Raises KeyError if the template name is unknown so callers fail loudly
    instead of silently sending an empty prompt.
    """
    template = PROMPT_TEMPLATES.get(name)
    if template is None:
        raise KeyError(f"unknown one-shot template: {name}")
    return template(variables or {})


def run_oneshot(
    *,
    instructions: str = "",
    user_input: str = "",
    template: Optional[str] = None,
    variables: Optional[Dict[str, Any]] = None,
    task: str = "title_generation",
    max_tokens: int = 1024,
    temperature: Optional[float] = 0.3,
    timeout: float = 60.0,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> str:
    """Run a single stateless LLM request and return its text.

    Provide either a registered ``template`` (+ ``variables``) or an explicit
    ``instructions`` / ``user_input`` pair. Returns the model's text answer,
    stripped of surrounding whitespace and any wrapping code fence.

    Raises RuntimeError when no LLM provider is configured (surfaced from
    :func:`call_llm`) and KeyError for an unknown template name.
    """
    if template:
        instructions, user_input = render_template(template, variables)

    if not (instructions or "").strip() and not (user_input or "").strip():
        raise ValueError("run_oneshot requires a template or instructions/user_input")

    messages = []
    if (instructions or "").strip():
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": user_input or ""})

    response = call_llm(
        task=task,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        main_runtime=main_runtime,
    )

    text = (extract_content_or_reasoning(response) or "").strip()
    return _strip_code_fence(text)


def _strip_code_fence(text: str) -> str:
    """Drop a single wrapping ``` fence the model may have added."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text
