"""Shared one-off LLM requests for non-conversational helpers.

A "one-shot" is a single stateless model call outside any conversation (commit
messages, rename suggestions, summaries): it never touches session history or
prompt caching. Call with explicit ``instructions``/``user_input`` or a registered
``template`` + ``variables`` so prompt engineering stays consistent across
CLI/TUI/desktop. Model selection rides :func:`agent.auxiliary_client.call_llm`:
``main_runtime`` inherits the live session's provider/model, else ``task``
resolves a cheap backend.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from agent.auxiliary_client import call_llm, extract_content_or_reasoning

logger = logging.getLogger(__name__)

# Templates are plain callables (not str.format) so diff/code payloads with
# literal "{" / "}" pass through untouched.
PromptTemplate = Callable[[Dict[str, Any]], Tuple[str, str]]


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + "\n…(truncated)"


_COMMIT_INSTRUCTIONS = (
    "You write git commit messages. Given a diff of staged changes, write ONE concise Conventional Commits "
    "message describing what the change does and why.\n"
    "Rules:\n"
    "- Subject line: type(scope): summary — imperative mood, lower-case, no trailing period, ≤ 72 "
    "characters. Types: feat, fix, refactor, perf, docs, test, build, chore, style, ci.\n"
    "- Omit the scope if it isn't obvious.\n"
    "- Add a short body (wrapped at ~72 cols) ONLY when the change needs explanation; skip it for "
    "small/obvious changes.\n"
    "- Describe the actual change, never restate the diff line-by-line.\n"
    "- Return ONLY the commit message text — no quotes, no markdown fences, no preamble."
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
    # "Regenerate" must yield something new even on greedy/server-pinned
    # temperature models; a nonce isn't enough, so hand back the previous
    # message and require a genuinely different one.
    avoid = _truncate(str(variables.get("avoid") or "").strip(), 1000)
    if avoid:
        parts.append(
            "You already proposed the message below and the user wants a different one. Write a NEW message with "
            "different wording (and, if reasonable, a different emphasis or scope framing) — do not repeat "
            f"it:\n{avoid}"
        )
    return _COMMIT_INSTRUCTIONS, "\n\n".join(parts)


# Registry of named templates; add an entry to give a new surface a reusable prompt.
PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "commit_message": _commit_message_template,
}


def render_template(name: str, variables: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Resolve a registered template into (instructions, user_input); KeyError if unknown."""
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
    """Run a single stateless LLM request and return its text (fence-stripped).

    Raises RuntimeError when no provider is configured (from :func:`call_llm`),
    KeyError for an unknown template, ValueError when the prompt is empty.
    """
    if template:
        instructions, user_input = render_template(template, variables)
    has_instructions = bool((instructions or "").strip())
    if not has_instructions and not (user_input or "").strip():
        raise ValueError("run_oneshot requires a template or instructions/user_input")
    messages = [{"role": "system", "content": instructions}] if has_instructions else []
    messages.append({"role": "user", "content": user_input or ""})
    response = call_llm(
        task=task,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        main_runtime=main_runtime,
    )
    return _strip_code_fence((extract_content_or_reasoning(response) or "").strip())


def _strip_code_fence(text: str) -> str:
    """Drop a single wrapping ``` fence the model may have added."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text
