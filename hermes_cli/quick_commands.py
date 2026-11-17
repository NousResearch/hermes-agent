"""``type: prompt`` quick commands, shared by the CLI, the TUI/Desktop gateway and the messaging gateway.

A prompt quick command is a named prompt from ``config.yaml`` sent to the agent as the user's turn:
``/release-notes v2.3`` → the configured text. Unlike a skill it has no scaffold and no index entry
in the system prompt; it is exactly the message the user would otherwise paste.
"""

from __future__ import annotations

from typing import Any


def render_prompt_quick_command(qcmd: dict[str, Any], user_args: str) -> str:
    """Agent-turn text for a ``type: prompt`` quick command ("" when it has no prompt).

    ``{args}`` is replaced with what the user typed after the command; without the placeholder the
    arguments are appended as their own paragraph, so ``/release-notes v2.3`` still reaches the model.
    """
    prompt = str(qcmd.get("prompt") or "").strip()
    if not prompt:
        return ""
    args = (user_args or "").strip()
    if "{args}" in prompt:
        return prompt.replace("{args}", args).strip()
    return f"{prompt}\n\n{args}" if args else prompt


def review_button_label(name: str, qcmd: Any) -> str | None:
    """Desktop Review pane button label for a prompt quick command that opts in, else None.

    ``review_button: true`` labels it with its description (or ``/name``); a string is the label.
    """
    if not isinstance(qcmd, dict) or qcmd.get("type") != "prompt" or not str(qcmd.get("prompt") or "").strip():
        return None
    flag = qcmd.get("review_button")
    if isinstance(flag, str) and flag.strip():
        return flag.strip()
    if flag is True:
        return str(qcmd.get("description") or "").strip() or f"/{name}"
    return None
