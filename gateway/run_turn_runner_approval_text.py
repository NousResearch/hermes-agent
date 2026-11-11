"""Plain-text ``/approve`` prompt for adapters whose ``send`` cuts long text instead of splitting it.

ntfy, DingTalk, Photon and LINE (``splits_long_messages = False``) deliver only the first N
characters of a message. The default prompt puts the command first, so on those surfaces a long
command pushed the answer instructions and the deadline past the cut. Here the instructions come
first, and a command that would still be cut is shown shortened with the prompt marked Deny-only
(a command that fits only beside Deny is shown whole, with a notice saying why only Deny is offered):
the same rule and notice the button cards use (``render_exec_approval``).
"""

from __future__ import annotations

from typing import Any

from agent.i18n import t


def _format_exec_approval_fallback(
    command: str, description: str, command_prefix: str, *, allow_permanent: bool = True,
    allow_session: bool = True, smart_denied: bool = False, explanation: str = "") -> str:
    """Render the text fallback from approval capabilities, not platform names. Same words as
    the button card (``BasePlatformAdapter._format_exec_approval``), plus the typed ``/approve``
    steps a surface without buttons needs. The command is never cut: ``/approve`` must only ever
    answer a command the user could read in full (#62359). ``explanation`` is the plain-English
    "what it does" line (``tools/approval_explain``); these adapters split long messages, so it
    never competes with the command for a character budget."""
    from gateway.platforms.base_exec_approval import (
        approval_timeout_seconds, ea_header_text, ea_reason_label_text, format_approval_deadline_line)
    heading = (t("gateway.exec_approval.smart_deny_heading") if smart_denied
               else f"⚠️ **{ea_header_text()}**")

    choices = [t("gateway.exec_approval.text_choice_once", prefix=command_prefix)]
    if not smart_denied and allow_session:
        choices.append(t("gateway.exec_approval.text_choice_session", prefix=command_prefix))
        if allow_permanent:
            choices.append(t("gateway.exec_approval.text_choice_always", prefix=command_prefix))
    choices.append(t("gateway.exec_approval.text_choice_deny", prefix=command_prefix))
    return (
        t("gateway.exec_approval.text_body", heading=heading, command=command,  # whole: send() splits it
          reason_label=ea_reason_label_text(), reason=description)
        + (f"{t('gateway.exec_approval.explain_label')}: {explanation}\n\n" if explanation else "")
        + t("gateway.exec_approval.text_choice_joiner").join(choices[:-1])
        + t("gateway.exec_approval.text_choice_last", choice=choices[-1])
        + format_approval_deadline_line(approval_timeout_seconds()))

def approval_text_prompt(adapter: Any, chat_id: str, session_key: str, request_id: str | None, command: str,
                         description: str, prefix: str, explanation: str = "", **flags: bool) -> str:
    """The plain-text approval prompt for ``adapter``. An adapter that splits long messages gets the
    default layout; one whose send cuts text gets ``_cutting_text_prompt``, and when even that cannot
    show the whole command the request is marked Deny-only like a card that could not.

    ``explanation`` (the plain-English "what it does" line) rides the splitting layout, which is
    never at risk of a cut. The cutting layout is already fitting instructions + command + deadline
    into one fixed message; it shows the command in full instead, which is the stricter contract."""
    from gateway.platforms.base_exec_approval import send_cannot_prove_whole
    if getattr(adapter, "splits_long_messages", True):
        text = _format_exec_approval_fallback(command, description, prefix, explanation=explanation, **flags)
        if not send_cannot_prove_whole(adapter, chat_id, text):  # a capped split drops the instructions too
            return text
    text, deny_only = _cutting_text_prompt(adapter, chat_id, command, description, prefix, **flags)
    if deny_only:
        from tools.approval import restrict_gateway_approval_card_to_deny
        restrict_gateway_approval_card_to_deny(session_key, request_id)
    return text


def _cutting_text_prompt(adapter: Any, chat_id: str, command: str, description: str, prefix: str, *,
                         allow_permanent: bool = True, allow_session: bool = True,
                         smart_denied: bool = False) -> tuple[str, bool]:
    """``(prompt, deny_only)``: instructions and deadline ahead of the command, so a cut can only
    shorten the command; when the whole prompt does not fit, a shortened command and Deny alone.
    LINE has no ``MAX_MESSAGE_LENGTH``, so its 4096 default is used: conservative, never cut."""
    from gateway.platforms.base import fit_prefix
    from gateway.platforms.base_exec_approval import (
        approval_timeout_seconds, code_fence, ea_header_text, fit_fenced, ea_reason_label_text, format_approval_deadline_line)
    limit = adapter.max_message_length_for_chat(chat_id)
    measure = adapter.message_len_fn_for_chat(chat_id)
    heading = t("gateway.exec_approval.smart_deny_heading") if smart_denied else f"⚠️ **{ea_header_text()}**"

    def prompt(choices: list[str], notice: str, shown: str, fence: str) -> str:
        instructions = (t("gateway.exec_approval.text_choice_joiner").join(choices[:-1])
                        + t("gateway.exec_approval.text_choice_last", choice=choices[-1])) if choices[1:] else f"{choices[0]}.\n"
        return (f"{heading}\n\n{instructions}{notice}{format_approval_deadline_line(approval_timeout_seconds())}"
                f"\n\n{ea_reason_label_text()}: {description}\n{fence}\n{shown}\n{fence}")

    choices = [t("gateway.exec_approval.text_choice_once", prefix=prefix)]
    if not smart_denied and allow_session:
        choices.append(t("gateway.exec_approval.text_choice_session", prefix=prefix))
        if allow_permanent:
            choices.append(t("gateway.exec_approval.text_choice_always", prefix=prefix))
    choices.append(t("gateway.exec_approval.text_choice_deny", prefix=prefix))
    full = prompt(choices, "", command, code_fence(command))
    if measure(full) <= limit:
        return full, False
    deny = [t("gateway.exec_approval.text_choice_deny", prefix=prefix)]
    # The whole command fits beside Deny alone but not beside every choice: say so, not "shortened".
    whole = prompt(deny, t("gateway.exec_approval.text_prompt_deny_only") + "\n", command, code_fence(command))
    if measure(whole) <= limit:
        return whole, True
    notice = t("gateway.exec_approval.truncated_deny_only", total=len(command)) + "\n"
    # The fence is sized from the shortened command, so it always closes within the cut; the budget stays
    # under the whole command so the "shortened" notice is never shown beside the full text.
    shown, fence = fit_fenced(command, lambda fence: fit_prefix(command, max(0, min(
        measure(command) - 1, limit - measure(prompt(deny, notice, "", fence)) - measure("..."))), measure))
    return prompt(deny, notice, shown, fence), True
