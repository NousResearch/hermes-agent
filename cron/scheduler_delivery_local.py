"""Cron delivery into a local (Desktop/Webapp) chat (``deliver: origin`` from a chat the app created).

A local chat has no push transport and no attachment lane: the result is admitted into the chat's
history as a note (``gateway/session_cron.py::deliver_to_local_chat``), so the note IS the delivery.
It keeps one ``MEDIA:<path>`` line per file that passed the media path policy, and names the job in
a card (``display_text``) the app paints as a collapsible row whose report renders that media as an
assistant reply's would. Before, the lane got the tag-stripped text only and every file vanished
while the run reported a clean delivery (#148449).
"""

from __future__ import annotations

from typing import Optional

# The card's display_text is capped at 2000 characters (``automation_display_metadata``).
_TITLE_NAME_CAP = 200


def deliver_local_note(job: dict, chat_id: str, text: str, media_files: list, *,
                       for_failure: bool) -> Optional[str]:
    """Admit the delivery into local chat *chat_id*; None once admitted, else the error.
    *text* is the redacted, tag-stripped delivery; *media_files* the policy-checked
    ``(path, is_voice)`` pairs ``extract_media`` found in it."""
    from gateway.session_cron import deliver_to_local_chat
    from hermes_constants import get_hermes_home
    return deliver_to_local_chat(get_hermes_home(), chat_id, note_text(text, media_files),
                                 diagnostic=for_failure, title=note_title(job, for_failure))


def note_text(text: str, media_files: list) -> str:
    """The delivered text plus one ``MEDIA:<path>`` line per file, last, as the auto-appended media
    of a gateway reply go (``run_turn_runner._append_auto_media_tags``)."""
    if not media_files:
        return text
    return "\n".join([text, "", *(f"MEDIA:{path}" for path, _is_voice in media_files)]).strip()


def note_title(job: dict, for_failure: bool) -> str:
    """The card's label: which job, and whether it finished or failed."""
    name = str(job.get("name") or job.get("id") or "")[:_TITLE_NAME_CAP]
    return f"Cron job '{name}' {'failed' if for_failure else 'finished'}"
