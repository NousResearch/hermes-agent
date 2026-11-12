"""Public send_message contract pieces a platform plugin declares on its ``PlatformEntry``.

Plugins import these instead of core-private helpers in ``tools/send_message_*``:

* ``target_ref_syntax_fn`` parsers — ``(target_ref) -> (chat_id, thread_id) | None | NOT_EXPLICIT``.
  None falls through to core's generic id rules (E.164, numeric, Matrix ``!``/``@``); ``NOT_EXPLICIT``
  skips them and goes straight to the channel directory.
* ``StandaloneSendPolicy`` — when send_message bypasses the live adapter for ``standalone_sender_fn``.

Pure ``re``/dataclasses: core imports this module before any adapter is materialized.
"""

from dataclasses import dataclass

NOT_EXPLICIT = object()  # sentinel: stop parsing, target is NOT explicit (skip generic rules)


def regex_groups_parser(regex, *, thread_group=True):
    """Explicit when ``regex`` fully matches: chat_id = group 1, thread = group 2 (or None)."""
    return lambda ref: ((m.group(1), m.group(2) if thread_group else None)
                        if (m := regex.fullmatch(ref)) else None)


def regex_stripped_parser(regex):
    """Explicit when ``regex`` fully matches; returns the stripped ref verbatim."""
    return lambda ref: (ref.strip(), None) if regex.fullmatch(ref) else None


def parse_nonempty(ref):
    # ntfy topics and WeCom ids (the adapter picks the send command) are explicit when non-empty.
    stripped = ref.strip()
    return (stripped, None) if stripped else None


@dataclass(frozen=True)
class StandaloneSendPolicy:
    """How send_message routes a platform through its ``standalone_sender_fn`` instead of the
    live adapter (Discord: forums/threads/multipart; Slack: files_upload_v2; WhatsApp: Baileys
    /send-media; Signal: attachments ride the JSON-RPC param on the final chunk)."""

    label: str = ""  # names the platform in "<label> plugin not registered ..." errors
    media: bool = False  # sends that carry media bypass the live adapter
    all_sends: bool = False  # every send bypasses it, text included (Discord)
    captionable: bool = False  # one captionable file + short text ride as the media caption
    # Non-final chunks get ``media_files=[]`` (True) or ``media_files=None`` (False).
    empty_list_between_chunks: bool = False
    forward_force_document: bool = False  # pass ``force_document`` through to the sender
    # ``mentions`` (re-pinged per message, so only the first payload carries them) force this route.
    mentions: bool = False
    # Generic text-only path: call the sender per chunk directly, skipping the live adapter and
    # any ``send_message_handler``.
    text_direct: bool = False
