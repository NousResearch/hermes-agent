"""send_message declarations for the Slack ``PlatformEntry`` (kept out of the adapter module)."""

import re

from gateway.platforms.send_targets import StandaloneSendPolicy

# Slack conversation IDs: C (public), G (private/group), D (DM); uppercase alnum, 9+ chars. User IDs
# (U...) become ``user:U...`` and are opened as D... conversations first (posting straight to a U/W
# id fails); ``@handle`` -> ``user_name:...`` resolves via users.list.
_SEND_TARGET_RE = re.compile(r"^\s*([CGD][A-Z0-9]{8,})\s*$")
_SEND_USER_ID_RE = re.compile(r"^\s*(U[A-Z0-9]{8,})\s*$")
_SEND_USER_NAME_RE = re.compile(r"^\s*@([A-Za-z0-9._-]{1,80})\s*$")
_SEND_MENTION_RE = re.compile(r"^\s*<@(U[A-Z0-9]{8,})(?:\|[^>]+)?>\s*$")
# Session-derived Slack thread targets use "<conversation_id>:<thread_ts>".
_SEND_THREAD_TARGET_RE = re.compile(r"^\s*([CGD][A-Z0-9]{8,}):([^\s:]+)\s*$")
# (regex, chat_id template, thread comes from group 2) — thread form before bare id.
_SEND_FORMS = ((_SEND_THREAD_TARGET_RE, "{}", True), (_SEND_TARGET_RE, "{}", False),
               (_SEND_USER_ID_RE, "user:{}", False), (_SEND_MENTION_RE, "user:{}", False),
               (_SEND_USER_NAME_RE, "user_name:{}", False))


def _parse_send_target(ref):
    return next(((template.format(m.group(1)), m.group(2) if has_thread else None)
                 for regex, template, has_thread in _SEND_FORMS if (m := regex.fullmatch(ref))), None)


SEND_ENTRY_KWARGS = {
    "target_ref_syntax_fn": _parse_send_target,
    # Media goes through files_upload_v2 in the standalone sender.
    "standalone_send": StandaloneSendPolicy(label="Slack", media=True, captionable=True,
                                            empty_list_between_chunks=True),
}
