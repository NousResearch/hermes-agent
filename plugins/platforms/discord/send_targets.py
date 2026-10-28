"""send_message declarations for the Discord ``PlatformEntry`` (kept out of the adapter module)."""

import re

from gateway.platforms.send_targets import StandaloneSendPolicy, regex_groups_parser

# "<channel>[:<thread>]" snowflakes: numeric, same shape as Telegram's "<id>[:<topic>]".
_SEND_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")

SEND_ENTRY_KWARGS = {
    "target_ref_syntax_fn": regex_groups_parser(_SEND_TARGET_RE),
    # Every send (text too) bypasses the live adapter: forums/threads/multipart via REST.
    "standalone_send": StandaloneSendPolicy(label="Discord", media=True, all_sends=True, captionable=True,
                                            empty_list_between_chunks=True),
}
