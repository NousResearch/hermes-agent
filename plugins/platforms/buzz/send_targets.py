"""send_message declarations for the Buzz ``PlatformEntry`` (kept out of the adapter module)."""

import re

from gateway.platforms.send_targets import regex_stripped_parser

# Buzz UUIDs are native targets the adapter accepts verbatim — explicit, never home-channel.
_SEND_UUID_RE = re.compile(r"^\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\s*$", re.IGNORECASE)

SEND_ENTRY_KWARGS = {"target_ref_syntax_fn": regex_stripped_parser(_SEND_UUID_RE)}
