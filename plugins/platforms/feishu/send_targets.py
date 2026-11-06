"""send_message declarations for the Feishu ``PlatformEntry`` (kept out of the adapter module)."""

import re

from gateway.platforms.send_targets import StandaloneSendPolicy, regex_groups_parser

_SEND_TARGET_RE = re.compile(r"^\s*((?:oc|ou|on|chat|open)_[-A-Za-z0-9]+)(?::([-A-Za-z0-9_]+))?\s*$")

SEND_ENTRY_KWARGS = {
    "target_ref_syntax_fn": regex_groups_parser(_SEND_TARGET_RE),
    "standalone_send": StandaloneSendPolicy(label="Feishu", media=True, text_direct=True),
}
