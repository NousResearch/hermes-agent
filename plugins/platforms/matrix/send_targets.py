"""send_message declarations for the Matrix ``PlatformEntry`` (kept out of the adapter module)."""


def _parse_send_target(ref):
    # "<room>:$<event_id>" addresses a thread (rfind: room ids contain ':'). Bare "!room" /
    # "@user" go via the generic rule so the numeric check keeps precedence.
    trimmed = ref.strip()
    split_idx = trimmed.rfind(":$")
    return (trimmed[:split_idx], trimmed[split_idx + 1:]) if split_idx > 0 else None


SEND_ENTRY_KWARGS = {"target_ref_syntax_fn": _parse_send_target}
