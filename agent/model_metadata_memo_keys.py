"""Which credential a memoized ``/models`` catalog belongs to, in process and on disk.

An endpoint may answer ``/models`` differently, or not at all (401/403), depending on the key, so a
catalog fetched with one credential must never answer a lookup made with another.
"""

from agent.credential_persistence import fingerprint_secret_value

_MINTED = "minted:"


def endpoint_memo_key(normalized: str, api_key: object) -> tuple[str, str]:
    """In-process memo key: the URL plus a fingerprint of a string key, or the identity of a minted one."""
    if api_key is None or isinstance(api_key, str):
        return normalized, fingerprint_secret_value(api_key) or ""
    # A minted key (key_cmd) is not fingerprinted (that would mint on every cache hit), but it needs a slot
    # of its own: in the keyless slot a keyless 401 memoized as {} would answer it without a request.
    return normalized, f"{_MINTED}{id(api_key):x}"


def endpoint_disk_slot(memo_key: tuple[str, str]) -> str:
    """Disk memo key: keyless probes keep the bare URL, keyed ones get their own entry.

    Minted keys share one cross-process slot, since their in-process identity means nothing on disk.
    """
    normalized, slot = memo_key
    if not slot:
        return normalized
    return f"{normalized} [{'minted' if slot.startswith(_MINTED) else slot}]"
