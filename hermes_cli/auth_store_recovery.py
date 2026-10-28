"""Corrupt auth-store recovery, subject to the process credential policy."""

from pathlib import Path
from typing import Any


class _ReadOnlyAuthStoreError(RuntimeError):
    """A corrupt readonly store must not degrade into credential recovery or fallback."""


def _recover_corrupt_auth_store(
    auth_file: Path, error: Exception, *, read_only: bool = False,
) -> dict[str, Any]:
    from agent.codex_auth_policy import codex_auth_read_only
    # Late-bind the facade so existing logger/copy patch seams still intercept recovery.
    from hermes_cli.auth import _empty_auth_store, logger, shutil

    # The store may contain Codex credentials. Refuse before the best-effort copy handler:
    # swallowing a write refusal would both hide a forbidden attempt and permit pool fallback.
    if read_only or codex_auth_read_only():
        # Parser exceptions can contain credential fragments; do not expose their context.
        raise _ReadOnlyAuthStoreError(
            "Cannot read corrupt auth store with --codex-auth-read-only") from None
    # Genuine JSON/Unicode corruption retains the legacy best-effort backup contract.
    # Never advertise preservation unless the copy actually completed.
    corrupt_path = auth_file.with_suffix(".json.corrupt")
    try:
        shutil.copy2(auth_file, corrupt_path)
        preserved = True
    except Exception:  # health: allow BLE001 -- legacy backup failures of any kind must preserve empty-store recovery
        preserved = False
        logger.debug("auth: could not preserve a copy of the corrupt store at %s", corrupt_path,
                     exc_info=True)
    logger.warning(
        "auth: failed to parse %s (%s), starting with empty store. %s %s",
        auth_file, error,
        "Corrupt file preserved at" if preserved else "A copy could NOT be preserved at",
        corrupt_path)
    return _empty_auth_store()
