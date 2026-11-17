"""Files the media-delivery gate never attaches because of what they hold.

Project-local secret env files (``.env``, ``.envrc``, ...) are read-blocked anywhere on disk by
``agent.file_safety.get_read_block_error``; the gate mirrors that (the denylist's stated invariant)
in every mode and location, judged on the resolved name so a symlink to a ``.env`` is caught.

``gateway.refuse_secret_files: true`` makes ``validate_media_delivery_path`` (the one gate every
native delivery passes: ``MEDIA:`` tags, bare local paths, ``send_message`` attachments, cron and
``hermes send``) drop a file whose bytes contain a value registered by
``agent.redact_env_values.registered_secret_forms``: the profile's ``.env`` secrets and its applied
secret-source values.

This is accident prevention, not a security boundary: the agent can still read the file with its
terminal and paste its contents (which the chat-egress redactor then masks). It stops the common
slip of attaching a config dump, a debug log or a copied ``.env`` that carries a live key.
Files larger than ``_MAX_SCAN_BYTES`` are not scanned.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Bound at import, never at call time: an unavailable predicate must deny, not skip the check.
from agent.file_safety import is_project_env_basename

logger = logging.getLogger(__name__)

_MAX_SCAN_BYTES = 16 * 1024 * 1024  # ~40 ms worst case on the delivery path, once per (path, mtime, size)
_CHUNK = 1024 * 1024
# (path, mtime_ns, size, forms) -> verdict: one reply checks the same path several times.
_VERDICTS: dict[tuple, bool] = {}
_VERDICTS_MAX = 256


def refuse_secret_files_enabled() -> bool:
    """``gateway.refuse_secret_files`` of the active profile; anything but ``true`` keeps it off."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.config_backend import ConfigBackendUnavailable

    try:
        gateway_cfg = load_config_readonly().get("gateway")
    # A refused backend read is a SystemExit subclass: it must not abort the delivery being judged.
    except (Exception, ConfigBackendUnavailable):  # health: allow BLE001 -- unreadable config keeps the legacy (off) default
        return False
    return isinstance(gateway_cfg, dict) and gateway_cfg.get("refuse_secret_files") is True


def _holds_any(path: Path, needles: tuple[bytes, ...]) -> bool:
    overlap = max(map(len, needles)) - 1
    tail = b""
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            window = tail + chunk
            if any(n in window for n in needles):
                return True
            tail = window[-overlap:] if overlap else b""
    return False


def file_holds_registered_secret(path: Path) -> bool:
    """True when ``path``'s bytes contain one of the active profile's registered secret values."""
    from agent.redact_env_values import registered_secret_forms

    forms = registered_secret_forms()
    if not forms:
        return False
    try:
        st = os.stat(path)
    except OSError:
        return False
    if st.st_size > _MAX_SCAN_BYTES:
        return False
    key = (str(path), st.st_mtime_ns, st.st_size, forms)
    verdict = _VERDICTS.get(key)
    if verdict is None:
        try:
            verdict = _holds_any(path, tuple(dict.fromkeys(f.encode("utf-8") for f in forms)))
        except OSError:
            return False  # unreadable here is unreadable for the upload too
        if len(_VERDICTS) >= _VERDICTS_MAX:
            _VERDICTS.clear()
        _VERDICTS[key] = verdict
    return verdict


def is_undeliverable_secret_file(path: Path) -> bool:
    """The delivery gate's content check on a resolved path: a project env file always; with the
    opt-in, a file holding a registered secret value (logged by name, never the value)."""
    if is_project_env_basename(path.name):
        return True
    if not refuse_secret_files_enabled() or not file_holds_registered_secret(path):
        return False
    logger.warning("Not delivering %s: it contains a secret value from this profile's .env or secret "
                   "sources (gateway.refuse_secret_files)", path.name)
    return True
