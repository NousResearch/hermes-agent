"""What identifies a WhatsApp activation: saved policy plus a completed pairing generation.

The gateway compares signatures to decide whether a running adapter must be replaced after
``hermes whatsapp`` or the dashboard saved a new pairing or policy."""

import hashlib
import secrets
from pathlib import Path


def whatsapp_activation_signature(home: Path, session_path: Path) -> tuple:
    """Identify saved policy and completed pairing without tracking mutable Baileys credentials."""
    from gateway.platforms._shared import get_scoped_secret

    signature = []
    for path in (Path(home) / ".env", Path(session_path) / ".hermes-pairing-generation"):
        try:
            signature.append(hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            signature.append(None)
    for key in ("WHATSAPP_MODE", "WHATSAPP_DM_POLICY", "WHATSAPP_ALLOWED_USERS",
                "WHATSAPP_ALLOW_FROM", "WHATSAPP_GROUP_POLICY", "WHATSAPP_GROUP_ALLOWED_USERS",
                "WHATSAPP_GROUP_ALLOW_FROM"):
        signature.append(get_scoped_secret(key, ""))
    return tuple(signature)


def stamp_whatsapp_pairing(session_path: Path) -> None:
    """Advance identity only after a new QR pairing completed successfully."""
    (Path(session_path) / ".hermes-pairing-generation").write_text(
        secrets.token_hex(16), encoding="utf-8")
