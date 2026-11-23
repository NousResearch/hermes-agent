"""The dashboard session token: where its value comes from.

The desktop shell mints the token via HERMES_DASHBOARD_SESSION_TOKEN; a
"file:<path>" value instead names a file to read it from (the Nix modules
render backend.sessionTokenFile this way, so a path -- not a secret -- sits
in the service environment). With no variable, a fresh token is minted per
server start. It dies with the process and is injected into the SPA HTML so
only the web UI can use it.
"""

import os
import secrets
from pathlib import Path


def resolve_session_token() -> str:
    env_token = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN")
    if not env_token:
        return secrets.token_urlsafe(32)
    # A "file:" prefix means the value is a token-file path, not a token:
    # deployment managers (agenix, sops-nix, systemd credentials) hand a
    # secret over as a file, and a path -- unlike the token -- can sit in a
    # systemd Environment= or a launchd plist without putting anything
    # readable into a store path.
    if env_token.startswith("file:"):
        token_file = env_token[len("file:") :]
        try:
            token = Path(token_file).read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError) as exc:
            raise SystemExit(
                f"HERMES_DASHBOARD_SESSION_TOKEN names an unreadable token file ({token_file}): {exc}"
            ) from exc
        if not token:
            raise SystemExit(
                f"HERMES_DASHBOARD_SESSION_TOKEN names an empty token file ({token_file})"
            )
        return token
    return env_token
