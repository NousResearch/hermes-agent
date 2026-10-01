"""The minimal environment shared by PM's proxy probe and the proxy daemon."""
from __future__ import annotations

import os

# A native PM tool probe runs before the application venv exists. Keep the
# allowlist here so it never imports configuration/YAML just to scrub env.
_PROXY_SUBPROCESS_ENV_ALLOWLIST = (
    "PATH", "HOME", "TMPDIR", "TZ", "LANG", "LC_ALL", "LC_CTYPE", "NO_COLOR",
    "SSL_CERT_DIR", "SSL_CERT_FILE", "SYSTEMROOT", "USERPROFILE",
)


def allowlisted_env() -> dict[str, str]:
    """Infrastructure-only env; never pass the operator's credentials."""
    return {name: os.environ[name] for name in _PROXY_SUBPROCESS_ENV_ALLOWLIST if name in os.environ}
