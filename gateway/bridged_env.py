"""Ownership markers of the YAML -> env bridge (``gateway.platforms._shared.yaml_env_setter``).

Stdlib only: the subprocess env builders (``tools.environments.local``) read these on the
post-update restart path, which must import nothing heavy (``hermes_cli.update_cmd_commit``).
"""

from __future__ import annotations

import hashlib
from typing import Optional

BRIDGED_ENV_MARKER_PREFIX = "HERMES_BRIDGED_"


def bridged_env_digest(value: Optional[str]) -> Optional[str]:
    """What a marker stores for *value* (None stays None, so an unset name never matches)."""
    return None if value is None else "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
