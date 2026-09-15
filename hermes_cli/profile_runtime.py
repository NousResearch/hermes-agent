"""Where a NEW profile's gateway will be supervised: the container's supervisor, or this
machine's service manager.

``hermes profile create`` cannot answer this by asking itself. ``is_container()`` and
``detect_service_manager()`` describe the CREATING process, and the interesting case is
precisely the one where they disagree with reality: a host whose ``~/.hermes`` is bind-mounted
into a container creates the profile directory exactly where the container reads it, while
being unable to see ``/run/service`` at all. So resolve it from what the SERVING gateway
declares about itself — ``runtime_kind`` in the active home's ``gateway_state.json`` — rather
than from what this process happens to be.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, Tuple

RuntimeKind = Literal["container", "native"]

RUNTIME_VALUES: tuple[str, ...] = ("auto", "container", "native")

_STATE_FILE = "gateway_state.json"


def _declared_runtime_kind(home: Path) -> Optional[str]:
    """``runtime_kind`` last stamped into ``home``'s gateway_state.json, or None.

    Deliberately does NOT check liveness: the question is which supervisor OWNS this profile,
    which a stopped gateway answers just as well as a running one.
    """
    try:
        data = json.loads((home / _STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    kind = data.get("runtime_kind")
    return kind if kind in ("container", "native") else None


def resolve_profile_runtime(explicit: Optional[str] = None) -> Tuple[RuntimeKind, str]:
    """Resolve to ``("container"|"native", human reason)``.

    Precedence: the ``--runtime`` flag, then ``profiles.runtime``, then the serving gateway's
    own declaration. ``auto`` resolves to ``native`` whenever nothing containerized has stamped
    the active home — which is byte-for-byte today's behaviour for every plain host install and
    every gateway older than the ``runtime_kind`` field.

    Raises ValueError naming the three choices for an unrecognised value: a typo must not
    silently pick a runtime.
    """
    from hermes_cli.config import load_config
    from hermes_constants import get_hermes_home

    choice = explicit
    source = "--runtime"
    if choice is None:
        try:
            configured = (load_config().get("profiles") or {}).get("runtime")
        except Exception:
            configured = None
        choice = configured if configured is not None else "auto"
        source = "profiles.runtime"

    if not isinstance(choice, str) or choice not in RUNTIME_VALUES:
        raise ValueError(
            f"invalid {source} value {choice!r}: choose one of "
            + ", ".join(repr(v) for v in RUNTIME_VALUES)
        )

    if choice != "auto":
        return choice, f"{source} = {choice}"  # type: ignore[return-value]

    home = get_hermes_home()
    declared = _declared_runtime_kind(home)
    if declared is None:
        return "native", "auto — no gateway here declares a container runtime"
    where = "in a container" if declared == "container" else "natively"
    return declared, f"auto — the gateway serving {home.name} declares it runs {where}"


def container_name() -> str:
    """``profiles.container_name`` — used only to render the ``docker exec <name> ...`` hand-off
    command. Never shelled out to, so a wrong value costs a wrong hint, not a wrong action."""
    from hermes_cli.config import load_config
    try:
        configured = (load_config().get("profiles") or {}).get("container_name")
    except Exception:
        configured = None
    return configured if isinstance(configured, str) and configured.strip() else "hermes"
