"""Once-per-profile memory for the Codex gpt-5.5 compression threshold autoraise notice.

Split out of ``agent/agent_init.py``, which shows the notice.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from hermes_constants import get_hermes_home


def _codex_gpt55_autoraise_notice_marker():
    """Per-profile marker path (``$HERMES_HOME`` is profile-scoped; not a config key)."""
    return get_hermes_home() / ".codex_gpt55_autoraise_notice"


def _codex_gpt55_autoraise_notice_state(autoraise: dict[str, Any]) -> str:
    """Notice identity keyed on what it displays (model + from→to percentages).

    An unchanged threshold stays silent across restarts; a changed global threshold or a
    different autoraised Codex model re-notifies once.
    """
    model = str(autoraise.get("model") or "").strip().lower().rsplit("/", 1)[-1]
    from_pct = round(float(autoraise["from"]) * 100)
    to_pct = round(float(autoraise["to"]) * 100)
    return f"{model}:{from_pct}:{to_pct}"


def _codex_gpt55_autoraise_notice_seen(autoraise: dict[str, Any]) -> bool:
    """True if this exact notice was already shown for this profile (unreadable = unseen)."""
    try:
        current = _codex_gpt55_autoraise_notice_state(autoraise)
        return _codex_gpt55_autoraise_notice_marker().read_text(
            encoding="utf-8"
        ).strip() == current
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _record_codex_gpt55_autoraise_notice(autoraise: dict[str, Any]) -> None:
    """Persist that the notice was shown. Best-effort: a failure only re-shows it later."""
    with suppress(OSError, KeyError, TypeError, ValueError):
        marker = _codex_gpt55_autoraise_notice_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(_codex_gpt55_autoraise_notice_state(autoraise), encoding="utf-8")
