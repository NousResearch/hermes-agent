"""Persisted per-chat profile bindings (Tier-2 manual control, salvaged from #24914).

A `/profile <name>` binding is a deliberate user action, so it overrides the
config routing table for that chat.  Bindings live in a small JSON file under
the sessions dir and survive restarts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from gateway.session import SessionSource

# A leading @<profile> followed by a message body (profile names are
# [a-z0-9][a-z0-9_-]*).  Requires a body so a bare "@name" isn't a route.
_MENTION_RE = re.compile(r"^@([a-z0-9][a-z0-9_-]*)\s+(\S.*)$", re.IGNORECASE | re.DOTALL)


def parse_profile_mention(text: str) -> tuple[str | None, str]:
    """Split a leading ``@profile`` override off *text*.

    Returns ``(profile, body)`` for ``@coder do x`` → ``("coder", "do x")``,
    else ``(None, text)`` unchanged.  The caller still gates on whether the
    named profile exists before routing.
    """
    m = _MENTION_RE.match(text or "")
    return (m.group(1), m.group(2)) if m else (None, text)


def chat_binding_key(source: SessionSource) -> str:
    """Stable per-chat key (platform + chat + thread)."""
    return f"{source.platform.value}:{source.chat_id or ''}:{source.thread_id or ''}"


class ChatBindings:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        try:
            self._data = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, profile: str) -> None:
        self._data[key] = profile
        self._save()

    def clear(self, key: str) -> None:
        self._data.pop(key, None)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))
