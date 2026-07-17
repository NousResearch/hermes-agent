"""Pending-review persistence for the pr_review_buttons plugin.

The cron review sweep *stages* each PR's verbatim review body here; the Slack
button click reads it back and posts the exact same text to GitHub. Storing the
body server-side (rather than encoding it into the button) is deliberate: Slack
caps a button ``value`` at 2000 chars, far smaller than a real review.

State file: ``~/.hermes/state/pr_review_pending.json``

    {
      "Strike48/matrix#42": {
        "repo": "Strike48/matrix", "number": 42, "head_sha": "abc123",
        "title": "...", "url": "...", "verdict": "Needs work",
        "body": "## Hermes Review\n\n...",
        "message_ts": "1700.1"   # set once posted to Slack
      }
    }

Writes are atomic (temp file + ``os.replace``) so a concurrent reader never
sees a truncated file. Pure functions of an explicit ``path`` argument — no
global state — so the store is trivially unit-testable.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# The fields the plugin persists. Anything else in a staged entry is dropped so
# a malformed cron payload can't bloat the file.
_FIELDS = ("repo", "number", "head_sha", "title", "url", "verdict", "body")


def default_path() -> Path:
    """Resolve the on-disk state file, honouring ``HERMES_HOME``."""
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return Path(home) / "state" / "pr_review_pending.json"


def key_for(repo: str, number: Any) -> str:
    """Build the canonical store key: ``owner/repo#N``."""
    return f"{repo}#{number}"


def load(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return the full store, or ``{}`` when the file is missing/unreadable."""
    try:
        raw = Path(path).read_text()
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: Dict[str, Dict[str, Any]]) -> None:
    """Atomically write ``data`` to ``path`` (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pr_review_pending.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _clean(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {k: entry.get(k) for k in _FIELDS}


def stage(entry: Dict[str, Any], *, path: Optional[Path] = None) -> str:
    """Persist one PR's review, keyed ``repo#number``. Returns the key.

    Re-staging the same key overwrites in place (a re-review at a new head SHA),
    never duplicates. A freshly staged entry has no ``message_ts`` yet — that is
    set by :func:`mark_published` once the Slack digest is posted.
    """
    path = path or default_path()
    data = load(path)
    key = key_for(entry["repo"], entry["number"])
    data[key] = _clean(entry)
    _write(path, data)
    return key


def get(key: str, *, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the staged entry for ``key`` or ``None``."""
    return load(path or default_path()).get(key)


def pop(key: str, *, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Atomically remove and return the entry for ``key``.

    Returns ``None`` if it was already gone — this is the double-click guard:
    the first click pops the entry and posts; a second click finds nothing.
    """
    path = path or default_path()
    data = load(path)
    entry = data.pop(key, None)
    if entry is not None:
        _write(path, data)
    return entry


def mark_published(key: str, *, message_ts: str, path: Optional[Path] = None) -> None:
    """Record the Slack message ts for a staged entry (marks it as posted)."""
    path = path or default_path()
    data = load(path)
    if key in data:
        data[key]["message_ts"] = message_ts
        _write(path, data)


def unpublished(*, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return staged entries that have not yet been posted to Slack.

    Each returned dict is a copy carrying an extra ``_key`` field so callers can
    build buttons without recomputing the key.
    """
    data = load(path or default_path())
    out: List[Dict[str, Any]] = []
    for key, entry in data.items():
        if entry.get("message_ts"):
            continue
        out.append(dict(entry, _key=key))
    return out
