"""_relative_time must tolerate legacy ISO-8601 timestamp strings.

SessionDB normalizes timestamps at the source, but rows written by older
builds can reach the formatter before the migration has run; the CLI list
must degrade gracefully instead of raising TypeError.
"""

import time
from datetime import datetime, timezone

from hermes_cli.main import _relative_time


def test_relative_time_accepts_iso_timestamps():
    ts = time.time() - 7200
    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    assert _relative_time(iso) == "2h ago"


def test_relative_time_accepts_epoch_strings():
    ts = time.time() - 120
    assert _relative_time(str(ts)) == "2m ago"


def test_relative_time_unparseable_returns_placeholder():
    assert _relative_time("not-a-timestamp") == "?"
    assert _relative_time(None) == "?"
