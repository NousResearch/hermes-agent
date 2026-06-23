"""Regression test for hardened HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS parsing.

Salvage of #13873 by @Junass1 (remaining gap): three of the four timeout knobs
that PR hardened already route through ``gateway.run._float_env`` on current
main, but the Telegram follow-up grace window was still parsed with a raw
``float(os.getenv(...))``.  A config/env typo (e.g.
``HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS=abc``) therefore raised ``ValueError``
inside the inbound Telegram message handler instead of falling back to the
documented 3.0s default.  This test pins the helper's behaviour so the knob
can never crash message handling again.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gateway.run import _float_env


def test_followup_grace_valid_value(monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "5.5")
    assert _float_env("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", 3.0) == 5.5


def test_followup_grace_invalid_value_falls_back(monkeypatch):
    # The exact typo from #13873 must NOT raise — it falls back to the default.
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "abc")
    assert _float_env("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", 3.0) == 3.0


def test_followup_grace_empty_falls_back(monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "")
    assert _float_env("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", 3.0) == 3.0


def test_followup_grace_unset_falls_back(monkeypatch):
    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)
    assert _float_env("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", 3.0) == 3.0
