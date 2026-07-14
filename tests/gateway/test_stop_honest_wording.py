"""AC4: /stop's user-facing string must be honest present-progressive wording.

The motivating incident (2026-07-14): /stop said "⚡ Stopped." (past-tense,
completed) while the turn was still running for 4.5 minutes. The gateway can
only set a COOPERATIVE interrupt flag; it cannot guarantee the turn has ceased
at the moment /stop returns. The string must therefore not assert a completed
stop -- it says the stop is in progress ("Stopping — finishing the current
step and halting").

This is a content assertion on the shipped i18n catalog (the string the
gateway `_handle_stop_command` returns via `t("gateway.stop.stopped")`).
"""

from __future__ import annotations

from pathlib import Path

import yaml

LOCALES_DIR = Path(__file__).resolve().parents[1].parent / "locales"


def _stopped_string(lang: str) -> str:
    with (LOCALES_DIR / f"{lang}.yaml").open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["gateway"]["stop"]["stopped"]


def test_english_stop_string_is_present_progressive_not_completed():
    s = _stopped_string("en")
    # Honest: it says it is STOPPING, not that it has STOPPED.
    assert "Stopping" in s, f"expected present-progressive wording, got: {s!r}"
    # Must NOT flatly assert a completed stop ("Stopped." as a full clause).
    assert "Stopped." not in s, (
        f"/stop must not claim the turn has already stopped; got: {s!r}"
    )
    # Still tells the user they can continue.
    assert "continue" in s.lower()


def test_stopped_pending_still_says_stopped():
    # The PENDING case (agent hadn't started) is genuinely stopped-before-start,
    # so its wording legitimately stays past-tense -- only the ACTIVE-turn
    # `stopped` string changes. This guards against an over-broad edit.
    with (LOCALES_DIR / "en.yaml").open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pending = data["gateway"]["stop"]["stopped_pending"]
    assert "Stopped" in pending
