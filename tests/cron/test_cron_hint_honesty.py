"""The cron delivery hint tells the model a failed or unverified check is never "nothing new".

Without it the [SILENT] instruction let monitoring runs answer [SILENT] when a source could not
be reached, so the user never heard that a check failed (weak-model replay: 12.5% of runs with a
planted failure answered [SILENT]; 5.0% with the clause, and healthy runs still went silent).
"""

from cron.scheduler_prompt import _CRON_HINT


def test_cron_hint_says_a_failed_check_is_reported_not_silent():
    silent = _CRON_HINT.index("SILENT:")
    honesty = _CRON_HINT.index("HONESTY:")
    assert silent < honesty < _CRON_HINT.index("RECURSION:")
    clause = _CRON_HINT[honesty:_CRON_HINT.index("RECURSION:")]
    assert "could not be verified" in clause and "instead of answering [SILENT]" in clause
