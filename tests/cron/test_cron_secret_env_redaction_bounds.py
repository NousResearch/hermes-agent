"""Job secret redaction is one pass over the original text, with a bounded output.

Replacing value after value let a later (short) value match inside an earlier replacement token:
a one-character value such as ``E`` rewrote every ``E`` of ``[REDACTED-SECRET-ENV]``, and each
further value multiplied the output again. Every check here reports counts and lengths only, so a
failing run never prints a value.
"""

from __future__ import annotations

import random

import pytest

from cron.secret_env import SECRET_ENV_REDACTION, scrub_secret_values

TOKEN = SECRET_ENV_REDACTION
# Synthetic test values, never real credentials.
LONG_VALUE = "FAKE-LONG-SECRET-VALUE-0123456789"


def _bound(n: int) -> int:
    """Each token stands for >= 1 hidden character and tokens are never adjacent."""
    return n + (len(TOKEN) - 1) * ((n + 1) // 2)


def _leaked(out: str, values) -> int:
    """Values still readable in the text the scrub kept (the spans between tokens)."""
    return sum(value in kept for kept in out.split(TOKEN) for value in values)


def test_replacement_token_is_never_rescanned_by_later_values():
    values = [LONG_VALUE, "E", "D", "[", "-", "T"]
    out = scrub_secret_values(f"start {LONG_VALUE} end", values)
    assert out == f"start {TOKEN} end"


@pytest.mark.parametrize("count", [1, 4, 8])
def test_one_character_values_cannot_grow_the_output_past_the_bound(count):
    values = list("[REDACT-SV]")[:count]
    rng = random.Random(count)
    text = "".join(rng.choice("[REDACT-SV] xyz\n") for _ in range(5000))
    out = scrub_secret_values(text, values)
    assert len(out) <= _bound(len(text))
    assert _leaked(out, values) == 0


def test_overlapping_values_hide_their_union():
    out = scrub_secret_values("xx-abcdef-yy", ["abcd", "cdef"])
    assert out == f"xx-{TOKEN}-yy"


def test_a_value_inside_another_and_on_its_own_is_hidden_both_ways():
    out = scrub_secret_values("abc and b", ["abc", "b"])
    assert out == f"{TOKEN} and {TOKEN}"


def test_adjacent_occurrences_collapse_into_one_token():
    assert scrub_secret_values("<aaaa>", ["a"]) == f"<{TOKEN}>"
    assert scrub_secret_values("<abcd>", ["ab", "cd"]) == f"<{TOKEN}>"


def test_values_that_overlap_the_token_text_leave_it_intact():
    text = f"before {TOKEN} after REDACTED"
    out = scrub_secret_values(text, ["REDACTED", "SECRET-ENV]"])
    assert _leaked(out, ["REDACTED", "SECRET-ENV]"]) == 0
    assert out.startswith("before ") and len(out) <= _bound(len(text))


def test_empty_and_whitespace_values_change_nothing():
    assert scrub_secret_values("", ["x"]) == ""
    assert scrub_secret_values("plain output", ["", "  ", "\n"]) == "plain output"
    assert scrub_secret_values("plain output", []) == "plain output"


def test_multiline_value_still_matches_any_line_break_run():
    out = scrub_secret_values("a FIRST\r\n\nSECOND b", ["FIRST\r\nSECOND", "S"])
    assert out == f"a {TOKEN} b"


@pytest.mark.parametrize("seed", range(40))
def test_random_values_never_leak_and_stay_within_the_bound(seed):
    rng = random.Random(seed)
    alphabet = "abcE-[] "
    values = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))
              for _ in range(rng.randint(1, 8))]
    text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 400)))
    out = scrub_secret_values(text, values)
    assert len(out) <= _bound(len(text))
    assert _leaked(out, [v for v in values if v.strip()]) == 0


def test_large_output_with_many_short_values_is_linear():
    values = list("[REDACT-]")
    text = "[REDACTED-SECRET-ENV] " * 20000
    out = scrub_secret_values(text, values)
    assert len(out) <= _bound(len(text))
    assert _leaked(out, values) == 0


_TWO_VALUES_SCRIPT = """\
import os
print("long=" + os.environ["CORP_TEST_LONG"] + " short=" + os.environ["CORP_TEST_SHORT"] + " DONE")
"""


@pytest.mark.platforms("windows")
def test_run_job_scrubs_a_one_character_value_without_chaining(tmp_path, monkeypatch, caplog):
    """Real ``run_job`` with an in-memory backend: a long value and a one-character value that
    occurs inside the replacement token are both hidden, and the token is not rewritten."""
    import hashlib

    from cron import secret_env
    from cron.jobs import create_job
    from cron.scheduler import run_job

    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    values = {"t/long": LONG_VALUE, "t/short": "E"}
    monkeypatch.setattr(secret_env, "_default_backends",
                        lambda: {"wincred": type("B", (), {"resolve": lambda self, t: values[t]})()})
    script = home / "scripts" / "two.py"
    script.write_text(_TWO_VALUES_SCRIPT, encoding="utf-8")
    job = create_job(
        prompt=None, schedule="every 5m", script="two.py", no_agent=True, deliver="local",
        secret_env={"CORP_TEST_LONG": "wincred://t/long", "CORP_TEST_SHORT": "wincred://t/short"},
        secret_env_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())

    success, _doc, final_response, error = run_job(job)

    assert success is True and error is None
    assert f"long={TOKEN} short={TOKEN} DON{TOKEN}" in final_response
    assert _leaked(final_response, [LONG_VALUE]) == 0 and LONG_VALUE not in caplog.text
