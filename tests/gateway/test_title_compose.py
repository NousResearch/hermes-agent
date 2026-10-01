"""Exact-string tests for the pure group-title composer (spec §2.4)."""

from __future__ import annotations

import pytest

from gateway.title_compose import (
    MAX_TITLE_LENGTH,
    REASONING_ORDINALS,
    SEPARATOR,
    compose_group_title,
)
from hermes_constants import VALID_REASONING_EFFORTS

SUBJECT = "Deploy cron pipeline"
LONG_SUBJECT = (
    "Investigate why the ingest job silently truncates the final CSV export when the "
    "upstream payload exceeds the batch window and the retry backoff stacks"
)


def test_separator_is_space_middot_space():
    assert SEPARATOR == " \u00b7 "
    assert len(SEPARATOR) == 3
    assert SEPARATOR.encode("unicode_escape").decode() == " \\xb7 "


# --- §2.4 worked examples, exact strings AND exact lengths ----------------------


@pytest.mark.parametrize(
    "subject, model, reasoning, expected, length",
    [
        (SUBJECT, "glm-5.3", "medium", f"{SUBJECT} \u00b7 glm-5.3 \u00b7 r3", 35),
        (SUBJECT, "glm-5.3", None, f"{SUBJECT} \u00b7 glm-5.3", 30),
        (SUBJECT, "space-bunny-free", None, f"{SUBJECT} \u00b7 space-bunny-free", 39),
        ("", "glm-5.3", "high", "glm-5.3 \u00b7 r4", 12),
        (
            LONG_SUBJECT,
            "glm-5.3",
            "max",
            "Investigate why the ingest job silently truncates the final CSV export when "
            "the upst\u2026 \u00b7 glm-5.3 \u00b7 r6",
            100,
        ),
    ],
)
def test_worked_examples(subject, model, reasoning, expected, length):
    got = compose_group_title(subject, model, reasoning)
    assert got == expected
    assert len(got) == length
    assert len(got) <= MAX_TITLE_LENGTH


# --- §2.2 reasoning ordinals ---------------------------------------------------


@pytest.mark.parametrize("index, effort", list(enumerate(VALID_REASONING_EFFORTS, start=1)))
def test_every_effort_maps_to_documented_ordinal(index, effort):
    assert REASONING_ORDINALS[effort] == index
    assert compose_group_title(SUBJECT, "glm-5.3", effort).endswith(f" \u00b7 r{index}")


def test_ordinal_map_is_exactly_the_documented_scale():
    assert REASONING_ORDINALS == {
        "minimal": 1,
        "low": 2,
        "medium": 3,
        "high": 4,
        "xhigh": 5,
        "max": 6,
        "ultra": 7,
    }
    assert len(REASONING_ORDINALS) == 7


def test_all_seven_efforts_produce_a_tag():
    for effort in VALID_REASONING_EFFORTS:
        assert f" \u00b7 r" in compose_group_title(SUBJECT, "glm-5.3", effort)


@pytest.mark.parametrize("reasoning", [None, "", "bogus", "disabled", "off", "none", "r0"])
def test_unknown_or_absent_reasoning_produces_no_tag(reasoning):
    got = compose_group_title(SUBJECT, "glm-5.3", reasoning)
    assert got == f"{SUBJECT} \u00b7 glm-5.3"
    assert "r0" not in got
    assert not got.endswith(SEPARATOR)


# --- §2.3 model rendering ------------------------------------------------------


def test_provider_prefix_is_stripped():
    assert compose_group_title(SUBJECT, "opencode-go/space-bunny-free", None) == (
        f"{SUBJECT} \u00b7 space-bunny-free"
    )


def test_bare_model_is_lowercased():
    assert compose_group_title(SUBJECT, "GLM-5.3", None) == f"{SUBJECT} \u00b7 glm-5.3"


def test_model_without_any_provider_prefix_is_untouched():
    assert compose_group_title(SUBJECT, "glm-5.3", None) == f"{SUBJECT} \u00b7 glm-5.3"
    assert compose_group_title(SUBJECT, "gpt-5.2-codex", None).endswith("gpt-5.2-codex")


def test_only_the_leading_provider_segment_is_stripped():
    # Card rule 4 renders opencode-go/space-bunny-free as space-bunny-free, so the
    # prefix stripped is whatever provider segment is present, not the literal "provider/".
    assert compose_group_title(SUBJECT, "provider/team/model-x", None).endswith("team/model-x")


def test_bare_provider_prefix_with_empty_remainder_does_not_crash():
    assert compose_group_title(SUBJECT, "provider/", None) == SUBJECT
    assert compose_group_title(SUBJECT, "provider/", "high") == f"{SUBJECT} \u00b7 r4"


def test_model_rendered_once_not_twice():
    assert compose_group_title(SUBJECT, "provider/glm-5.3", None) == f"{SUBJECT} \u00b7 glm-5.3"


# --- §2.4 empty / whitespace subject -------------------------------------------


def test_empty_subject_yields_bare_suffix_without_leading_separator():
    got = compose_group_title("", "glm-5.3", "high")
    assert got == "glm-5.3 \u00b7 r4"
    assert not got.startswith(SEPARATOR)
    assert not got.startswith(" ")


@pytest.mark.parametrize("subject", ["", "   ", "\t", "\n", " \t\n "])
def test_whitespace_only_subject_behaves_as_empty(subject):
    assert compose_group_title(subject, "glm-5.3", "high") == "glm-5.3 \u00b7 r4"


def test_empty_subject_model_and_no_reasoning_is_empty_string():
    assert compose_group_title("", "", None) == ""
    assert compose_group_title("   ", "provider/", "bogus") == ""


def test_subject_only_when_model_and_reasoning_are_empty():
    assert compose_group_title(SUBJECT, "", None) == SUBJECT


# --- truncation ----------------------------------------------------------------


@pytest.mark.parametrize("effort", list(VALID_REASONING_EFFORTS) + [None, "", "bogus"])
def test_truncation_fits_the_cap_at_every_effort_level(effort):
    got = compose_group_title(LONG_SUBJECT, "opencode-go/space-bunny-free", effort)
    assert len(got) <= MAX_TITLE_LENGTH
    ordinal = REASONING_ORDINALS.get(effort or "")
    expected_tail = f"{SEPARATOR}space-bunny-free"
    expected_tail += f" \u00b7 r{ordinal}" if ordinal else ""
    assert got == got[: got.rindex("\u2026")] + "\u2026" + expected_tail
    assert got.endswith("space-bunny-free" + (f" \u00b7 r{ordinal}" if ordinal else ""))
    # rstrip may shorten the result below the cap; never overshoot it
    assert len(got) >= MAX_TITLE_LENGTH - 1


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max", "ultra"])
def test_truncation_is_exactly_100_when_nothing_is_rstripped(effort):
    # No space at the cut point -> the budget is filled exactly.
    subject = "z" * 200
    got = compose_group_title(subject, "glm-5.3", effort)
    assert len(got) == MAX_TITLE_LENGTH
    assert got == subject[: 100 - len(f"{SEPARATOR}glm-5.3 \u00b7 r{REASONING_ORDINALS[effort]}") - 1] \
        + "\u2026" + f"{SEPARATOR}glm-5.3 \u00b7 r{REASONING_ORDINALS[effort]}"


def test_truncation_never_cuts_the_suffix():
    for effort in VALID_REASONING_EFFORTS:
        got = compose_group_title(LONG_SUBJECT, "glm-5.3", effort)
        assert got.endswith(f" \u00b7 glm-5.3 \u00b7 r{REASONING_ORDINALS[effort]}")
    assert compose_group_title(LONG_SUBJECT, "glm-5.3", None).endswith(" \u00b7 glm-5.3")


def test_single_ellipsis_exactly_once_and_is_u2026():
    got = compose_group_title(LONG_SUBJECT, "glm-5.3", "max")
    assert got.count("\u2026") == 1
    assert "\u2026" in got
    # not the ASCII three-dot lookalike
    assert "..." not in got
    # separator is U+00B7, not a bullet/lookalike
    assert SEPARATOR in got
    for lookalike in ("\u2022", "\u2007", "\u22c5", "|"):
        assert lookalike not in got
    assert got.count(SEPARATOR) == 2


def test_ellipsis_charged_to_the_budget():
    # -4 (separator 3 + ellipsis 1) is load-bearing: 100 exactly, never 101.
    got = compose_group_title(LONG_SUBJECT, "glm-5.3", "max")
    assert len(got) == 100
    assert not got.startswith(LONG_SUBJECT[: 100])


def test_subject_under_the_cap_is_verbatim():
    got = compose_group_title(SUBJECT, "glm-5.3", "medium")
    assert "\u2026" not in got
    assert got == f"{SUBJECT} \u00b7 glm-5.3 \u00b7 r3"


def test_subject_only_truncation_when_no_suffix():
    long_subject = "x" * 250
    got = compose_group_title(long_subject, "", None)
    assert len(got) == MAX_TITLE_LENGTH
    assert got == "x" * 99 + "\u2026"


def test_truncation_rstrips_before_the_ellipsis():
    # A space landing exactly at the cut point must not double up with the ellipsis.
    subject = "a" * 83 + " " + "b" * 20
    got = compose_group_title(subject, "glm-5.3", "max")
    tail = f"{SEPARATOR}glm-5.3 \u00b7 r6"
    budget = MAX_TITLE_LENGTH - len(tail) - 1
    assert budget == 84
    assert subject[budget - 1] == " "  # the cut slice ends on the space
    assert got == "a" * 83 + "\u2026" + tail
    assert "\u2026  " not in got  # ellipsis then exactly one separator, not two spaces
    assert got[len("a" * 83) + 1:] == tail
    assert len(got) == 99


def test_pathological_model_id_overruns_the_cap_by_design():
    """The suffix is never truncated (contract rule 6), so a model id longer than the cap
    yields an over-cap title rather than a lossy one. Real model ids are far shorter."""
    got = compose_group_title("", "a-very-long-model-name-" * 10, "max")
    assert got.endswith(" \u00b7 r6")
    assert got.startswith("a-very-long-model-name-")
    assert len(got) > MAX_TITLE_LENGTH


def test_pure_function_does_not_mutate_inputs():
    subject, model = LONG_SUBJECT, "opencode-go/space-bunny-free"
    compose_group_title(subject, model, "max")
    assert subject == LONG_SUBJECT
    assert model == "opencode-go/space-bunny-free"
