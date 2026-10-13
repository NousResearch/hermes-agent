"""Unit tests for ``cron/skill_lists.py`` — the one definition of "repr-shaped skill list"
(issue #132867).

The module is deliberately stdlib-only so a tick path can import it mid-upgrade; these tests
pin the two things its callers depend on: which inputs unwrap, and that everything else comes
back as the *same object* (the doctor detects an unwrap with ``is not``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.skill_lists import unwrap_skill_repr


@pytest.mark.parametrize("value, expected", [
    ("['x', 'y']", ["x", "y"]),                     # what a model caller sends
    (["['x', 'y']"], ["x", "y"]),                   # what a pre-fix normalizer stored
    ('["x", "y"]', ["x", "y"]),
    ("[ 'x' , 'y' ]", ["x", "y"]),
    ("('x', 'y')", ["x", "y"]),                     # a tuple literal is a tuple of names
    ("['x']", ["x"]),
    ("[]", []),                                     # an empty literal means no skills, not a skill
    ("['']", [""]),                                 # empties are the callers' to strip
    ("['a', 'b', 'a']", ["a", "b", "a"]),           # dedupe is the callers' job, not this one's
    ("[\"['y']\"]", ["y"]),                         # double-encoded: unwrapped one level at a time
    (["keep", "['x', 'y']"], ["keep", "x", "y"]),   # a repr entry beside ordinary names
    (("keep", "['x']"), ["keep", "x"]),
])
def test_repr_shaped_names_are_unwrapped(value, expected):
    assert unwrap_skill_repr(value) == expected


@pytest.mark.parametrize("value", [
    "plain-skill",
    "[weird-name]",              # bracketed, but not a literal: a skill really named this
    "[1, 2]",                    # a literal, but numbers are not skill names
    "['x', 2]",                  # a mixed literal is refused whole, never half-applied
    "['x', ['y', 'z']]",         # nested lists are not a flat skill list
    "[unclosed",
    "'x'",                       # a literal, but not a list
    "['x'] + ['y']",             # parses to no literal at all
    "",
])
def test_non_repr_input_is_returned_unchanged(value):
    assert unwrap_skill_repr(value) is value


def test_a_clean_list_is_returned_by_identity():
    """Identity is the contract — not just equality — so a caller can tell healed from untouched."""
    skills = ["a", "b"]
    assert unwrap_skill_repr(skills) is skills
    assert unwrap_skill_repr(("a", "b")) == ("a", "b")


def test_non_sequence_scalars_pass_through():
    assert unwrap_skill_repr(None) is None
    assert unwrap_skill_repr(7) == 7
    assert unwrap_skill_repr({"skills": "['x']"}) == {"skills": "['x']"}


def test_an_empty_literal_unwraps_to_no_skills():
    """A caller sending ``skills="[]"`` means "none", and a pre-fix normalizer stored that as
    ``["[]"]``; both must land on no skills rather than on a skill called ``"[]"``."""
    assert unwrap_skill_repr("[]") == []
    assert unwrap_skill_repr(["[]"]) == []
    assert unwrap_skill_repr(["keep", "[]"]) == ["keep"]


def test_an_empty_list_argument_is_still_returned_by_identity():
    skills: list[str] = []
    assert unwrap_skill_repr(skills) is skills


@pytest.mark.parametrize("kwargs, expected", [
    ({}, []),
    ({"skill": "one"}, ["one"]),
    ({"skills": "plain-skill"}, ["plain-skill"]),
    ({"skills": ["a", "b"]}, ["a", "b"]),
    ({"skills": ("a", "b")}, ["a", "b"]),
    ({"skills": ["a", "a", "b"]}, ["a", "b"]),                 # unique, first-seen order
    ({"skills": [" a ", "", "b"]}, ["a", "b"]),                # stripped, empties dropped
    ({"skill": "one", "skills": ["a"]}, ["a"]),                # the plural wins
    ({"skills": "['a', 'b']"}, ["a", "b"]),                    # what a model caller sends
    ({"skills": ["['a', 'b']"]}, ["a", "b"]),                  # what a pre-fix release stored
    ({"skill": "['a', 'b']"}, ["a", "b"]),                     # legacy singular, repr-shaped
    ({"skill": "['a', 'b']", "skills": ["c"]}, ["c"]),         # plural still wins
])
def test_normalize_skill_names(kwargs, expected):
    from cron.skill_lists import normalize_skill_names
    assert normalize_skill_names(**kwargs) == expected
