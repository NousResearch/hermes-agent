"""The contract's activity helpers return exactly the values of the activity rules the backend runs (``session_rules.activity``)."""

import pytest

from persistence.conversation import sessions as contract
from session_rules import activity as rules

_PROVENANCES = [
    None, "", "  ", "unknown", "bogus", "AGENT.COMPRESSION", " agent.compression ", "agent.compression\n",
    *(member.value for member in rules.ActivityProvenance), *rules.ActivityProvenance,
]
_DESCRIPTIONS = [None, "", "   ", " walking ", "x" * 119, "x" * 120, "x" * 121, " " + "y" * 120 + " ", "z" * 500, "\u00e9" * 121]


def test_the_activity_vocabulary_is_the_enum():
    # CONTRACT: the description budget, the unknown label and the provenance set are the activity rules' values.
    assert contract.ACTIVITY_DESCRIPTION_MAX == rules.ACTIVITY_DESCRIPTION_MAX
    assert contract.ACTIVITY_PROVENANCE_UNKNOWN == rules.ActivityProvenance.UNKNOWN.value
    assert contract.ACTIVITY_PROVENANCES == {member.value for member in rules.ActivityProvenance}


@pytest.mark.parametrize("provenance", _PROVENANCES)
def test_a_normalized_provenance_is_the_enum_value(provenance):
    # CONTRACT: the stored provenance is the value of the member the activity rules normalize to, unknown included.
    assert contract.normalize_activity_provenance(provenance) == rules.normalize_activity_provenance(provenance).value


@pytest.mark.parametrize("description", _DESCRIPTIONS)
def test_a_bounded_description_matches_the_activity_rules(description):
    # CONTRACT: the stored description is stripped and clamped as the activity rules clamp it.
    assert contract.bound_activity_description(description) == rules.bound_activity_description(description)
