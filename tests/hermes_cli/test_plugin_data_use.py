"""Data-use requirements are parsed at the boundary: a closed vocabulary, never a bare string."""

import pytest

from tools.connectors.data_use import parse_data_use_requirements


@pytest.mark.parametrize("requires,parsed", [
    (["no_training"], ("no_training",)),
    (("zero_retention", "no_training", "zero_retention"), ("no_training", "zero_retention")),
    (iter(["zero_retention"]), ("zero_retention",)),
    ([], ()),
])
def test_valid_requirements_parse_sorted_and_deduplicated(requires, parsed):
    assert parse_data_use_requirements(requires) == parsed


@pytest.mark.parametrize("requires", ["no_training", "", b"no_training", ["no_training", "no_storage"], [None], 3])
def test_a_bare_string_an_unknown_name_or_a_non_list_is_refused(requires):
    with pytest.raises(ValueError):
        parse_data_use_requirements(requires)
