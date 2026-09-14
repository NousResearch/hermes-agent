"""Proven choice-page value and parser contracts, independent of slash ownership."""

from unittest.mock import AsyncMock

import pytest

from gateway.choice_picker import (
    ChoicePage,
    ChoiceProgress,
    choice_action,
    choice_index,
    choice_label,
)


def choices(count=12):
    return [{"label": f"File {i}", "value": f"identity-{i}"} for i in range(count)]


def test_page_copies_immutable_values_and_bounds_native_display_not_identity():
    raw = [{"label": "研究" * 100, "value": "id" * 200}]
    page = ChoicePage("Files", raw)
    raw[0]["value"] = "changed"
    assert page.choices[0]["value"] == "id" * 200
    assert len(choice_label(page.choices[0], 64)) == 64
    with pytest.raises(TypeError):
        page.choices[0]["value"] = "changed"


@pytest.mark.parametrize("count", [0, 13])
def test_page_never_silently_truncates_navigation(count):
    with pytest.raises(ValueError):
        ChoicePage("Files", choices(count))


@pytest.mark.parametrize(
    "choice",
    [
        {"value": ""},
        {"value": 1},
        {"value": "x" * 1025},
        {"value": "x", "label": "x" * 4097},
    ],
)
def test_invalid_page_fields(choice):
    with pytest.raises(ValueError):
        ChoicePage("Files", [choice])


def test_revision_bound_actions_do_not_retarget_indices():
    action = choice_action("1234abcd", 9, 11)
    assert len(action.encode()) < 64
    assert choice_index(action, "1234abcd", 9, 12) == 11
    for stale in [
        action,
        "cp:1234abcd:10:-1",
        "cp:1234abcd:10:12",
        "cp:1234abcd:10:１",
    ]:
        assert choice_index(stale, "1234abcd", 10, 12) is None


def test_progress_requires_bounded_text_and_work():
    with pytest.raises(ValueError):
        ChoiceProgress("", AsyncMock())
    with pytest.raises(ValueError):
        ChoiceProgress("Sending", None)
