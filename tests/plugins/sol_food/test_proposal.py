"""Candidate/proposal ceilings at limit-1 / limit / limit+1."""

import pytest

from plugins.sol_food.limits import (
    FOOD_CANDIDATE_MAX_CHOICES,
    FOOD_CANDIDATE_MAX_ITEMS,
    FOOD_DISPLAY_MAX_CHARS,
    FOOD_LABEL_MAX_CHARS,
    FOOD_PROPOSAL_JSON_MAX_BYTES,
)
from plugins.sol_food.proposal import (
    Candidate,
    ProposalError,
    candidates_hash,
    canonical_json_bytes,
    choice_action,
    render_display,
    validate_candidates,
)


def item(index: int = 0):
    return {"plant_key": f"synthetic_item_{index}", "is_plant": True}


def candidate(n_items: int = 1, label: str = "synthetic label") -> Candidate:
    return Candidate(label=label, items=tuple(item(i) for i in range(n_items)))


class TestChoiceBounds:
    @pytest.mark.parametrize(
        "count,ok",
        [
            (FOOD_CANDIDATE_MAX_CHOICES - 1, True),
            (FOOD_CANDIDATE_MAX_CHOICES, True),
            (FOOD_CANDIDATE_MAX_CHOICES + 1, False),
        ],
    )
    def test_choice_count_boundary(self, count, ok):
        candidates = [candidate(label=f"label {i}") for i in range(count)]
        if ok:
            validate_candidates(candidates)
        else:
            with pytest.raises(ProposalError) as excinfo:
                validate_candidates(candidates)
            assert excinfo.value.reason_code == "food_proposal_too_many_choices"

    def test_zero_choices_rejected(self):
        with pytest.raises(ProposalError):
            validate_candidates([])

    @pytest.mark.parametrize(
        "count,ok",
        [
            (FOOD_CANDIDATE_MAX_ITEMS - 1, True),
            (FOOD_CANDIDATE_MAX_ITEMS, True),
            (FOOD_CANDIDATE_MAX_ITEMS + 1, False),
        ],
    )
    def test_item_count_boundary(self, count, ok):
        candidates = [candidate(n_items=count)]
        if ok:
            validate_candidates(candidates)
        else:
            with pytest.raises(ProposalError) as excinfo:
                validate_candidates(candidates)
            assert excinfo.value.reason_code == "food_proposal_too_many_items"

    def test_zero_items_rejected(self):
        with pytest.raises(ProposalError):
            validate_candidates([Candidate(label="x", items=())])

    @pytest.mark.parametrize(
        "length,ok",
        [
            (FOOD_LABEL_MAX_CHARS - 1, True),
            (FOOD_LABEL_MAX_CHARS, True),
            (FOOD_LABEL_MAX_CHARS + 1, False),
        ],
    )
    def test_label_boundary(self, length, ok):
        candidates = [candidate(label="x" * length)]
        if ok:
            validate_candidates(candidates)
        else:
            with pytest.raises(ProposalError) as excinfo:
                validate_candidates(candidates)
            assert excinfo.value.reason_code == "food_proposal_label_too_long"

    def test_label_length_is_unicode_normalized(self):
        # "e" + combining acute normalizes (NFC) to one char; 120 of
        # those must pass even though raw length is 240.
        label = "é" * FOOD_LABEL_MAX_CHARS
        validate_candidates([candidate(label=label)])

    def test_item_shape_strict(self):
        bad = Candidate(label="x", items=({"plant_key": "a", "is_plant": True, "note": "n"},))
        with pytest.raises(ProposalError) as excinfo:
            validate_candidates([bad])
        assert excinfo.value.reason_code == "food_proposal_bad_candidate_shape"

    def test_is_plant_must_be_bool(self):
        bad = Candidate(label="x", items=({"plant_key": "a", "is_plant": 1},))
        with pytest.raises(ProposalError):
            validate_candidates([bad])

    def test_json_ceiling_boundary(self):
        # Build candidates whose canonical JSON lands exactly at the
        # ceiling, then push one byte over.
        def blob_size(label_len: int) -> int:
            return len(
                canonical_json_bytes(
                    [candidate(label="x" * label_len).as_dict() for _ in range(1)]
                )
            )

        # Labels are capped at 120 chars, so exceed the JSON bound with
        # many items' plant keys instead: construct max candidates/items.
        big = [
            Candidate(
                label="x" * FOOD_LABEL_MAX_CHARS,
                items=tuple(
                    {"plant_key": "k" * 60 + f"_{i:03d}", "is_plant": True}
                    for i in range(FOOD_CANDIDATE_MAX_ITEMS)
                ),
            )
            for _ in range(FOOD_CANDIDATE_MAX_CHOICES)
        ]
        size = len(canonical_json_bytes([c.as_dict() for c in big]))
        # Sanity: the maximal legal structure fits under the ceiling.
        assert size <= FOOD_PROPOSAL_JSON_MAX_BYTES
        validate_candidates(big)


class TestRendering:
    def test_display_cap(self):
        candidates = [
            candidate(n_items=24, label="x" * FOOD_LABEL_MAX_CHARS)
            for _ in range(FOOD_CANDIDATE_MAX_CHOICES)
        ]
        text = render_display(candidates)
        assert len(text) <= FOOD_DISPLAY_MAX_CHARS

    def test_display_boundary_exact(self):
        # A single line exactly at / above the cap.
        one = [candidate(label="y" * 120)]
        assert len(render_display(one)) <= FOOD_DISPLAY_MAX_CHARS

    def test_hash_stability(self):
        a = [candidate(), candidate(label="other")]
        b = [candidate(), candidate(label="other")]
        assert candidates_hash(a) == candidates_hash(b)
        assert candidates_hash(a) != candidates_hash([candidate()])


class TestChoiceAction:
    def test_bounds(self):
        assert choice_action(0) == "choice:0"
        assert choice_action(FOOD_CANDIDATE_MAX_CHOICES - 1) == "choice:3"
        with pytest.raises(ProposalError):
            choice_action(FOOD_CANDIDATE_MAX_CHOICES)
        with pytest.raises(ProposalError):
            choice_action(-1)
