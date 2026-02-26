"""Tests for the hermes_cli models module."""

from hermes_cli.models import OPENROUTER_MODELS, menu_labels, model_ids


class TestModelIds:
    def test_returns_non_empty_list(self):
        ids = model_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_ids_match_models_list(self):
        ids = model_ids()
        expected = [mid for mid, _ in OPENROUTER_MODELS]
        assert ids == expected

    def test_all_ids_contain_provider_slash(self):
        """Model IDs should follow the provider/model format."""
        for mid in model_ids():
            assert "/" in mid, f"Model ID '{mid}' missing provider/ prefix"

    def test_no_duplicate_ids(self):
        ids = model_ids()
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"


class TestMenuLabels:
    def test_same_length_as_model_ids(self):
        assert len(menu_labels()) == len(model_ids())

    def test_first_label_marked_recommended(self):
        labels = menu_labels()
        assert "recommended" in labels[0].lower()

    def test_each_label_contains_its_model_id(self):
        for label, mid in zip(menu_labels(), model_ids()):
            assert mid in label, f"Label '{label}' doesn't contain model ID '{mid}'"

    def test_non_recommended_labels_have_no_tag(self):
        """Only the first model should have (recommended)."""
        labels = menu_labels()
        for label in labels[1:]:
            assert "recommended" not in label.lower(), f"Unexpected 'recommended' in '{label}'"


class TestOpenRouterModels:
    def test_structure_is_list_of_tuples(self):
        for entry in OPENROUTER_MODELS:
            assert isinstance(entry, tuple) and len(entry) == 2
            mid, desc = entry
            assert isinstance(mid, str) and len(mid) > 0
            assert isinstance(desc, str)

    def test_at_least_5_models(self):
        """Sanity check that the models list hasn't been accidentally truncated."""
        assert len(OPENROUTER_MODELS) >= 5
