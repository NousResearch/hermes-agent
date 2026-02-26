"""Tests for the hermes_cli models module."""

from hermes_cli.models import OPENROUTER_MODELS, menu_labels, model_ids


class TestModelIds:
    def test_returns_strings(self):
        ids = model_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(mid, str) for mid in ids)

    def test_ids_match_models_list(self):
        ids = model_ids()
        expected = [mid for mid, _ in OPENROUTER_MODELS]
        assert ids == expected


class TestMenuLabels:
    def test_same_length_as_model_ids(self):
        labels = menu_labels()
        ids = model_ids()
        assert len(labels) == len(ids)

    def test_recommended_in_first(self):
        labels = menu_labels()
        assert "recommended" in labels[0].lower()

    def test_labels_contain_model_ids(self):
        labels = menu_labels()
        ids = model_ids()
        for label, mid in zip(labels, ids):
            assert mid in label
