"""Tests para el delta pre vs post activación."""
import os
import pytest
from collections import Counter

from tests.routing.veracity._helpers import load_activation, load_post_activation_events


class TestPreVsPostActivationDelta:
    """Valida que hay distribución de modelos post-activación."""

    def test_model_distribution_post_activation(self):
        """
        La distribución de modelos post-activación debe cubrir ≥ 2 tiers distintos.
        """
        activation = load_activation()
        
        if activation is None:
            require_activation = os.environ.get("HERMES_VERACITY_REQUIRE_ACTIVATION", "0")
            if require_activation != "1":
                pytest.skip("No hay activation para determinar cutoff")
            else:
                pytest.fail("No hay activation con HERMES_VERACITY_REQUIRE_ACTIVATION=1")
        
        events = load_post_activation_events(activation=activation)
        
        if len(events) < 50:
            pytest.skip(f"Menos de 50 eventos post-activación ({len(events)})")
        
        models = [e.get("model", "") for e in events]
        models = [m for m in models if m]  # Filtrar vacíos
        
        assert len(models) > 0, "No hay modelos en eventos post-activación"
        
        model_counts = Counter(models)
        unique_models = set(model_counts.keys())
        
        # Verificar que hay al menos 2 modelos distintos (mini + algo más)
        assert len(unique_models) >= 2, f"Solo hay {len(unique_models)} modelo(s) único(s): {unique_models}"

    def test_no_empty_models_post_activation(self):
        """No debe haber eventos con model vacío post-activación."""
        activation = load_activation()
        
        if activation is None:
            require_activation = os.environ.get("HERMES_VERACITY_REQUIRE_ACTIVATION", "0")
            if require_activation != "1":
                pytest.skip("No hay activation para determinar cutoff")
            else:
                pytest.fail("No hay activation con HERMES_VERACITY_REQUIRE_ACTIVATION=1")
        
        events = load_post_activation_events(activation=activation)
        
        if len(events) < 50:
            pytest.skip(f"Menos de 50 eventos post-activación ({len(events)})")
        
        empty_models = [e for e in events if not e.get("model", "")]
        
        assert len(empty_models) == 0, f"{len(empty_models)} eventos con model vacío"
