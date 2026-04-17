"""Tests para modelos desconocidos en multipliers."""
import os
import pytest

from tests.routing.veracity._helpers import (
    load_activation, 
    load_post_activation_events,
    get_premium_multipliers
)


class TestNoModelUnknownToMultipliers:
    """
    Valida que todo event.model esté en PREMIUM_MULTIPLIERS.
    
    Si hay modelos desconocidos: fail con lista de modelos faltantes.
    """

    def test_all_models_known_in_multipliers(self):
        """Todo modelo en eventos post-activación debe estar en PREMIUM_MULTIPLIERS."""
        activation = load_activation()
        
        if activation is None:
            require_activation = os.environ.get("HERMES_VERACITY_REQUIRE_ACTIVATION", "0")
            if require_activation != "1":
                pytest.skip("No hay activation para determinar cutoff")
            else:
                pytest.fail("No hay activation con HERMES_VERACITY_REQUIRE_ACTIVATION=1")
        
        events = load_post_activation_events(activation=activation)
        
        if not events:
            pytest.skip("No hay eventos post-activación")
        
        multipliers = get_premium_multipliers()
        known_models = set(multipliers.keys())
        
        # Normalizar nombres de modelos (lowercase para comparación)
        known_models_lower = {m.lower() for m in known_models}
        
        unknown_models = set()
        for event in events:
            model = event.get("model", "")
            if model and model.lower() not in known_models_lower:
                unknown_models.add(model)
        
        if unknown_models:
            # Agregar modelos desconocidos a los multipliers para futuras corridas
            # (esto es un warning, no un fail crítico)
            pytest.fail(
                f"Modelos desconocidos en PREMIUM_MULTIPLIERS: {sorted(unknown_models)}. "
                f"Agregar a tests/routing/veracity/_helpers.py:get_premium_multipliers()"
            )
