"""Tests para savings claim con bootstrap."""
import os
import pytest

from tests.routing.veracity._helpers import (
    load_activation, 
    load_post_activation_events, 
    bootstrap_ci,
    get_premium_multipliers
)


def compute_savings_vs_baseline_all_opus(events):
    """
    Computa el savings ratio vs baseline (todos opus).
    
    savings = 1 - (costo_real / costo_baseline_opus)
    
    Donde costo_baseline_opus asume que todo habría ido a opus-4.6.
    """
    if not events:
        return 0.0
    
    multipliers = get_premium_multipliers()
    opus_multiplier = multipliers.get("claude-opus-4.6", 1.0)
    
    total_real_cost = 0.0
    total_baseline_cost = 0.0
    
    for event in events:
        model = event.get("model", "")
        tokens = event.get("total_tokens", event.get("prompt_tokens", 0) + event.get("completion_tokens", 0))
        
        # Costo real
        mult = multipliers.get(model, 0.5)  # Default 0.5 si desconocido
        total_real_cost += mult * tokens
        
        # Baseline: todo a opus
        total_baseline_cost += opus_multiplier * tokens
    
    if total_baseline_cost == 0:
        return 0.0
    
    savings = 1.0 - (total_real_cost / total_baseline_cost)
    return max(0.0, min(1.0, savings))  # Clamp a [0, 1]


class TestSavingsClaimWithBootstrap:
    """
    Valida el savings claim con bootstrap confidence interval.
    
    Asserts:
    - IC 95% lower >= 0.40 (parametrizable via HERMES_VERACITY_MIN_SAVINGS)
    - IC 95% upper <= 1.0
    """

    def test_savings_bootstrap_ci(self):
        """Bootstrap savings CI debe cumplir thresholds."""
        activation = load_activation()
        
        if activation is None:
            require_activation = os.environ.get("HERMES_VERACITY_REQUIRE_ACTIVATION", "0")
            if require_activation != "1":
                pytest.skip("No hay activation para determinar cutoff")
            else:
                pytest.fail("No hay activation con HERMES_VERACITY_REQUIRE_ACTIVATION=1")
        
        events = load_post_activation_events(activation=activation)
        
        if len(events) < 200:
            pytest.skip(f"Menos de 200 eventos post-activación ({len(events)})")
        
        # Verificar diversidad de modelos (top 2 deben tener >= 50 eventos cada uno)
        from collections import Counter
        model_counts = Counter(e.get("model", "") for e in events)
        top_models = model_counts.most_common(2)
        
        for model, count in top_models:
            if count < 50:
                pytest.skip(f"Modelo '{model}' tiene solo {count} eventos (< 50)")
        
        # Bootstrap CI
        min_savings = float(os.environ.get("HERMES_VERACITY_MIN_SAVINGS", "0.40"))
        
        lower, upper = bootstrap_ci(
            compute_savings_vs_baseline_all_opus,
            events,
            n=1000
        )
        
        assert lower >= min_savings, f"Savings IC lower ({lower:.3f}) < {min_savings}"
        assert upper <= 1.0, f"Savings IC upper ({upper:.3f}) > 1.0"
