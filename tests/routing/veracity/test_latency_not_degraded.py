"""Tests para latencia no degradada."""
import os
import pytest
from collections import defaultdict

from tests.routing.veracity._helpers import load_activation, load_post_activation_events


# Thresholds configurables por modelo (en ms)
LATENCY_THRESHOLDS = {
    "gpt-5-mini": 2000,
    "gpt-5": 3000,
    "claude-sonnet-4.6": 10000,
    "claude-sonnet-4.7": 10000,
    "claude-opus-4.6": 30000,
    "claude-opus-4.7": 30000,
    "kimi-k2.5": 5000,
    "deepseek-v3.2": 8000,
    "qwen3.5:397b": 3000,
    "glm-5.1": 2000,
}

# Default threshold si el modelo no está en la lista
DEFAULT_THRESHOLD = 5000


def compute_percentile(values, percentile):
    """Computa el percentil de una lista de valores."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * percentile / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


class TestLatencyNotDegraded:
    """
    Valida que las latencias post-activación no excedan thresholds configurables.
    
    Usa Mann-Whitney U conceptualmente, pero como no tenemos baseline pre-activación real,
    comparamos p95 post vs threshold configurable.
    """

    def test_latency_p95_vs_thresholds(self):
        """p95 por modelo no debe superar el threshold configurado."""
        activation = load_activation()
        
        if activation is None:
            require_activation = os.environ.get("HERMES_VERACITY_REQUIRE_ACTIVATION", "0")
            if require_activation != "1":
                pytest.skip("No hay activation para determinar cutoff")
            else:
                pytest.fail("No hay activation con HERMES_VERACITY_REQUIRE_ACTIVATION=1")
        
        events = load_post_activation_events(activation=activation)
        
        # Agrupar latencias por modelo
        latencies_by_model = defaultdict(list)
        
        for event in events:
            model = event.get("model", "")
            latency = event.get("latency_ms", event.get("latency", event.get("response_time_ms")))
            
            if latency is not None and model:
                try:
                    latencies_by_model[model].append(float(latency))
                except (ValueError, TypeError):
                    continue
        
        warnings = []
        failures = []
        
        for model, latencies in latencies_by_model.items():
            if len(latencies) < 30:
                # Skip por baja muestra
                warnings.append(f"Modelo '{model}': solo {len(latencies)} eventos (< 30)")
                continue
            
            p95 = compute_percentile(latencies, 95)
            threshold = LATENCY_THRESHOLDS.get(model, DEFAULT_THRESHOLD)
            
            if p95 > threshold:
                # Warning si n < 100 (baja potencia)
                if len(latencies) < 100:
                    warnings.append(
                        f"Modelo '{model}': p95={p95:.0f}ms > threshold={threshold}ms (n={len(latencies)}, baja potencia)"
                    )
                else:
                    failures.append(
                        f"Modelo '{model}': p95={p95:.0f}ms > threshold={threshold}ms (n={len(latencies)})"
                    )
        
        # Reportar warnings pero no fallar
        for w in warnings:
            print(f"WARNING: {w}")
        
        assert len(failures) == 0, f"Fallos de latencia: {'; '.join(failures)}"
