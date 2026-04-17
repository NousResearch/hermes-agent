> **DATOS SINTÉTICOS — reemplazar tras 100+ turnos reales.**

# Routing Reliability Report

Generated: 2026-04-17T00:19:09.413533+00:00
Event span: 2026-04-10T00:19:09.369378+00:00 → 2026-04-16T23:28:45.369378+00:00 (6.96 days)
Total events: 200

## Model table

| model | req | success% (95% CI) | avg_ms | p95_ms | premium_units | top_fail |
|-------|-----|-------------------|--------|--------|---------------|----------|
| claude-opus-4.7 | 10 | 80.0% [49.0, 94.3] | 3000.0 | 5000.0 | 120.00 | TimeoutError |
| claude-sonnet-4.6 | 50 | 100.0% [92.9, 100.0] | 600.0 | 600.0 | 60.00 | - |
| gpt-5-mini | 140 | 100.0% [97.3, 100.0] | 100.0 | 100.0 | 0.00 | - |

## Worst / Optimal

- **worst_model**: `claude-opus-4.7`
- **optimal_model**: `gpt-5-mini`

## Dream scope (ahorro proyectado)

- turnos/día promedio: **28.7**
- unidades premium/día actual: **25.84**
- baseline all-Opus/día: **105.53**
- **AHORRO/día: 79.68 unidades (75.5%)**
- Proyección 30d conservadora: 2390.5 unidades
- Proyección 30d optimista: 2868.6 unidades

## Savings vs baseline (lifetime of store)

- total_premium_units: 180.00
- baseline_all_opus_units: 735.00
- savings_vs_baseline: **75.5%**

