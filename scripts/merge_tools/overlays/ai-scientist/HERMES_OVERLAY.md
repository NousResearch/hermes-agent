# Hermes AI-Scientist overlay

Fork-specific experiment templates synced into `vendor/openclaw-mirror/AI-Scientist/templates/`
by `scripts/sync_ai_scientist_vendor.py`.

| Template | Purpose |
|----------|---------|
| `nc_kan` | Neural-collapse metrics on synthetic classification (KAN-style feature layers). |
| `nc_kan_proof` | Tight bound / proof-oriented NC metric variant for ShinkaEvolve alignment. |
| `hermes_self_evolve` | Simulated agent task-loop benchmark for Hermes self-evolution research. |

Upstream Sakana templates are replaced on sync; paths under `templates/nc_kan/**`,
`templates/hermes_self_evolve/**`, and `templates/nc_kan_proof/**` are restored from
this overlay directory.
