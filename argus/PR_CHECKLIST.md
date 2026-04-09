# Argus PR Checklist

Pre-flight checklist before submitting Argus to hermes-agent.

## Structure Compliance

- [ ] `argus/__init__.py` exists with public API exports in `__all__`
- Tests are in `tests/argus/` (not `argus/tests/`)
- Production code uses relative imports (`from . import x`)
- No `sys.path.insert` in production code (argus.py, wal_monitor.py exempted for standalone mode)

## Optional Features

All modules can be toggled via `config.yaml`. Most default to `true` (enabled).
ML features default to `false` (disabled).

### Full Configuration (`~/.hermes/config.yaml`)

```yaml
argus:
  # Core settings
  poll_interval: 30
  session_timeout_minutes: 60
  entropy_threshold: 3
  quality_threshold: 0.92
  max_restart_count: 3
  
  # Module toggles (all optional, default ON)
  entropy_detection_enabled: true    # detect_repeat_tool_calls, stuck_loops, etc.
  actions_enabled: true              # restart, kill, inject prompts
  notifications_enabled: true        # Telegram, Discord, Slack, etc.
  metrics_enabled: true              # Prometheus metrics export
  wal_monitor_enabled: true          # Real-time WAL monitoring
  provider_health_enabled: true      # Provider health tracking
  prime_directives_enabled: true     # Prime directive checking
  cleanup_enabled: true              # Orphaned session cleanup
  drift_detection_enabled: true       # Quality drift detection
  resource_checks_enabled: true      # Resource exhaustion monitoring
  audit_trail_enabled: true          # Audit logging
  
  # ML data export (optional, default OFF)
  ml_data_enabled: false             # Export trajectories to ~/.hermes/argus/ml_data/
  ml_memory_enabled: false           # Record to holographic memory
```

### ML Data Export (`ml_data.py`)

Exports entropy detections as training data for model improvement.

**What it does:**
1. Generates trajectories showing failure → detection → recovery
2. Records entropy facts to holographic memory (if available)
3. Outputs to `~/.hermes/argus/ml_data/`

**Integration targets:**
- holographic_memory.db (via hermes_state)
- ShareGPT format for fine-tuning
- Native hermes-agent ML pipeline (if available)

### Cost Monitoring (`cost_monitor.py`)

Budget and cost alerts using Hermes insights (no tracking duplication).

**User Configuration (config.yaml):**
```yaml
argus:
  cost_monitoring:
    enabled: true                    # Disabled by default - set to true to enable
    daily_budget: 20.00              # Default $20 (only applies if enabled)
    alert_at_percent: 80             # Alert threshold
    expensive_session_threshold: 2.00
    per_provider_limits:             # Optional, auto-populated
      anthropic: 5.00
      fireworks: 3.00
```

**How it works:**
1. Disabled by default - user must explicitly enable in config.yaml
2. Discovers providers dynamically from session history
3. Queries Hermes insights engine for current spend
4. Alerts on: daily budget, per-provider limits, expensive sessions
5. Records alerts to audit trail
6. Sends notifications if enabled

**No hardcoded providers** - works with any provider the user configures.

### Circuit Breaker (`circuit_breaker.py`)

Automatic provider disable on failure using Hermes config integration.

**User Configuration (config.yaml):**
```yaml
argus:
  circuit_breaker:
    enabled: false                  # Disabled by default - set to true to enable
    failure_threshold: 3            # Consecutive failures to open circuit
    error_rate_threshold: 0.50    # 50% error rate over window
    min_requests_for_rate: 5      # Min requests before rate check
    recovery_timeout: 300           # Seconds before retry (half-open)
    half_open_requests: 2         # Successful requests to close
    notify_on_open: true            # Alert when circuit opens
    notify_on_close: true           # Alert when circuit closes
```

**How it works:**
1. Uses Hermes `load_config()` and `save_config()` to manage `provider_routing.ignore`
2. Uses Argus `provider_health` module for failure detection
3. States: CLOSED (normal) → OPEN (disabled after failures) → HALF_OPEN (testing) → CLOSED (recovered)
4. Alerts on circuit open/close
5. Records events to audit trail

**Uses Hermes classes/functions:**
- `hermes_cli.config.load_config()` - read current provider_routing
- `hermes_cli.config.save_config()` - write updated provider_routing
- `provider_health.run_provider_check()` - get error rates

## File Organization

### Production (ships with package)
```
argus/
├── __init__.py          # Package entrypoint
├── argus.py            # Main class
├── entropy.py          # Detection algorithms
├── actions.py          # Response actions
├── notifications.py    # Alert system
├── wal_monitor.py      # WAL monitoring
├── provider_health.py  # Provider tracking
└── watcher_schema.sql  # DB schema
```

### Dev-only (excluded from package)
```
tests/argus/
├── test_*.py           # Unit tests
└── simulation/         # Simulation framework (dev-only)
    ├── conftest.py     # Path setup (centralized)
    └── ...
```

## Verification Steps

```bash
# 1. Check structure
./argus/bin/check-argus

# 2. Run smoke tests
cd tests/argus/simulation && python3 run_all_tests.py --quick

# 3. Verify imports (requires Python 3.10+ env)
python3 -c "from argus import Argus, detect_repeat_tool_calls"

# 4. Check pyproject.toml excludes
# Ensure: exclude = ["argus.tests*"] or similar
```

## Common Issues

| Issue | Fix |
|-------|-----|
| `No module named 'argus'` | Add `argus` to `[tool.setuptools.packages.find]` in pyproject.toml |
| `ModuleNotFoundError: entropy` | Use relative imports: `from . import entropy` |
| `sys.path.insert` in test files | Use `conftest.py` for path setup instead |
| Tests in wrong location | Move `argus/tests/` → `tests/argus/` |

## pyproject.toml Configuration

```toml
[tool.setuptools.packages.find]
include = ["argus"]
exclude = ["argus.tests*", "tests.argus*"]
```

## Notes

- Simulation framework is **dev-only** and won't ship with production package
- `sys.path.insert` in `argus.py` and `wal_monitor.py` is intentional for standalone script execution
- Tests should use pytest with `conftest.py` for path setup, not per-file `sys.path.insert`
