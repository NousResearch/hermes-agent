# Argus PR Checklist

Pre-flight checklist before submitting Argus to hermes-agent.

## Structure Compliance

- [ ] `argus/__init__.py` exists with public API exports in `__all__`
- [ ] Tests are in `tests/argus/` (not `argus/tests/`)
- [ ] Production code uses relative imports (`from . import x`)
- [ ] No `sys.path.insert` in production code (argus.py, wal_monitor.py exempted for standalone mode)

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
