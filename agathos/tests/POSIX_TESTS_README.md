# POSIX Compliance Tests

Unit tests and standalone verification for Agathos cross-platform compatibility.

## Files

| File | Purpose | Run Command |
|------|---------|-------------|
| `test_posix_compliance.py` | Full pytest test suite (27 tests) | `pytest agathos/tests/test_posix_compliance.py -v` |
| `run_posix_compliance_check.py` | Standalone check (no pytest needed) | `python agathos/tests/run_posix_compliance_check.py` |

## Quick Start

### With pytest (recommended)

```bash
cd ~/Projects/hermes-dev
source .local/venv/bin/activate
python -m pytest agathos/tests/test_posix_compliance.py -p no:xdist --override-ini="addopts=" -v
```

Expected: 24 passed, 3 skipped (platform-specific tests)

### Without pytest (quick check)

```bash
cd ~/Projects/hermes-dev
source .local/venv/bin/activate
python agathos/tests/run_posix_compliance_check.py
```

Expected: 6 checks passed, 0 failed

## Test Categories

### TestPlatformDetection (4 tests)
- Verifies all modules define `_IS_MACOS`, `_IS_LINUX`, `_IS_WINDOWS`
- Ensures exactly one is True for current platform

### TestPathSeparatorCompliance (4 tests)
- Verifies `os.pathsep` used instead of hardcoded `:`
- Checks PATH construction and splitting
- Source code scan for hardcoded separators

### TestCrossPlatformPaths (4 tests)
- Platform-specific std paths are appropriate
- venv bin dir uses `bin/` (POSIX) or `Scripts/` (Windows)
- Service directory matches platform conventions

### TestServiceManagementGuards (4 tests)
- Service functions check platform before executing
- Non-macOS platforms get graceful degradation
- Status returns consistent dict structure

### TestPathConstruction (3 tests)
- PATH includes hermes-specific directories
- HOME environment variable set
- No duplicate PATH entries

### TestImportsWorkOnAllPlatforms (3 tests)
- All modules import successfully
- No platform-specific import failures

### Platform-Specific Tests (5 tests)
- Windows: Uses `Scripts/`, `;` separator
- macOS: Uses `bin/`, includes Homebrew paths
- Linux: Standard Unix paths only

## Test Output Example

```
agathos/tests/test_posix_compliance.py::TestPlatformDetection::test_venv_utils_platform_constants PASSED
gathos/tests/test_posix_compliance.py::TestPathSeparatorCompliance::test_build_venv_aware_env_uses_pathsep PASSED
...
======================== 24 passed, 3 skipped in 0.09s =========================
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
- name: POSIX Compliance Tests
  run: |
    python -m pytest agathos/tests/test_posix_compliance.py -v
```

## Troubleshooting

### Import errors
Ensure project root is in PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### pytest configuration conflicts
Override pyproject.toml addopts:
```bash
python -m pytest agathos/tests/test_posix_compliance.py -p no:xdist --override-ini="addopts=" -v
```

### Windows-specific failures
Some tests skip on non-Windows platforms (expected behavior).

## Maintenance

When adding new platform-specific code:
1. Define platform constants (`_IS_MACOS`, etc.)
2. Use `os.pathsep` for PATH operations
3. Use `os.sep` or `pathlib.Path` for file paths
4. Add tests for new platform behavior
5. Run both test files before committing
