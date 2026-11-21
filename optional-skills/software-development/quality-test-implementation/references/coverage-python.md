# Python Test Coverage

Target: **90% line and branch** coverage. Primary runner: **pytest** with **pytest-cov** (Coverage.py under the hood). Also documents the **unittest + coverage.py** path.

## Prerequisites

```bash
python -m pip install pytest pytest-cov coverage
# optional type checking
python -m pip install mypy
```

## pytest + pytest-cov (recommended)

```bash
# terminal report (missing lines) + XML for CI
pytest --cov=src --cov-report=term-missing --cov-report=xml:coverage.xml
```

### Enforce the threshold (build fails below target)

```bash
pytest --cov=src --cov-report=xml --cov-fail-under=90
```

### Configuration (pyproject.toml)

```toml
[tool.coverage.run]
branch = true
source = ["src"]

[tool.coverage.report]
show_missing = true
fail_under = 90

[tool.pytest.ini_options]
addopts = "--cov=src --cov-report=term-missing"
```

Then a plain `pytest` collects coverage and enforces the gate.

## unittest + coverage.py (no pytest)

```bash
coverage run -m unittest discover -s tests
coverage report -m
coverage xml   # produces coverage.xml
```

## Type checking (part of quality gate)

```bash
mypy src
```

## Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| `--cov=src` empty | No source measured | Point `--cov` at the package import name, not the folder path |
| `branch` not enabled | Branch coverage ignored | Add `--cov-branch` or `branch = true` in config |
| XML not produced for CI | Pipeline can't read coverage | Add `--cov-report=xml:coverage.xml` |
| Threshold never enforced | Coverage silently regresses | Use `--cov-fail-under=90` |
