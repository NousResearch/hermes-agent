#!/usr/bin/env bash
set -euo pipefail

# Validation script for Python
# Runs Pylint, Flake8, Black and pytest

echo "Validating Python..."

# Pylint
python -m pylint src/

# Flake8
python -m flake8 src/

# Black (formatter)
python -m black --check src/

# pytest (tests with coverage)
python -m pytest --cov=src tests/ -v

echo "Python validation completed."
