#!/usr/bin/env bash
set -euo pipefail

# Validation script for JavaScript/TypeScript
# Runs ESLint, Prettier and Vitest/Jest

echo "Validating JavaScript/TypeScript..."

# ESLint
npx eslint src/

# Prettier (formatter)
npx prettier --check src/

# Vitest/Jest (tests with coverage)
if grep -q "vitest" package.json; then
  npx vitest run --coverage
elif grep -q "jest" package.json; then
  npm test -- --coverage
else
  npm test
fi

echo "JavaScript/TypeScript validation completed."
