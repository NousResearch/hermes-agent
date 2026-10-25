#!/usr/bin/env bash
set -euo pipefail

# Validation script for Ruby
# Runs RuboCop, Rufo and rspec

echo "Validating Ruby..."

# RuboCop
bundle exec rubocop

# Rufo (formatter)
bundle exec rufo --check .

# rspec (tests with coverage)
bundle exec rspec --coverage

echo "Ruby validation completed."
