#!/usr/bin/env bash
# Healthcheck for Self-Heal Pipeline
# Silent unless there is an error.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

failed=0

if ! uv lock --check > /dev/null 2>&1; then
    echo "Healthcheck failed: uv lock out of date"
    uv lock --check
    failed=1
fi

if [ $failed -eq 0 ] && ! uv run ruff check . > /dev/null 2>&1; then
    echo "Healthcheck failed: ruff check"
    uv run ruff check .
    failed=1
fi

if [ $failed -eq 0 ] && ! uv run ty check > /dev/null 2>&1; then
    echo "Healthcheck failed: ty check"
    uv run ty check
    failed=1
fi

if [ $failed -eq 0 ] && ! scripts/run_tests.sh > /dev/null 2>&1; then
    echo "Healthcheck failed: tests"
    scripts/run_tests.sh
    failed=1
fi

if [ $failed -ne 0 ]; then
    exit 1

fi
