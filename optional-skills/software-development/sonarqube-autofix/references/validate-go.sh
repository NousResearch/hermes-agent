#!/usr/bin/env bash
set -euo pipefail

# Validation script for Go
# Runs golangci-lint, gofmt and go test

echo "Validating Go..."

# golangci-lint
golangci-lint run

# gofmt (formatter)
gofmt -l .

# go test (tests with coverage)
go test -cover ./...

echo "Go validation completed."
