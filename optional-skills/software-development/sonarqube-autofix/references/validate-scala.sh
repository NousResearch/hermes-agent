#!/usr/bin/env bash
set -euo pipefail

# Validation script for Scala
# Runs Scalastyle, Scapegoat and sbt test

echo "Validating Scala..."

# Scalastyle
sbt scalastyle

# Scapegoat
sbt scapegoat

# sbt test (tests with coverage)
sbt clean coverage test coverageReport

echo "Scala validation completed."
