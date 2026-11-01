#!/usr/bin/env bash
set -euo pipefail

# Validation script for C#/.NET
# Runs Roslyn Analyzers, dotnet format and dotnet test

echo "Validating C#/.NET..."

# Roslyn Analyzers
dotnet build /p:RunAnalyzersDuringBuild=true

# dotnet format (formatter)
dotnet format --verify-no-changes

# dotnet test (tests with coverage)
dotnet test /p:CollectCoverage=true /p:CoverageFormat=cobertura

echo "C#/.NET validation completed."
