#!/usr/bin/env bash
#
# run-coverage.sh — detect the project stack and run the appropriate
# test-coverage command for the quality-test-implementation skill.
#
# Usage:
#   bash references/run-coverage.sh [path]
#
# Supported stacks: .NET (dotnet), Java (Maven/Gradle), Python (pytest).
# Prerequisites per stack are documented in references/coverage-*.md.
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

shopt -s nullglob

run_dotnet() {
  echo ">> .NET: collecting code coverage (XPlat / Coverlet)"
  dotnet test --collect:"XPlat Code Coverage" --results-directory ./TestResults
  if command -v reportgenerator >/dev/null 2>&1; then
    reportgenerator \
      -reports:"TestResults/**/coverage.cobertura.xml" \
      -targetdir:"CoverageReport" \
      -reporttypes:"Html;Cobertura"
  fi
}

run_maven() {
  echo ">> Java/Maven: JaCoCo report"
  mvn -Dmaven.repo.local=./.m2/repository clean test jacoco:report
}

run_gradle() {
  echo ">> Java/Gradle: JaCoCo report"
  ./gradlew test jacocoTestReport
}

run_python() {
  echo ">> Python: pytest with coverage"
  python -m pytest --cov=src --cov-report=term-missing --cov-report=xml:coverage.xml
}

dotnet_files=("$ROOT"/*.sln "$ROOT"/*.csproj)
gradle_files=("$ROOT"/build.gradle*)
req_files=("$ROOT"/requirements*.txt)

if [ "${#dotnet_files[@]}" -gt 0 ]; then
  run_dotnet
elif [ -f "$ROOT/pom.xml" ]; then
  run_maven
elif [ "${#gradle_files[@]}" -gt 0 ]; then
  run_gradle
elif [ -f "$ROOT/pyproject.toml" ] || [ -f "$ROOT/setup.py" ] || [ "${#req_files[@]}" -gt 0 ]; then
  run_python
else
  echo "No supported stack detected (.NET / Java / Python)." >&2
  exit 1
fi
