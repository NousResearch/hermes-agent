#!/usr/bin/env bash
set -euo pipefail

# Script to automatically detect the project stack
# Output: STACK, BUILD_TOOL, TEST_FRAMEWORK, COVERAGE_TOOL

if [ -f "pom.xml" ]; then
  echo "STACK=java"
  echo "BUILD_TOOL=maven"
  echo "TEST_FRAMEWORK=junit"
  echo "COVERAGE_TOOL=jacoco"
elif [ -f "build.gradle" ] || [ -f "build.gradle.kts" ]; then
  echo "STACK=java"
  echo "BUILD_TOOL=gradle"
  echo "TEST_FRAMEWORK=junit"
  echo "COVERAGE_TOOL=jacoco"
elif [ -f "package.json" ]; then
  if grep -q "typescript" package.json; then
    echo "STACK=typescript"
  else
    echo "STACK=javascript"
  fi
  echo "BUILD_TOOL=npm"
  if grep -q "vitest" package.json || grep -q "vitest" package-lock.json 2>/dev/null; then
    echo "TEST_FRAMEWORK=vitest"
  elif grep -q "jest" package.json; then
    echo "TEST_FRAMEWORK=jest"
  else
    echo "TEST_FRAMEWORK=mocha"
  fi
  echo "COVERAGE_TOOL=istanbul"
elif [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
  echo "STACK=python"
  echo "BUILD_TOOL=pip"
  echo "TEST_FRAMEWORK=pytest"
  echo "COVERAGE_TOOL=coverage"
elif [ -f "*.csproj" ] || [ -f "*.sln" ]; then
  echo "STACK=csharp"
  echo "BUILD_TOOL=dotnet"
  echo "TEST_FRAMEWORK=xunit"
  echo "COVERAGE_tool=opencover"
elif [ -f "go.mod" ]; then
  echo "STACK=go"
  echo "BUILD_TOOL=go"
  echo "TEST_FRAMEWORK=gotest"
  echo "COVERAGE_TOOL=gotest"
elif [ -f "Gemfile" ]; then
  echo "STACK=ruby"
  echo "BUILD_TOOL=bundler"
  echo "TEST_FRAMEWORK=rspec"
  echo "COVERAGE_TOOL=simplecov"
elif [ -f "composer.json" ]; then
  echo "STACK=php"
  echo "BUILD_TOOL=composer"
  echo "TEST_FRAMEWORK=phpunit"
  echo "COVERAGE_TOOL=phpunit"
elif [ -f "build.sbt" ]; then
  echo "STACK=scala"
  echo "BUILD_TOOL=sbt"
  echo "TEST_FRAMEWORK=scalatest"
  echo "COVERAGE_TOOL=sbt"
else
  echo "STACK=unknown"
  echo "BUILD_TOOL=unknown"
  echo "TEST_FRAMEWORK=unknown"
  echo "COVERAGE_TOOL=unknown"
fi
