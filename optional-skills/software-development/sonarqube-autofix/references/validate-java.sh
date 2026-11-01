#!/usr/bin/env bash
set -euo pipefail

# Validation script for Java/Kotlin
# Runs Checkstyle, PMD, Spotless and JaCoCo

echo "Validating Java/Kotlin..."

# Checkstyle
if [ -f "pom.xml" ]; then
  mvn -Dmaven.repo.local=./.m2/repository checkstyle:check
elif [ -f "build.gradle" ] || [ -f "build.gradle.kts" ]; then
  gradle --gradle-user-home ./.gradle checkstyleMain checkstyleTest
fi

# PMD
if [ -f "pom.xml" ]; then
  mvn -Dmaven.repo.local=./.m2/repository pmd:check
elif [ -f "build.gradle" ] || [ -f "build.gradle.kts" ]; then
  gradle --gradle-user-home ./.gradle pmdMain pmdTest
fi

# Spotless (formatter)
if [ -f "pom.xml" ]; then
  mvn -Dmaven.repo.local=./.m2/repository spotless:check
elif [ -f "build.gradle" ] || [ -f "build.gradle.kts" ]; then
  gradle --gradle-user-home ./.gradle spotlessCheck
fi

# JaCoCo (coverage)
if [ -f "pom.xml" ]; then
  mvn -Dmaven.repo.local=./.m2/repository jacoco:check
elif [ -f "build.gradle" ] || [ -f "build.gradle.kts" ]; then
  gradle --gradle-user-home ./.gradle jacocoTestCoverageVerification
fi

echo "Java/Kotlin validation completed."
