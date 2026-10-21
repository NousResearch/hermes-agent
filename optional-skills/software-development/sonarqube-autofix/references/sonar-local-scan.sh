#!/usr/bin/env bash
set -euo pipefail

# Script to run SonarQube locally and revalidate fixes
# Usage: bash sonar-local-scan.sh [project-key]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Detect project key
PROJECT_KEY="${1:-}"
if [ -z "$PROJECT_KEY" ]; then
    # Try to extract from sonar-project.properties
    if [ -f "$PROJECT_ROOT/sonar-project.properties" ]; then
        PROJECT_KEY=$(grep "^sonar.projectKey=" "$PROJECT_ROOT/sonar-project.properties" | cut -d'=' -f2)
    fi
    # If still not found, use the folder name
    if [ -z "$PROJECT_KEY" ]; then
        PROJECT_KEY=$(basename "$PROJECT_ROOT")
    fi
fi

echo "🔍 Running local SonarQube scan for project: $PROJECT_KEY"
echo "📂 Project directory: $PROJECT_ROOT"

# Check if sonar-scanner is installed
if ! command -v sonar-scanner &> /dev/null; then
    echo "❌ Error: sonar-scanner is not installed"
    echo "📖 Install at: https://docs.sonarsource.com/sonarqube-server/latest/analyzing-source-code/scanners/sonarscanner/"
    echo "   Or run ./install-skill-tools.sh --sonar in this repository."
    exit 1
fi

# Create sonar-project.properties if it does not exist
if [ ! -f "$PROJECT_ROOT/sonar-project.properties" ]; then
    echo "📝 Creating sonar-project.properties..."
    cat > "$PROJECT_ROOT/sonar-project.properties" << EOF
sonar.projectKey=$PROJECT_KEY
sonar.sources=src
sonar.tests=tests
sonar.exclusions=**/node_modules/**,**/dist/**,**/bin/**,**/obj/**,**/target/**,**/build/**
sonar.coverage.exclusions=**/*Tests.cs,**/Program.cs,**/Startup.cs,**/AssemblyInfo.cs
EOF
fi

# Detect stack and configure coverage reports
if [ -f "$PROJECT_ROOT/package.json" ]; then
    # JavaScript/TypeScript/Angular
    echo "sonar.javascript.lcov.reportPaths=coverage/lcov.info" >> "$PROJECT_ROOT/sonar-project.properties"
    echo "sonar.typescript.lcov.reportPaths=coverage/lcov.info" >> "$PROJECT_ROOT/sonar-project.properties"
elif [ -f "$PROJECT_ROOT/pom.xml" ] || [ -f "$PROJECT_ROOT/build.gradle" ]; then
    # Java/Kotlin
    echo "sonar.coverage.jacoco.xmlReportPaths=target/site/jacoco/jacoco.xml" >> "$PROJECT_ROOT/sonar-project.properties"
elif [ -f "$PROJECT_ROOT/.csproj" ] || [ -f "$PROJECT_ROOT/*.sln" ]; then
    # C# .NET
    echo "sonar.cs.vscoveragexml.reportPaths=coverage.xml" >> "$PROJECT_ROOT/sonar-project.properties"
fi

# Run scan
echo "🚀 Running sonar-scanner..."
cd "$PROJECT_ROOT"
sonar-scanner \
    -Dsonar.projectKey="$PROJECT_KEY" \
    -Dsonar.host.url="http://localhost:9000" \
    -Dsonar.login="${SONAR_TOKEN:-admin}"

echo "✅ Scan completed"
echo "🌐 Open: http://localhost:9000/dashboard?id=$PROJECT_KEY"
