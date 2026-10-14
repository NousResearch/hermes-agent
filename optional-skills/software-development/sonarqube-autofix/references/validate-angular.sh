#!/usr/bin/env bash
set -euo pipefail

# Validation script for Angular
# Runs linting, tests and formatting for Angular projects

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔍 Validating Angular project..."

# Check if it is an Angular project
if [ ! -f "angular.json" ] && [ ! -f "angular-cli.json" ]; then
    echo "❌ Error: Not an Angular project (angular.json not found)"
    exit 1
fi

# Linting
echo "🔧 Running ng lint..."
if command -v ng &> /dev/null; then
    ng lint || {
        echo "❌ Error: ng lint failed"
        exit 1
    }
else
    echo "⚠️  ng not found, trying eslint..."
    npx eslint src/ || {
        echo "❌ Error: eslint failed"
        exit 1
    }
fi

# Formatting with Prettier
echo "✨ Formatting code with Prettier..."
npx prettier --write src/ || {
    echo "⚠️  Warning: Prettier failed, continuing..."
}

# Formatting with ESLint --fix
echo "✨ Formatting code with ESLint..."
npx eslint --fix src/ || {
    echo "⚠️  Warning: ESLint --fix failed, continuing..."
}

# Tests with coverage
echo "🧪 Running tests with coverage..."
ng test --code-coverage --watch=false || {
    echo "❌ Error: ng test failed"
    exit 1
}

# Build for validation
echo "🏗️  Running build for validation..."
ng build --configuration=production || {
    echo "❌ Error: ng build failed"
    exit 1
}

echo "✅ Angular validation completed successfully"
