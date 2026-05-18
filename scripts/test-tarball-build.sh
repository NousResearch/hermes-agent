#!/bin/bash
# Test script for tarball installation

set -e

echo "=== Testing Tarball Build Scripts ==="
echo ""

# Test 1: Build script help
echo "Test 1: Build script help..."
if bash scripts/build-tarball.sh --help > /dev/null 2>&1; then
    echo "✓ Build script help works"
else
    echo "✗ Build script help failed"
    exit 1
fi

# Test 2: Install script help
echo "Test 2: Install script help..."
if bash scripts/install-tarball.sh --help > /dev/null 2>&1; then
    echo "✓ Install script help works"
else
    echo "✗ Install script help failed"
    exit 1
fi

# Test 3: Check release.py syntax
echo "Test 3: Release script syntax..."
if python3 -m py_compile scripts/release.py; then
    echo "✓ Release script syntax is valid"
else
    echo "✗ Release script syntax error"
    exit 1
fi

# Test 4: Check required files exist
echo "Test 4: Required files..."
REQUIRED_FILES=(
    "scripts/build-tarball.sh"
    "scripts/install-tarball.sh"
    "scripts/release.py"
    ".build-metadata.schema.json"
    "docs/tarball-installation.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file missing"
        exit 1
    fi
done

# Test 5: Check version resolution
echo "Test 5: Version resolution..."
VERSION=$(grep -E '^version\s*=' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
if [ -n "$VERSION" ]; then
    echo "  ✓ Version: $VERSION"
else
    echo "  ✗ Could not resolve version"
    exit 1
fi

echo ""
echo "=== All tests passed! ==="
