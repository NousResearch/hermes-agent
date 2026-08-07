#!/usr/bin/env bash
# Defuddle skill verification script
# Run after installing the defuddle skill to confirm it works end-to-end.
# Usage: bash scripts/verify.sh

set -euo pipefail

TEST_URL="https://stephango.com/saw"
EXPECTED_TITLE="Use the saw, fear the saw"
PASS=0
FAIL=0

echo "=== Defuddle Skill Verification ==="
echo ""

# Check prerequisites
echo -n "1. Node.js available... "
if command -v node &>/dev/null; then
    echo "OK (v$(node --version 2>/dev/null | sed 's/v//'))"
    PASS=$((PASS+1))
else
    echo "FAIL — node not found"
    FAIL=$((FAIL+1))
fi

echo -n "2. npx available... "
if command -v npx &>/dev/null; then
    echo "OK"
    PASS=$((PASS+1))
else
    echo "FAIL — npx not found"
    FAIL=$((FAIL+1))
fi

# Test 1: URL → Markdown
echo -n "3. URL → Markdown... "
OUTPUT=$(npx -y defuddle@latest parse "$TEST_URL" --markdown 2>/dev/null)
if echo "$OUTPUT" | grep -q "Fear the saw"; then
    echo "OK"
    PASS=$((PASS+1))
else
    echo "FAIL — expected content not found"
    FAIL=$((FAIL+1))
fi

# Test 2: URL → JSON with metadata
echo -n "4. URL → JSON metadata... "
JSON_OUTPUT=$(npx -y defuddle@latest parse "$TEST_URL" --json 2>/dev/null)
if echo "$JSON_OUTPUT" | grep -q "$EXPECTED_TITLE"; then
    echo "OK"
    PASS=$((PASS+1))
else
    echo "FAIL — title not found in JSON"
    FAIL=$((FAIL+1))
fi

# Test 3: Frontmatter output
echo -n "5. Markdown + frontmatter... "
FM_OUTPUT=$(npx -y defuddle@latest parse "$TEST_URL" --markdown --frontmatter 2>/dev/null)
if echo "$FM_OUTPUT" | head -1 | grep -q "^---"; then
    echo "OK"
    PASS=$((PASS+1))
else
    echo "FAIL — no YAML frontmatter"
    FAIL=$((FAIL+1))
fi

# Test 4: Pipe from curl
echo -n "6. Pipe from curl (stdin mode)... "
PIPE_OUTPUT=$(curl -sL "$TEST_URL" | npx -y defuddle@latest parse --markdown 2>/dev/null)
if echo "$PIPE_OUTPUT" | grep -q "Fear the saw"; then
    echo "OK"
    PASS=$((PASS+1))
else
    echo "FAIL — piped content not extracted"
    FAIL=$((FAIL+1))
fi

# Test 5: Single property extraction
echo -n "7. Single property (--property title)... "
TITLE_OUTPUT=$(npx -y defuddle@latest parse "$TEST_URL" --property title 2>/dev/null | tr -d '[:space:]')
if echo "$TITLE_OUTPUT" | grep -q "Usethesaw,fearthesaw"; then
    echo "OK"
    PASS=$((PASS+1))
else
    echo "FAIL — title property mismatch (got: $TITLE_OUTPUT)"
    FAIL=$((FAIL+1))
fi

# Test 6: Wikipedia article (heavier content)
echo -n "8. Wikipedia article extraction... "
WIKI_OUTPUT=$(npx -y defuddle@latest parse "https://en.wikipedia.org/wiki/Readability" --markdown 2>/dev/null | head -5)
if echo "$WIKI_OUTPUT" | grep -q "Readability"; then
    echo "OK"
    PASS=$((PASS+1))
else
    echo "FAIL — Wikipedia content not extracted"
    FAIL=$((FAIL+1))
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ $FAIL -eq 0 ]; then
    echo "ALL TESTS PASSED"
    exit 0
else
    echo "SOME TESTS FAILED"
    exit 1
fi