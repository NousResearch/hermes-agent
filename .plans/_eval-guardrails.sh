#!/usr/bin/env bash
# Guardrail checks for Phase 2 D1-D3 Evaluation
# TC-9: Scope guardrail — no excluded features in deliverables
# TC-10: D4 confirmed absent (deferred)
set -e
cd /home/jarrad/.hermes/hermes-agent

FAIL_COUNT=0

SEARCH_FILES=(
    "scripts/code-scan/extract_imports.py"
    "skills/code-analysis/code-scan/SKILL.md"
    "skills/code-analysis/validation-gate/SKILL.md"
)

EXCLUDED_PATTERNS=(
    "dashboard"
    "vite"
    "tree.sitter"
    "tree-sitter"
    "wasm"
    "sqlite"
    "flywheel scan"
)

# Special: requesting-code-review — only match if it's an implementation reference, not a "we don't do this" statement
# Special: auto-injection — same; check for positive references not exclusions
POSITIVE_CHECKS=("requesting-code-review" "auto.injection" "react")

echo "--- TC-9: Scope Guardrail ---"
for f in "${SEARCH_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  SKIP: $f not found"
        continue
    fi
    for pat in "${EXCLUDED_PATTERNS[@]}"; do
        if grep -qi "$pat" "$f" 2>/dev/null; then
            echo "  GUARDRAIL HIT: $f contains '$pat'"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
    # Special checks: these patterns are OK if in a "no / not / exclude / don't" context
    # but would be a HIT if the file implements them
    for pat in "${POSITIVE_CHECKS[@]}"; do
        # Match lines with the pattern that are NOT negated
        # (i.e., don't start a sentence with "No", "Not", "exclude", etc.)
        if grep -qi "$pat" "$f" 2>/dev/null; then
            # Check if all occurrences are in negative/exclusion context
            bad_context=$(grep -i "$pat" "$f" | grep -iv 'no \|not \|exclude\|don'\''t\|never\|forbidden\|forbidden_file\|non-goal\|no-\|not allowed' 2>/dev/null || true)
            if [ -n "$bad_context" ]; then
                echo "  GUARDRAIL HIT (positive use): $f implements/positively references '$pat'"
                echo "    Context: $bad_context"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        fi
    done
done
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "  TC-9 PASS: No excluded features in deliverables"
else
    echo "  TC-9 FAIL: $FAIL_COUNT excluded-feature matches found"
fi

echo "--- TC-10: D4 Absent ---"
if git show 5a39c7fc7 --name-only 2>/dev/null | grep -q "requesting-code-review"; then
    echo "  TC-10 FAIL: D4 touch detected in D1-D3 commit"
    FAIL_COUNT=$((FAIL_COUNT + 1))
else
    echo "  TC-10 PASS: D4 correctly absent from D1-D3 commit"
fi

if grep -q "deferred" .beads/phase2-d4-review-integration-deferred.md 2>/dev/null; then
    echo "  TC-10 PASS: D4 bead still marked deferred"
else
    echo "  TC-10 FAIL: D4 bead missing deferred marker"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "GUARDRAIL OVERALL: PASS"
    exit 0
else
    echo "GUARDRAIL OVERALL: FAIL ($FAIL_COUNT)"
    exit 1
fi
