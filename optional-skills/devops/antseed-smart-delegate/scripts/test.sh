#!/usr/bin/env bash
# antseed-smart-delegate/test.sh — validate skill structure, syntax, and secrets
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0; FAIL=0

pass() { echo "  PASS $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL $1"; FAIL=$((FAIL+1)); }

echo "=== antseed-smart-delegate: skill validation ==="
echo ""

echo "[Structure]"
test -f "$SKILL_DIR/SKILL.md" && pass "SKILL.md exists" || fail "SKILL.md missing"
test -d "$SKILL_DIR/scripts" && pass "scripts/ dir exists" || fail "scripts/ missing"
test -d "$SKILL_DIR/references" && pass "references/ dir exists" || fail "references/ missing"
for sh in best-peer.sh test.sh; do
    test -f "$SKILL_DIR/scripts/$sh" && pass "$sh exists" || fail "$sh missing"
done
test -f "$SKILL_DIR/references/setup.md" && pass "setup.md exists" || fail "setup.md missing"
test -f "$SKILL_DIR/references/model-catalog.md" && pass "model-catalog.md exists" || fail "model-catalog.md missing"

echo ""
echo "[Frontmatter]"
head -1 "$SKILL_DIR/SKILL.md" | grep -q "^---$" && pass "starts with ---" || fail "no --- opener"
grep -q "^name:" "$SKILL_DIR/SKILL.md" && pass "has name" || fail "no name"
grep -q "^description:" "$SKILL_DIR/SKILL.md" && pass "has description" || fail "no description"
grep -q "^version:" "$SKILL_DIR/SKILL.md" && pass "has version" || fail "no version"
grep -q "^author:" "$SKILL_DIR/SKILL.md" && pass "has author" || fail "no author"
grep -q "^license:" "$SKILL_DIR/SKILL.md" && pass "has license" || fail "no license"
grep -q "tags:" "$SKILL_DIR/SKILL.md" && pass "has tags" || fail "no tags"
grep -q "related_skills:" "$SKILL_DIR/SKILL.md" && pass "has related_skills" || fail "no related_skills"
grep -q "requires_toolsets:" "$SKILL_DIR/SKILL.md" && pass "has requires_toolsets" || fail "no requires_toolsets"
grep -q "⚡ Quick Reference" "$SKILL_DIR/SKILL.md" && pass "has Quick Reference table" || fail "no Quick Reference"
grep -q "DO NOT read script files" "$SKILL_DIR/SKILL.md" && pass "has DO NOT read instruction" || fail "no DO NOT read instruction"

echo ""
echo "[Size limits]"
python3 -c "
import re
t=open('$SKILL_DIR/SKILL.md').read()
m=re.search(r'^description: (.*)$', t, re.M)
assert m and len(m.group(1).strip()) <= 1024, 'description too long: %d' % len(m.group(1).strip())
assert len(t) <= 6000, 'SKILL.md too large: %d chars' % len(t)
" 2>&1 && pass "size OK" || fail "size FAIL"

echo ""
echo "[Scripts syntax + permissions]"
for sh in best-peer.sh; do
    f="$SKILL_DIR/scripts/$sh"
    test -x "$f" && pass "$sh executable" || fail "$sh not executable"
    bash -n "$f" 2>/dev/null && pass "$sh syntax OK" || fail "$sh syntax FAIL"
done

echo ""
echo "[No secrets leaked]"
(grep -rqE "0x[0-9a-f]{20,}" "$SKILL_DIR/SKILL.md" 2>/dev/null) && fail "wallet addr leak" || pass "no wallet addrs"
(grep -rqE "89\.110|192\.168\." "$SKILL_DIR/SKILL.md" 2>/dev/null) && fail "IP leak" || pass "no IPs"
(grep -rqE "ghp_|gho_|sk-[A-Za-z0-9]{20,}" "$SKILL_DIR/SKILL.md" 2>/dev/null) && fail "token leak" || pass "no tokens"
(grep -rq "ryptotalent|sava" "$SKILL_DIR/SKILL.md" 2>/dev/null) && fail "username leak" || pass "no usernames"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]