#!/usr/bin/env bash
# antseed-smart-delegate/test.sh — validate skill structure and scripts
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
for sh in preflight.sh best-peer.sh cost-report.sh test.sh; do
    test -f "$SKILL_DIR/scripts/$sh" && pass "$sh exists" || fail "$sh missing"
done

echo ""
echo "[Frontmatter]"
head -1 "$SKILL_DIR/SKILL.md" | grep -q "^---$" && pass "starts with ---" || fail "no --- opener"
grep -q "^name:" "$SKILL_DIR/SKILL.md" && pass "has name" || fail "no name"
grep -q "^description:" "$SKILL_DIR/SKILL.md" && pass "has description" || fail "no description"
grep -q "^version:" "$SKILL_DIR/SKILL.md" && pass "has version" || fail "no version"
grep -q "^author:" "$SKILL_DIR/SKILL.md" && pass "has author" || fail "no author"
grep -q "^license:" "$SKILL_DIR/SKILL.md" && pass "has license" || fail "no license"
grep -q "tags:" "$SKILL_DIR/SKILL.md" && pass "has tags" || fail "no tags"
grep -q "required_environment_variables" "$SKILL_DIR/SKILL.md" && pass "has required_env_vars" || fail "no required_env_vars"

echo ""
echo "[Size limits]"
python3 -c "
import re
t=open('$SKILL_DIR/SKILL.md').read()
m=re.search(r'^description: (.*)$', t, re.M)
assert m and len(m.group(1)) <= 1024, 'description too long'
assert len(t) <= 100000, 'skill too large'
" 2>&1 && pass "size OK" || fail "size FAIL"

echo ""
echo "[Scripts syntax + permissions]"
for sh in preflight.sh best-peer.sh cost-report.sh; do
    f="$SKILL_DIR/scripts/$sh"
    test -x "$f" && pass "$sh executable" || fail "$sh not executable"
    bash -n "$f" 2>/dev/null && pass "$sh syntax OK" || fail "$sh syntax FAIL"
done

echo ""
echo "[No secrets leaked]"
(grep -rqE "0x[0-9a-f]{20,}" "$SKILL_DIR/SKILL.md") 2>/dev/null && fail "wallet addr leak" || pass "no wallet addrs"
(grep -rqE "89\.110|192\.168\." "$SKILL_DIR/SKILL.md") 2>/dev/null && fail "IP leak" || pass "no IPs"
(grep -rqE "ghp_|gho_|sk-[A-Za-z0-9]{20,}" "$SKILL_DIR/SKILL.md") 2>/dev/null && fail "token leak" || pass "no tokens"
(grep -rq "ryptotalent|sava" "$SKILL_DIR/SKILL.md") 2>/dev/null && fail "username leak" || pass "no usernames"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
