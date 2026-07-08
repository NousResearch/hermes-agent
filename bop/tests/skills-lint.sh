#!/usr/bin/env bash
# Self-contained lint for BU-2 Hermes assistant skills.

set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
INSTALL_SH="$ROOT_DIR/bop/install.sh"
SKILLS_DIR="$ROOT_DIR/bop/skills"

TMP_DIR=$(mktemp -d) && [ -n "$TMP_DIR" ] || exit 1

pass_count=0
fail_count=0

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

pass() {
  printf 'PASS %s\n' "$1"
  pass_count=$((pass_count + 1))
}

fail() {
  printf 'FAIL %s\n' "$1"
  if [ "${2:-}" != "" ]; then
    printf '  %s\n' "$2"
  fi
  fail_count=$((fail_count + 1))
}

check_frontmatter() {
  local file=$1

  python3 - "$file" <<'PY'
import pathlib
import re
import sys

try:
    import yaml
except Exception:
    sys.exit(42)

content = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
if not content.startswith("---"):
    sys.exit(1)
match = re.search(r"\n---\s*\n", content[3:])
if not match:
    sys.exit(1)
frontmatter = content[3:match.start() + 3]
parsed = yaml.safe_load(frontmatter)
if not isinstance(parsed, dict):
    sys.exit(1)
PY
  local status=$?
  if [ "$status" -eq 0 ]; then
    return 0
  fi
  if [ "$status" -ne 42 ]; then
    return 1
  fi

  if [ "$(sed -n '1p' "$file")" != "---" ]; then
    return 1
  fi
  awk 'NR > 1 && $0 == "---" { found = 1; exit } END { exit found ? 0 : 1 }' "$file"
}

frontmatter_name() {
  local file=$1
  python3 - "$file" <<'PY'
import pathlib
import re
import sys

content = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"\n---\s*\n", content[3:])
if not content.startswith("---") or not match:
    sys.exit(1)
frontmatter = content[3:match.start() + 3]
name_match = re.search(r"(?m)^name:\s*['\"]?([^'\"\n#]+)['\"]?\s*$", frontmatter)
if not name_match:
    sys.exit(1)
print(name_match.group(1).strip())
PY
}

file_checksum_listing() {
  local dir=$1
  find "$dir" -type f | sort | while IFS= read -r file; do
    cksum "$file"
  done
}

check_skill_file() {
  local skill=$1
  local file="$SKILLS_DIR/$skill/SKILL.md"
  local name

  if [ -f "$file" ]; then
    pass "$skill SKILL.md exists"
  else
    fail "$skill SKILL.md exists" "$file missing"
    return
  fi

  if check_frontmatter "$file"; then
    pass "$skill frontmatter parses"
  else
    fail "$skill frontmatter parses" "frontmatter is not valid YAML and failed sandwich fallback"
  fi

  name=$(frontmatter_name "$file" 2>/dev/null || true)
  if [ "$name" = "$skill" ]; then
    pass "$skill name matches directory"
  else
    fail "$skill name matches directory" "expected $skill, got ${name:-<missing>}"
  fi

  if grep -q 'Source canon' "$file"; then
    pass "$skill source canon present"
  else
    fail "$skill source canon present" "literal Source canon missing"
  fi

  if grep -Eq '[0-9]{3}-[0-9]{2}-[0-9]{4}' "$file"; then
    fail "$skill NPI self-audit" "SSN-shaped literal found"
  else
    pass "$skill NPI self-audit"
  fi
}

for skill in ledger-writer capture-intake transcript-followup; do
  check_skill_file "$skill"
done

HERMES_TEST_HOME="$TMP_DIR/hermes"
mkdir -p "$HERMES_TEST_HOME/skills/existing"
printf 'keep me\n' > "$HERMES_TEST_HOME/skills/existing/SKILL.md"
existing_before=$(cksum "$HERMES_TEST_HOME/skills/existing/SKILL.md")

HERMES_HOME="$HERMES_TEST_HOME" bash "$INSTALL_SH" > "$TMP_DIR/install-1.out" 2>&1
install_status=$?
if [ "$install_status" -eq 0 ]; then
  pass "install.sh scratch run exits 0"
else
  fail "install.sh scratch run exits 0" "exit $install_status"
fi

installed_lines=$(grep -c '^installed skill:' "$TMP_DIR/install-1.out" || true)
if [ "$installed_lines" -eq 3 ]; then
  pass "install.sh prints three installed skill lines"
else
  fail "install.sh prints three installed skill lines" "got $installed_lines"
fi

for skill in ledger-writer capture-intake transcript-followup; do
  if [ -d "$HERMES_TEST_HOME/skills/$skill" ]; then
    pass "$skill installed directory exists"
  else
    fail "$skill installed directory exists" "$HERMES_TEST_HOME/skills/$skill missing"
  fi

  if [ -f "$HERMES_TEST_HOME/skills/$skill/SKILL.md" ]; then
    pass "$skill installed SKILL.md exists"
  else
    fail "$skill installed SKILL.md exists" "$HERMES_TEST_HOME/skills/$skill/SKILL.md missing"
  fi

  mode=$(python3 - "$HERMES_TEST_HOME/skills/$skill/SKILL.md" <<'PY'
import os
import sys

print(format(os.stat(sys.argv[1]).st_mode & 0o777, "03o"))
PY
)
  if [ "$mode" = "644" ]; then
    pass "$skill installed SKILL.md mode 644"
  else
    fail "$skill installed SKILL.md mode 644" "got $mode"
  fi
done

existing_after=$(cksum "$HERMES_TEST_HOME/skills/existing/SKILL.md")
if [ "$existing_before" = "$existing_after" ]; then
  pass "install.sh preserves unrelated existing skill"
else
  fail "install.sh preserves unrelated existing skill" "existing skill changed"
fi

before_listing=$(file_checksum_listing "$HERMES_TEST_HOME/skills")
HERMES_HOME="$HERMES_TEST_HOME" bash "$INSTALL_SH" > "$TMP_DIR/install-2.out" 2>&1
rerun_status=$?
after_listing=$(file_checksum_listing "$HERMES_TEST_HOME/skills")

if [ "$rerun_status" -eq 0 ]; then
  pass "install.sh rerun exits 0"
else
  fail "install.sh rerun exits 0" "exit $rerun_status"
fi

if [ "$before_listing" = "$after_listing" ]; then
  pass "install.sh rerun is idempotent"
else
  fail "install.sh rerun is idempotent" "skill file checksums changed"
fi

if [ "$fail_count" -ne 0 ]; then
  printf 'skills-lint: %d passed, %d failed\n' "$pass_count" "$fail_count"
  exit 1
fi

printf 'skills-lint: %d passed, 0 failed\n' "$pass_count"
