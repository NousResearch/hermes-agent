#!/usr/bin/env bash
# Smoke test for the ast-grep skill on POSIX (macOS / Linux / WSL / Git Bash).
#
# Tests:
#   1. helper --version
#   2. helper langs                              (lists 25 languages)
#   3. helper validate '\w+' --lang ts           (must exit 2 with hint)
#   4. helper validate 'console.log($MSG)'       (must exit 0 plausible)
#   5. helper validate 'def $F($$$):' --lang py  (must exit 2 - trailing colon)
#   6. helper validate 'function $N' --lang ts   (must exit 2 - incomplete)
#   7. helper validate 'foo|bar' --lang ts       (must exit 2 - alternation)
#   8. helper doctor                             (informational; tolerates no-binary)
#   9. helper search w/o binary => exit 3 with install hint
#  10. install.sh --help                         (parses)
#  11. SKILL.md frontmatter shape                (canonical)
#
# Runs against the helper using ONLY stdlib python3 - no ast-grep needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HELPER="python3 $SKILL_DIR/scripts/ast_grep_helper.py"

OUTPUT_DIR="$(mktemp -d -t ast-grep-skill-smoke-XXXXXX)"
trap 'rm -rf "$OUTPUT_DIR"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

omo_runtime_slug() {
  case "$(uname -s)" in
    Darwin) local os_slug="darwin" ;;
    MINGW*|MSYS*|CYGWIN*) local os_slug="win32" ;;
    *) local os_slug="linux" ;;
  esac

  case "$(uname -m | tr '[:upper:]' '[:lower:]')" in
    arm64|aarch64) local arch_slug="arm64" ;;
    *) local arch_slug="x64" ;;
  esac

  printf '%s-%s' "$os_slug" "$arch_slug"
}

fake_sg() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
  cat > "$target" <<'SH'
#!/usr/bin/env bash
if [ "${1:-}" = "--version" ]; then
  printf 'ast-grep 0.45.0 fake\n'
else
  printf 'fake ast-grep\n'
fi
SH
  chmod +x "$target"
}

# 1. --version
$HELPER --version | grep -q "ast-grep-helper" || fail "--version output missing"
pass "--version"

# 2. langs (must list 25)
LANG_LINES=$($HELPER langs | grep -E '^  [a-z]' | wc -l | tr -d ' ')
[ "$LANG_LINES" -ge 25 ] || fail "langs listed only $LANG_LINES (expected >=25)"
pass "langs lists at least 25 languages"

# 3. regex misuse: \w+
set +e
$HELPER validate '\w+' --lang ts > "$OUTPUT_DIR/v1.out" 2>&1
RC=$?
set -e
[ $RC -eq 2 ] || fail "validate '\\w+' should exit 2, got $RC (output: $(cat $OUTPUT_DIR/v1.out))"
grep -qi 'regex' "$OUTPUT_DIR/v1.out" || fail "validate '\\w+' should mention regex (output: $(cat $OUTPUT_DIR/v1.out))"
pass "validate detects \\w regex misuse"

# 4. valid pattern
set +e
$HELPER validate 'console.log($MSG)' --lang ts > "$OUTPUT_DIR/v2.out" 2>&1
RC=$?
set -e
[ $RC -eq 0 ] || fail "validate 'console.log(\$MSG)' should exit 0, got $RC (output: $(cat $OUTPUT_DIR/v2.out))"
pass "validate accepts plausible pattern"

# 5. Python trailing colon
set +e
$HELPER validate 'def $F($$$):' --lang py > "$OUTPUT_DIR/v3.out" 2>&1
RC=$?
set -e
[ $RC -eq 2 ] || fail "validate 'def \$F(\$\$\$):' should exit 2, got $RC"
grep -qi 'colon\|trailing' "$OUTPUT_DIR/v3.out" || fail "validate should mention trailing colon"
pass "validate detects Python trailing colon"

# 6. Incomplete TS function
set +e
$HELPER validate 'function $N' --lang ts > "$OUTPUT_DIR/v4.out" 2>&1
RC=$?
set -e
[ $RC -eq 2 ] || fail "validate 'function \$N' should exit 2, got $RC"
grep -qi 'incomplete\|params\|body' "$OUTPUT_DIR/v4.out" || fail "validate should hint about params/body"
pass "validate detects incomplete TS function"

# 7. Alternation pipe
set +e
$HELPER validate 'foo|bar' --lang ts > "$OUTPUT_DIR/v5.out" 2>&1
RC=$?
set -e
[ $RC -eq 2 ] || fail "validate 'foo|bar' should exit 2, got $RC"
grep -qi 'alternation\|regex' "$OUTPUT_DIR/v5.out" || fail "validate should mention alternation"
pass "validate detects literal | alternation"

# 8. doctor (informational)
$HELPER doctor > "$OUTPUT_DIR/doc.out" 2>&1 || true
grep -q "ast-grep-helper" "$OUTPUT_DIR/doc.out" || fail "doctor missing helper version line"
pass "doctor produces output"

# Given OMO_AST_GREP_SG_PATH points at a fake executable.
# When doctor resolves ast-grep, then it reports that exact path.
FAKE_ENV_SG="$OUTPUT_DIR/fake-env/sg"
fake_sg "$FAKE_ENV_SG"
OMO_AST_GREP_SG_PATH="$FAKE_ENV_SG" $HELPER doctor > "$OUTPUT_DIR/omo-env.out" 2>&1
grep -Fq "ast-grep binary: $FAKE_ENV_SG" "$OUTPUT_DIR/omo-env.out" || fail "OMO_AST_GREP_SG_PATH was not preferred: $(cat "$OUTPUT_DIR/omo-env.out")"
pass "OMO_AST_GREP_SG_PATH resolves first"

# Given HOME has an OMO runtime sg executable.
# When doctor resolves ast-grep, then it reports the HOME runtime path.
RUNTIME_HOME="$OUTPUT_DIR/home"
RUNTIME_SLUG="$(omo_runtime_slug)"
RUNTIME_BIN="sg"
case "$RUNTIME_SLUG" in
  win32-*) RUNTIME_BIN="sg.exe" ;;
esac
FAKE_RUNTIME_SG="$RUNTIME_HOME/.omo/runtime/ast-grep/$RUNTIME_SLUG/$RUNTIME_BIN"
fake_sg "$FAKE_RUNTIME_SG"
HOME="$RUNTIME_HOME" CODEX_HOME= OMO_AST_GREP_SG_PATH= $HELPER doctor > "$OUTPUT_DIR/omo-runtime.out" 2>&1
grep -Fq "ast-grep binary: $FAKE_RUNTIME_SG" "$OUTPUT_DIR/omo-runtime.out" || fail "OMO HOME runtime was not resolved: $(cat "$OUTPUT_DIR/omo-runtime.out")"
pass "OMO HOME runtime resolves before standalone fallback"

# 9. search w/o binary => either runs successfully (binary found) OR exits 3 with hint
# The CI may or may not have ast-grep installed; both are valid.
set +e
$HELPER -q search 'foo()' --lang ts /nonexistent-path-xyzzy > "$OUTPUT_DIR/s1.out" 2>&1
RC=$?
set -e
case "$RC" in
  0|1|4)
    # Binary present but no matches OR sg returned non-fatal error - both fine
    pass "search runs (rc=$RC, ast-grep available)"
    ;;
  3)
    grep -qi 'install' "$OUTPUT_DIR/s1.out" || fail "search rc=3 should print install hint"
    pass "search without binary prints install hint"
    ;;
  *)
    fail "search returned unexpected rc=$RC: $(cat $OUTPUT_DIR/s1.out)"
    ;;
esac

# 10. install.sh syntax check + --help
bash -n "$SKILL_DIR/install.sh" || fail "install.sh has syntax errors"
$SKILL_DIR/install.sh --help > "$OUTPUT_DIR/inst.out" 2>&1 || true
grep -qi 'install' "$OUTPUT_DIR/inst.out" || fail "install.sh --help missing keyword 'install'"
pass "install.sh syntax + --help"

# 11. SKILL.md frontmatter
python3 - <<PY
import re, sys
src = open("$SKILL_DIR/SKILL.md").read()
assert src.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
end = src.find("\n---\n", 4)
assert end > 0, "SKILL.md missing closing ---"
fm = src[4:end]
assert re.search(r"^name:\s*ast-grep\s*$", fm, re.M), "frontmatter missing name: ast-grep"
assert re.search(r"^description:", fm, re.M), "frontmatter missing description"
PY
pass "SKILL.md frontmatter shape"

# 12. All required reference files exist
for f in references/install.md references/patterns.md references/pitfalls.md \
         references/recipes.md references/cli.md references/yaml-rules.md \
         references/sgconfig.md; do
  test -f "$SKILL_DIR/$f" || fail "missing $f"
done
pass "all references present"

python3 - "$SKILL_DIR" <<'PY' || fail "Korean characters found in skill content"
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
hangul = re.compile(r"[\uac00-\ud7a3]")
targets = [root / "SKILL.md", root / "README.md"]
for d in ("references", "scripts", "tests", ".github"):
    targets.extend((root / d).rglob("*"))
targets.extend([root / "install.sh", root / "install.ps1"])
hits = 0
for p in targets:
    if not p.is_file():
        continue
    try:
        text = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for n, line in enumerate(text.splitlines(), 1):
        if hangul.search(line):
            print(f"{p}:{n}: {line.rstrip()}")
            hits += 1
sys.exit(1 if hits else 0)
PY
pass "no Korean in skill content"

echo ""
echo "all smoke tests passed"
