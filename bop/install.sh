#!/usr/bin/env bash
# Idempotent installer for BuiltOnPurpose Hermes guardrails.

set -u

usage() {
  printf 'Usage: bash bop/install.sh [--force]\n'
}

force=0
for arg in "$@"; do
  case "$arg" in
    --force)
      force=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOOK_SRC="$ROOT_DIR/bop/agent-hooks"
SKILL_SRC="$ROOT_DIR/bop/skills"
TEMPLATE="$ROOT_DIR/bop/config/config.yaml.template"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
HOOK_DST="$HERMES_HOME/agent-hooks"
SKILL_DST="$HERMES_HOME/skills"
CONFIG_DST="$HERMES_HOME/config.yaml"

mkdir -p "$HOOK_DST" "$SKILL_DST"

for script in write-fence.sh repo-guard.sh; do
  cp "$HOOK_SRC/$script" "$HOOK_DST/$script"
  chmod 755 "$HOOK_DST/$script"
  printf 'installed hook: %s\n' "$HOOK_DST/$script"
done

cp "$HOOK_SRC/patch_parser.py" "$HOOK_DST/patch_parser.py"
chmod 644 "$HOOK_DST/patch_parser.py"
printf 'installed parser: %s\n' "$HOOK_DST/patch_parser.py"

for skill in ledger-writer capture-intake transcript-followup booking email-triage followup-drafter osr-intake dsm-reporter herk2-specforge herk2-watchdog claude-code-delegate brain-note memory-consolidate; do
  mkdir -p "$SKILL_DST/$skill"
  cp -R "$SKILL_SRC/$skill/." "$SKILL_DST/$skill/"
  chmod 644 "$SKILL_DST/$skill/SKILL.md"
  printf 'installed skill: %s\n' "$SKILL_DST/$skill"
done

if [ ! -f "$CONFIG_DST" ]; then
  cp "$TEMPLATE" "$CONFIG_DST"
  chmod 600 "$CONFIG_DST"
  printf 'installed config: %s\n' "$CONFIG_DST"
  printf 'left untouched: %s\n' "$HERMES_HOME/.env"
  printf 'left untouched: %s\n' "$HERMES_HOME/auth.json"
  exit 0
fi

if [ "$force" -ne 1 ]; then
  printf 'existing config preserved: %s\n' "$CONFIG_DST"
  printf 'diff against template follows; rerun with --force to overwrite.\n'
  diff -u "$CONFIG_DST" "$TEMPLATE" || true
  printf 'left untouched: %s\n' "$HERMES_HOME/.env"
  printf 'left untouched: %s\n' "$HERMES_HOME/auth.json"
  exit 0
fi

timestamp=$(date +%Y%m%d%H%M%S)
backup="$CONFIG_DST.bak.$timestamp"
cp "$CONFIG_DST" "$backup"
cp "$TEMPLATE" "$CONFIG_DST"
chmod 600 "$CONFIG_DST"

printf 'backed up config: %s\n' "$backup"
printf 'overwrote config: %s\n' "$CONFIG_DST"
printf 'left untouched: %s\n' "$HERMES_HOME/.env"
printf 'left untouched: %s\n' "$HERMES_HOME/auth.json"
