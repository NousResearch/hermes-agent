#!/bin/bash
# pin-assigned-skills.sh — Ring-fence all Kensei-assigned skills via curator pin
# Extracts every skill from all profile configs (root + profiles/*/config.yaml)
# and pins them so the Hermes curator never archives without governance review.
#
# Idempotent — pinning an already-pinned skill is a no-op.
# Cron: daily 06:00, no_agent=true, delivers to #governance

# Env overrides (P13 isolation / local disposable runs):
#   PIN_DRY_RUN=1 — print the skills that would be pinned, do not call
#                   `hermes curator pin`, do not touch the lockfile
#   HERMES_HOME    — override the .hermes root (default $HOME/.hermes)
#                   so a temp home is used instead of /home/kensei/.hermes
set -uo pipefail
# NOTE: errexit (set -e) intentionally OFF so the dry-run path and the
# auth-failure path can print diagnostics without aborting early.

DRY_RUN=0
if [ "${PIN_DRY_RUN:-0}" = "1" ]; then DRY_RUN=1; fi

# ── Safety: require Bash 4+ for associative arrays ──
if [ -z "${BASH_VERSINFO:-}" ] || [ "${BASH_VERSINFO:-0}" -lt 4 ]; then
    echo "ERROR: Requires Bash 4+ for associative arrays" >&2
    exit 1
fi

# ── PID lockfile to prevent concurrent runs ──
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
LOCKFILE="$HERMES_HOME/.pin_assigned_skills.lock"
if [ "$DRY_RUN" = "1" ]; then
    : # dry-run: never touch the lockfile
elif [ -f "$LOCKFILE" ]; then
    OLD_PID=$(cat "$LOCKFILE" 2>/dev/null)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "WARNING: Another pin instance is running (PID $OLD_PID). Exiting." >&2
        exit 0
    fi
    rm -f "$LOCKFILE"
    echo $$ > "$LOCKFILE"
    trap 'rm -f "$LOCKFILE"' EXIT
fi

BASE="$HERMES_HOME"
SKILL_NAME_RE='^[A-Za-z0-9._-]+$'

# Problem lines are buffered; routine successes are NOT printed so a clean run
# stays silent (the delivery envelope suppresses empty/[SILENT] output).
declare -a PROBLEMS=()

# ── Helper: extract skill names from a config YAML via Python ──
# Prints one skill per line to stdout; errors to stderr (not mixed)
extract_skills() {
    local cfg_path="$1"
    local label="${2:-root}"
    python3 -c "
import yaml, json, sys
try:
    with open('$cfg_path') as f:
        c = yaml.safe_load(f) or {}
    sb = c.get('skills', {}) or {}
    for key in ['always_skills', 'enabled_skills']:
        val = sb.get(key, [])
        if isinstance(val, str):
            try: val = json.loads(val)
            except: val = []
        for s in (val or []):
            s = str(s).strip()
            if s and not s.startswith('/'):
                print(s)
except Exception as e:
    print(f'ERROR:{label}: {e}', file=sys.stderr)
" 2>/dev/null  # stderr kept separate, not merged
}

# ── Collect all skill names ──
declare -A ALL_SKILLS
TOTAL=0

# Root config
while IFS= read -r s; do
    [ -n "$s" ] || continue
    ALL_SKILLS["$s"]=1
    TOTAL=$((TOTAL + 1))
done < <(extract_skills "$BASE/config.yaml" "root")

# Profile configs
shopt -s nullglob
for PROFILE_DIR in "$BASE"/profiles/*/; do
    PROFILE=$(basename "$PROFILE_DIR")
    [[ "$PROFILE" == _* || "$PROFILE" == .* ]] && continue
    CFG="$PROFILE_DIR/config.yaml"
    [ -f "$CFG" ] || continue

    while IFS= read -r s; do
        [ -n "$s" ] || continue
        ALL_SKILLS["$s"]=1
        TOTAL=$((TOTAL + 1))
    done < <(extract_skills "$CFG" "$PROFILE")
done
shopt -u nullglob

UNIQUE=${#ALL_SKILLS[@]}

# ── Pin each skill ──
PINNED=0
SKIPPED=0
FAILED=0

for SKILL in "${!ALL_SKILLS[@]}"; do
    # Validate skill name format
    if ! [[ "$SKILL" =~ $SKILL_NAME_RE ]]; then
        PROBLEMS+=("  ⚠️  $SKILL — invalid characters, skipping")
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Verify skill exists on disk
    SKILL_PATH=$(find "$BASE/skills" -path "*/${SKILL}/SKILL.md" \
        -not -path '*/_archived/*' 2>/dev/null | head -1)
    if [ -z "$SKILL_PATH" ]; then
        PROBLEMS+=("  ⚠️  $SKILL — not found on disk (stale reference)")
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Pin via hermes curator — check exit code, not output text
    if [ "$DRY_RUN" = "1" ]; then
        echo "dry-run: would pin $SKILL"
        PINNED=$((PINNED + 1))
    elif hermes curator pin "$SKILL" > /dev/null 2>&1; then
        PINNED=$((PINNED + 1))
    else
        FAILED=$((FAILED + 1))
        PROBLEMS+=("  ❌ $SKILL — pin failed (exit code non-zero)")
    fi
done

# Speak only when there is something to act on; a fully clean run is silent.
if [ "$FAILED" -gt 0 ]; then
    echo "🔴 Skill pinning · $FAILED failed, $SKIPPED skipped (of $UNIQUE)"
    printf '%s\n' "${PROBLEMS[@]}"
elif [ "$SKIPPED" -gt 0 ]; then
    echo "🟡 Skill pinning · $SKIPPED stale/invalid ref(s) of $UNIQUE ($PINNED pinned)"
    printf '%s\n' "${PROBLEMS[@]}"
fi
# else: silent — clean run, no output
