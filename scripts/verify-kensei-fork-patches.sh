#!/usr/bin/env bash
# ================================================================
# KENSEI Fork Patches - Integrity Verification Script
# ================================================================
# Reads ~/.hermes/kensei/fork-patches.yaml and checks every
# customisation point against the current codebase.
#
# Exit 0: all checks pass (cron output suppressed)
# Exit 1: one or more checks fail → summary printed to stdout
#
# Cron: daily 08:30, silent when healthy
# Manual: run after every upstream merge
# ================================================================
set -euo pipefail

# no_agent cron contract: empty stdout on success = silent delivery.
# Buffer all stdout; only emit if there are failures.
_OUTPUT_BUFFER=$(mktemp)
exec 3>&1 4>&2
exec 1>"$_OUTPUT_BUFFER" 2>&4

MANIFEST="$HOME/.hermes/kensei/fork-patches.yaml"
REPO="$HOME/repos/KenseiAgent"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
PASS="${GREEN}PASS${NC}"
FAIL="${RED}FAIL${NC}"

declare -i TOTAL=0 PASSED=0 FAILED=0
FAILURES=""

# ── Check functions ──

check_grep_min() {
    local file="$1" pattern="$2" min="$3"
    local count
    count=$(grep -cE "$pattern" "$REPO/$file" 2>/dev/null || echo "0")
    if [ "$count" -ge "$min" ] 2>/dev/null; then
        return 0
    else
        echo "    expected >=$min matches, found $count"
        return 1
    fi
}

check_file_contains() {
    local file="$1" pattern="$2"
    file="${file/#~\//$HOME/}"
    if grep -qF "$pattern" "$file" 2>/dev/null; then
        return 0
    else
        echo "    pattern '$pattern' not found in $file"
        return 1
    fi
}

check_file_exists() {
    local path="$1"
    path="${path/#~\//$HOME/}"
    if [ -e "$path" ]; then
        return 0
    else
        echo "    $path does not exist"
        return 1
    fi
}

check_symlink_exists() {
    local path="$1" pattern="$2"
    path="${path/#~\//$HOME/}"
    if [ -L "$path" ]; then
        local target
        target=$(readlink "$path" 2>/dev/null || echo "")
        if [ -n "$target" ] && echo "$target" | grep -qF "$pattern"; then
            return 0
        else
            echo "    symlink points to '$target' (expected containing '$pattern')"
            return 1
        fi
    else
        echo "    $path is not a symlink or does not exist"
        return 1
    fi
}

check_systemd_contains() {
    local service="$1" pattern="$2"
    local output
    output=$(systemctl cat "$service" 2>/dev/null)
    # P15 moved gateways from the mutable primary checkout to the canonical
    # candidate runtime.  Preserve the invariant (a KenseiAgent venv is used)
    # without pinning a retired checkout path.
    if [[ "$pattern" == "KenseiAgent/.venv" ]]; then
        if grep -qE 'KenseiAgent(-p15-candidate)?/\.venv' <<<"$output"; then
            return 0
        fi
    elif grep -qF "$pattern" <<<"$output"; then
        return 0
    fi
    echo "    '$pattern' not found in $service ExecStart"
    return 1
}

check_systemd_min() {
    local pattern="$1" min="$2"
    local count
    count=$(systemctl list-units --type=service --state=running 2>/dev/null | grep -cE "$pattern" || echo "0")
    if [ "$count" -ge "$min" ] 2>/dev/null; then
        return 0
    else
        echo "    found $count running services, minimum $min"
        return 1
    fi
}

check_dir_count() {
    local dir="$1" pattern="$2" min="$3"
    dir="${dir/#~\//$HOME/}"
    local count
    count=$(find "$dir" -name "$pattern" -maxdepth 2 2>/dev/null | wc -l)
    if [ "$count" -ge "$min" ] 2>/dev/null; then
        return 0
    else
        echo "    found $count '$pattern' files, minimum $min"
        return 1
    fi
}

# ── Run a single check ──
run_check() {
    local name="$1" check="$2" file="$3" path="$4" pattern="$5" min="$6" service="$7" desc="$8"
    
    TOTAL=$((TOTAL + 1))
    printf "%-45s " "  $name"
    
    local rc=0
    local extra=""
    
    case "$check" in
        grep_min)
            extra=$(check_grep_min "$file" "$pattern" "$min" 2>&1) || rc=$?
            ;;
        file_contains)
            extra=$(check_file_contains "$file" "$pattern" 2>&1) || rc=$?
            ;;
        file_exists)
            extra=$(check_file_exists "$path" 2>&1) || rc=$?
            ;;
        symlink_exists)
            extra=$(check_symlink_exists "$path" "$pattern" 2>&1) || rc=$?
            ;;
        systemd_contains)
            extra=$(check_systemd_contains "$service" "$pattern" 2>&1) || rc=$?
            ;;
        systemd_min)
            extra=$(check_systemd_min "$pattern" "$min" 2>&1) || rc=$?
            ;;
        dir_count)
            extra=$(check_dir_count "$path" "$pattern" "$min" 2>&1) || rc=$?
            ;;
        *)
            echo -e "${YELLOW}SKIP${NC} (unknown: $check)"
            return
            ;;
    esac
    
    if [ "$rc" -eq 0 ]; then
        echo -e "$PASS"
        PASSED=$((PASSED + 1))
    else
        echo -e "$FAIL"
        [ -n "$extra" ] && echo "$extra"
        FAILED=$((FAILED + 1))
        FAILURES+="  ${RED}$name${NC} - $desc"$'\n'
    fi
}

# ── Main ──

if [ ! -f "$MANIFEST" ]; then
    echo -e "${RED}ERROR: Manifest not found at $MANIFEST${NC}"
    exit 1
fi

echo "=== KENSEI Fork Patches - Integrity Check ==="
echo "Manifest: $MANIFEST"
echo "Repo:     $REPO"
echo ""

# Parse YAML - extract blocks with simple line-based parsing
current_name=""
current_file=""
current_path=""
current_check=""
current_pattern=""
current_min=""
current_service=""
current_desc=""

while IFS= read -r line; do
    # New check block
    if echo "$line" | grep -qE '^  - name:'; then
        # Run previous check
        if [ -n "$current_name" ] && [ -n "$current_check" ]; then
            run_check "$current_name" "$current_check" "$current_file" "$current_path" \
                      "$current_pattern" "$current_min" "$current_service" "$current_desc"
        fi
        
        current_name=$(echo "$line" | sed 's/^  - name: //')
        current_file=""; current_path=""; current_check=""
        current_pattern=""; current_min=""; current_service=""; current_desc=""
    fi
    
    case "$line" in
        *"file:"*)      current_file=$(echo "$line" | sed 's/.*file: //') ;;
        *"path:"*)      current_path=$(echo "$line" | sed 's/.*path: //') ;;
        *"check:"*)     current_check=$(echo "$line" | sed 's/.*check: //') ;;
        *"pattern:"*)
            p=$(echo "$line" | sed 's/.*pattern: //')
            p="${p#\"}"; p="${p%\"}"
            current_pattern="$p"
            ;;
        *"min:"*)       current_min=$(echo "$line" | sed 's/.*min: //') ;;
        *"service:"*)   current_service=$(echo "$line" | sed 's/.*service: //') ;;
        *"desc:"*)
            d=$(echo "$line" | sed 's/.*desc: //')
            d="${d#\"}"; d="${d%\"}"
            current_desc="$d"
            ;;
    esac
done < "$MANIFEST"

# Run final block
if [ -n "$current_name" ] && [ -n "$current_check" ]; then
    run_check "$current_name" "$current_check" "$current_file" "$current_path" \
              "$current_pattern" "$current_min" "$current_service" "$current_desc"
fi

# ── Summary ──
# no_agent cron contract: empty stdout on success = silent delivery.
# Only emit buffered stdout when there are failures.
if [ "$FAILED" -gt 0 ]; then
    # Restore stdout and print the full buffer
    exec 1>&3 2>&4
    cat "$_OUTPUT_BUFFER"
    echo ""
    echo -e "${RED}=== FAILURES ===${NC}"
    echo -e "$FAILURES"
    echo ""
    echo -e "${YELLOW}Restore reference: ${NC}~/.hermes/skills/devops/hermes-update/references/known-fork-patches.md"
    echo -e "${YELLOW}Add new check:     ${NC}$MANIFEST"
    rm -f "$_OUTPUT_BUFFER"
    exit 1
fi

# Success — silent (log to stderr only)
echo "All $PASSED fork customisations intact." >&4
rm -f "$_OUTPUT_BUFFER"
exit 0
