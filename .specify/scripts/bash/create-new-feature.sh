#!/usr/bin/env bash

set -e

JSON_MODE=false
DRY_RUN=false
ALLOW_EXISTING=false
SHORT_NAME=""
BRANCH_NUMBER=""
USE_TIMESTAMP=false
NUMBER_EXPLICIT=false
ARGS=()
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --json)
            JSON_MODE=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --allow-existing-branch)
            ALLOW_EXISTING=true
            ;;
        --short-name)
            if [ $((i + 1)) -gt $# ]; then
                echo 'Error: --short-name requires a value' >&2
                exit 1
            fi
            i=$((i + 1))
            next_arg="${!i}"
            # Check if the next argument is another option (starts with --)
            if [[ "$next_arg" == --* ]]; then
                echo 'Error: --short-name requires a value' >&2
                exit 1
            fi
            SHORT_NAME="$next_arg"
            ;;
        --number)
            if [ $((i + 1)) -gt $# ]; then
                echo 'Error: --number requires a value' >&2
                exit 1
            fi
            i=$((i + 1))
            next_arg="${!i}"
            if [[ "$next_arg" == --* ]]; then
                echo 'Error: --number requires a value' >&2
                exit 1
            fi
            BRANCH_NUMBER="$next_arg"
            if [ -n "$BRANCH_NUMBER" ]; then
                NUMBER_EXPLICIT=true
            fi
            ;;
        --timestamp)
            USE_TIMESTAMP=true
            ;;
        --help|-h)
            echo "Usage: $0 [--json] [--dry-run] [--allow-existing-branch] [--short-name <name>] [--number N] [--timestamp] <feature_description>"
            echo ""
            echo "Options:"
            echo "  --json              Output in JSON format"
            echo "  --dry-run           Compute feature name and paths without creating directories or files"
            echo "  --allow-existing-branch  Reuse an existing feature directory if it already exists"
            echo "  --short-name <name> Provide a custom short name (2-4 words) for the feature"
            echo "  --number N          Prefer a feature number (auto-corrected if its specs prefix exists)"
            echo "  --timestamp         Use timestamp prefix (YYYYMMDD-HHMMSS) instead of sequential numbering"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 'Add user authentication system' --short-name 'user-auth'"
            echo "  $0 'Implement OAuth2 integration for API' --number 5"
            echo "  $0 --timestamp --short-name 'user-auth' 'Add user authentication'"
            exit 0
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
    i=$((i + 1))
done

FEATURE_DESCRIPTION="${ARGS[*]}"
if [ -z "$FEATURE_DESCRIPTION" ]; then
    echo "Usage: $0 [--json] [--dry-run] [--allow-existing-branch] [--short-name <name>] [--number N] [--timestamp] <feature_description>" >&2
    exit 1
fi

# Trim whitespace and validate description is not empty (e.g., user passed only whitespace)
FEATURE_DESCRIPTION=$(echo "$FEATURE_DESCRIPTION" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')
if [ -z "$FEATURE_DESCRIPTION" ]; then
    echo "Error: Feature description cannot be empty or contain only whitespace" >&2
    exit 1
fi

MAX_FEATURE_NUMBER=9223372036854775807
MAX_BRANCH_LENGTH=244

is_feature_number_in_range() {
    local value="$1"
    local normalized="${value#"${value%%[!0]*}"}"
    [ -n "$normalized" ] || normalized=0
    [ ${#normalized} -lt ${#MAX_FEATURE_NUMBER} ] && return 0
    [ ${#normalized} -gt ${#MAX_FEATURE_NUMBER} ] && return 1
    # Equal-length digit strings must be compared without arithmetic overflow.
    # shellcheck disable=SC2071
    [[ "$normalized" < "$MAX_FEATURE_NUMBER" || "$normalized" == "$MAX_FEATURE_NUMBER" ]]
}

# Function to get highest number from specs directory
get_highest_from_specs() {
    local specs_dir="$1"
    local highest=0

    if [ -d "$specs_dir" ]; then
        for dir in "$specs_dir"/*; do
            [ -d "$dir" ] || continue
            dirname=$(basename "$dir")
            # Match sequential prefixes (>=3 digits), but skip timestamp dirs.
            if echo "$dirname" | grep -Eq '^[0-9]{3,}-' && ! echo "$dirname" | grep -Eq '^[0-9]{8}-[0-9]{6}-'; then
                number=$(echo "$dirname" | grep -Eo '^[0-9]+')
                if is_feature_number_in_range "$number"; then
                    number=$((10#$number))
                    if [ "$number" -gt "$highest" ]; then
                        highest=$number
                    fi
                fi
            fi
        done
    fi

    echo "$highest"
}

# Return success when a spec directory owns the given numeric prefix.
spec_prefix_exists() {
    local specs_dir="$1"
    local feature_num="$2"

    for spec_path in "$specs_dir/${feature_num}-"*; do
        [ -d "$spec_path" ] && return 0
    done
    return 1
}

# Function to clean and format a branch name
clean_branch_name() {
    local name="$1"
    echo "$name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | sed 's/^-//' | sed 's/-$//'
}

# Fit a feature prefix and suffix within GitHub's branch-name limit.
fit_branch_name() {
    local feature_num="$1"
    local branch_suffix="$2"
    local branch_name="${feature_num}-${branch_suffix}"

    if [ ${#branch_name} -gt $MAX_BRANCH_LENGTH ]; then
        local prefix_length=$(( ${#feature_num} + 1 ))
        local max_suffix_length=$((MAX_BRANCH_LENGTH - prefix_length))
        local truncated_suffix
        truncated_suffix=$(printf '%s' "$branch_suffix" | cut -c "1-$max_suffix_length" | sed 's/-$//')
        branch_name="${feature_num}-${truncated_suffix}"
    fi

    printf '%s' "$branch_name"
}

# Quote a value for POSIX shell reuse, byte-identical to Python's shlex.quote
# so the persistence hints match the Python variant exactly (printf %q output
# differs between bash versions and from shlex.quote for spaces/metachars).
shell_quote() {
    local value="$1" LC_ALL=C
    if [[ "$value" =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then
        printf '%s' "$value"
    else
        local q="'\"'\"'"
        printf "'%s'" "${value//\'/$q}"
    fi
}

# Resolve repository root using common.sh functions which prioritize .specify
SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

REPO_ROOT=$(get_repo_root) || exit 1

cd "$REPO_ROOT"

SPECS_DIR="$REPO_ROOT/specs"
if [ "$DRY_RUN" != true ]; then
    mkdir -p "$SPECS_DIR"
fi

# Function to generate branch name with stop word filtering and length filtering
generate_branch_name() {
    local description="$1"

    # Common stop words to filter out
    local stop_words="^(i|a|an|the|to|for|of|in|on|at|by|with|from|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|should|could|can|may|might|must|shall|this|that|these|those|my|your|our|their|want|need|add|get|set)$"

    # Convert to lowercase and split into words
    local clean_name=$(printf '%s' "$description" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/ /g')

    # Filter words: remove stop words and words shorter than 3 chars (unless they're uppercase acronyms in original)
    local meaningful_words=()
    for word in $clean_name; do
        # Skip empty words
        [ -z "$word" ] && continue

        # Keep words that are NOT stop words AND (length >= 3 OR are potential acronyms)
        if ! echo "$word" | grep -qiE "$stop_words"; then
            if [ ${#word} -ge 3 ]; then
                meaningful_words+=("$word")
            # Keep short words that appear as an uppercase acronym in the original.
            # Uppercase via tr and match with grep -w (both portable) rather than
            # bash's 4+ "^^" case expansion (breaks on macOS bash 3.2) and \b (non-POSIX).
            elif printf '%s' "$description" | grep -qw -- "$(printf '%s' "$word" | tr '[:lower:]' '[:upper:]')"; then
                meaningful_words+=("$word")
            fi
        fi
    done

    # If we have meaningful words, use first 3-4 of them
    if [ ${#meaningful_words[@]} -gt 0 ]; then
        local max_words=3
        if [ ${#meaningful_words[@]} -eq 4 ]; then max_words=4; fi

        local result=""
        local count=0
        for word in "${meaningful_words[@]}"; do
            if [ $count -ge $max_words ]; then break; fi
            if [ -n "$result" ]; then result="$result-"; fi
            result="$result$word"
            count=$((count + 1))
        done
        echo "$result"
    else
        # Fallback to original logic if no meaningful words found
        local cleaned=$(clean_branch_name "$description")
        echo "$cleaned" | tr '-' '\n' | grep -v '^$' | head -3 | tr '\n' '-' | sed 's/-$//'
    fi
}

# Generate branch name
if [ -n "$SHORT_NAME" ]; then
    # Use provided short name, just clean it up
    BRANCH_SUFFIX=$(clean_branch_name "$SHORT_NAME")
else
    # Generate from description with smart filtering
    BRANCH_SUFFIX=$(generate_branch_name "$FEATURE_DESCRIPTION")
fi

# Warn if --number and --timestamp are both specified
if [ "$USE_TIMESTAMP" = true ] && [ -n "$BRANCH_NUMBER" ]; then
    >&2 echo "[specify] Warning: --number is ignored when --timestamp is used"
    BRANCH_NUMBER=""
fi

# Determine branch prefix
if [ "$USE_TIMESTAMP" = true ]; then
    FEATURE_NUM=$(date +%Y%m%d-%H%M%S)
    BRANCH_NAME="${FEATURE_NUM}-${BRANCH_SUFFIX}"
else
    if [ -n "$BRANCH_NUMBER" ] && [[ ! "$BRANCH_NUMBER" =~ ^[0-9]+$ ]]; then
        echo "Error: --number must be an unsigned integer, got '$BRANCH_NUMBER'" >&2
        exit 1
    fi

    # Bash arithmetic is signed 64-bit; reject digit strings that would wrap.
    if [ -n "$BRANCH_NUMBER" ] && ! is_feature_number_in_range "$BRANCH_NUMBER"; then
        echo "Error: --number must be between 0 and $MAX_FEATURE_NUMBER, got '$BRANCH_NUMBER'" >&2
        exit 1
    fi

    # Determine branch number from existing feature directories
    if [ -z "$BRANCH_NUMBER" ]; then
        HIGHEST=$(get_highest_from_specs "$SPECS_DIR")
        if [ "$HIGHEST" -eq "$MAX_FEATURE_NUMBER" ]; then
            echo "Error: feature number must be between 0 and $MAX_FEATURE_NUMBER, got '9223372036854775808'" >&2
            exit 1
        fi
        BRANCH_NUMBER=$((HIGHEST + 1))
    fi

    # Force base-10 interpretation to prevent octal conversion (e.g., 010 → 8 in octal, but should be 10 in decimal)
    FEATURE_NUM=$(printf "%03d" "$((10#$BRANCH_NUMBER))")

    # Treat an explicit number as a preference when its prefix is already used
    # by a feature directory. Auto-detected numbers are already conflict-free.
    if [ "$NUMBER_EXPLICIT" = true ]; then
        SPEC_CONFLICT=false
        REQUESTED_BRANCH_NAME=$(fit_branch_name "$FEATURE_NUM" "$BRANCH_SUFFIX")
        REQUESTED_DIR="$SPECS_DIR/$REQUESTED_BRANCH_NAME"
        if [ "$ALLOW_EXISTING" != true ] || [ ! -d "$REQUESTED_DIR" ]; then
            spec_prefix_exists "$SPECS_DIR" "$FEATURE_NUM" && SPEC_CONFLICT=true
        fi

        if [ "$SPEC_CONFLICT" = true ]; then
            REQUESTED_NUM="$FEATURE_NUM"
            HIGHEST=$(get_highest_from_specs "$SPECS_DIR")
            BRANCH_NUMBER=$HIGHEST
            while true; do
                if [ "$BRANCH_NUMBER" -eq "$MAX_FEATURE_NUMBER" ]; then
                    echo "Error: feature number must be between 0 and $MAX_FEATURE_NUMBER, got '9223372036854775808'" >&2
                    exit 1
                fi
                BRANCH_NUMBER=$((BRANCH_NUMBER + 1))
                FEATURE_NUM=$(printf "%03d" "$((10#$BRANCH_NUMBER))")
                spec_prefix_exists "$SPECS_DIR" "$FEATURE_NUM" || break
            done
            >&2 echo "[specify] Warning: --number $REQUESTED_NUM conflicts with an existing spec directory; using $FEATURE_NUM instead"
        fi
    fi

fi

# GitHub enforces a 244-byte limit on branch names
# Validate and truncate if necessary
ORIGINAL_BRANCH_NAME="${FEATURE_NUM}-${BRANCH_SUFFIX}"
BRANCH_NAME=$(fit_branch_name "$FEATURE_NUM" "$BRANCH_SUFFIX")
if [ "$BRANCH_NAME" != "$ORIGINAL_BRANCH_NAME" ]; then
    >&2 echo "[specify] Warning: Branch name exceeded GitHub's 244-byte limit"
    >&2 echo "[specify] Original: $ORIGINAL_BRANCH_NAME (${#ORIGINAL_BRANCH_NAME} bytes)"
    >&2 echo "[specify] Truncated to: $BRANCH_NAME (${#BRANCH_NAME} bytes)"
fi

FEATURE_DIR="$SPECS_DIR/$BRANCH_NAME"
SPEC_FILE="$FEATURE_DIR/spec.md"

if [ "$DRY_RUN" != true ]; then
    if [ -d "$FEATURE_DIR" ] && [ "$ALLOW_EXISTING" != true ]; then
        if [ "$USE_TIMESTAMP" = true ]; then
            >&2 echo "Error: Feature directory '$FEATURE_DIR' already exists. Rerun to get a new timestamp or use a different --short-name."
        else
            >&2 echo "Error: Feature directory '$FEATURE_DIR' already exists. Please use a different feature name or specify a different number with --number."
        fi
        exit 1
    fi

    NEEDS_SPEC=false
    SPEC_TEMPLATE_FOUND=false
    SPEC_TEMPLATE_CONTENT=""
    if [ ! -f "$SPEC_FILE" ]; then
        NEEDS_SPEC=true
        if SPEC_TEMPLATE_CONTENT=$(resolve_template_content "spec-template" "$REPO_ROOT"; status=$?; printf x; exit "$status"); then
            SPEC_TEMPLATE_CONTENT="${SPEC_TEMPLATE_CONTENT%x}"
            SPEC_TEMPLATE_FOUND=true
        else
            resolve_status=$?
            if [ "$resolve_status" -ne 1 ]; then
                exit "$resolve_status"
            fi
        fi
    fi

    mkdir -p "$FEATURE_DIR"

    if [ "$NEEDS_SPEC" = true ]; then
        if [ "$SPEC_TEMPLATE_FOUND" = true ]; then
            printf '%s' "$SPEC_TEMPLATE_CONTENT" > "$SPEC_FILE"
        else
            echo "Warning: Spec template not found; created empty spec file" >&2
            touch "$SPEC_FILE"
        fi
    fi

    # Persist to .specify/feature.json so downstream commands can find the feature
    _persist_feature_json "$REPO_ROOT" "$FEATURE_DIR"

    # Inform the user how to set feature state in their own shell
    printf '# To persist: export SPECIFY_FEATURE=%s\n' "$(shell_quote "$BRANCH_NAME")" >&2
    printf '#              export SPECIFY_FEATURE_DIRECTORY=%s\n' "$(shell_quote "$FEATURE_DIR")" >&2
fi

if $JSON_MODE; then
    if command -v jq >/dev/null 2>&1; then
        if [ "$DRY_RUN" = true ]; then
            jq -cn \
                --arg branch_name "$BRANCH_NAME" \
                --arg spec_file "$SPEC_FILE" \
                --arg feature_num "$FEATURE_NUM" \
                '{BRANCH_NAME:$branch_name,SPEC_FILE:$spec_file,FEATURE_NUM:$feature_num,DRY_RUN:true}'
        else
            jq -cn \
                --arg branch_name "$BRANCH_NAME" \
                --arg spec_file "$SPEC_FILE" \
                --arg feature_num "$FEATURE_NUM" \
                '{BRANCH_NAME:$branch_name,SPEC_FILE:$spec_file,FEATURE_NUM:$feature_num}'
        fi
    else
        if [ "$DRY_RUN" = true ]; then
            printf '{"BRANCH_NAME":"%s","SPEC_FILE":"%s","FEATURE_NUM":"%s","DRY_RUN":true}\n' "$(json_escape "$BRANCH_NAME")" "$(json_escape "$SPEC_FILE")" "$(json_escape "$FEATURE_NUM")"
        else
            printf '{"BRANCH_NAME":"%s","SPEC_FILE":"%s","FEATURE_NUM":"%s"}\n' "$(json_escape "$BRANCH_NAME")" "$(json_escape "$SPEC_FILE")" "$(json_escape "$FEATURE_NUM")"
        fi
    fi
else
    echo "BRANCH_NAME: $BRANCH_NAME"
    echo "SPEC_FILE: $SPEC_FILE"
    echo "FEATURE_NUM: $FEATURE_NUM"
    if [ "$DRY_RUN" != true ]; then
        printf '# To persist in your shell: export SPECIFY_FEATURE=%s\n' "$(shell_quote "$BRANCH_NAME")"
        printf '#                           export SPECIFY_FEATURE_DIRECTORY=%s\n' "$(shell_quote "$FEATURE_DIR")"
    fi
fi
