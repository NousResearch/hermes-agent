#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
WORKFLOW_NAME="opencode-sdd-orchestrator"
DEFAULT_AGENT="sdd-orchestrator"
INVALID_AGENT_ALIAS="sdd-orquesador"

REPO_INPUT=""
REPO_ROOT=""
BASE_BRANCH=""
BASE_START_REF=""
CHANGE_TYPE="feat"
TASK_SLUG=""
AGENT_NAME="$DEFAULT_AGENT"
TASK_FILE_INPUT=""
TASK_FILE=""
PR_TITLE=""
PR_BODY_FILE_INPUT=""
PR_BODY_FILE=""
DRY_RUN=0
PUSH_CHANGES=0
CREATE_PR=0

WORKTREE_PATH=""
BRANCH_NAME=""
LOG_DIR=""
OPENCODE_LOG_FILE=""
VERIFICATION_LOG_FILE=""
VERIFICATION_STATUS="not-run"
VERIFICATION_RAN=0
OPENCODE_RAW_OUTPUT=""
PR_URL=""
SUMMARY_STATUS="BLOCKED"
SUMMARY_MESSAGE="Not started"

usage() {
    cat <<EOF
${SCRIPT_NAME} - Hermes/OpenCode SDD workflow helper

Usage:
  ${SCRIPT_NAME} --repo PATH --slug SLUG --task-file FILE [options]

Required:
  --repo PATH            Path inside the target git repository
  --slug SLUG            Stable slug used for branch/worktree naming
  --task-file FILE       Fully populated brief passed to OpenCode (not a raw template)

Options:
  --base BRANCH          Base branch to branch from (default: origin/HEAD or main)
  --type TYPE            Branch prefix/type (default: feat)
  --agent NAME           OpenCode agent name (must be: ${DEFAULT_AGENT})
  --push                 Push branch after OpenCode finishes
  --create-pr            Create a GitHub pull request
  --pr-title TITLE       Pull request title (required with --create-pr)
  --pr-body-file FILE    Pull request body file passed to gh pr create (required with --create-pr)
  --dry-run              Validate and print planned actions without mutating git/worktree/GitHub/OpenCode state
  -h, --help             Show this help message

Behavior:
  - Fails closed unless the exact OpenCode agent '${DEFAULT_AGENT}' is available.
  - Explicitly rejects '${INVALID_AGENT_ALIAS}' to avoid silent fallback to 'build'.
  - Must be started from the main repository checkout (not a linked worktree root).
  - Uses an isolated git worktree under <repo>/.worktrees/.
  - Expects --task-file to already be filled; this helper validates but does not auto-populate templates.
  - Relative --task-file/--pr-body-file paths resolve from the caller's current working directory.
  - Stores logs under \${HERMES_HOME:-\$HOME/.hermes}/logs/${WORKFLOW_NAME}/.
EOF
}

log_info() {
    printf '[INFO] %s\n' "$*" >&2
}

log_warn() {
    printf '[WARN] %s\n' "$*" >&2
}

log_error() {
    printf '[ERROR] %s\n' "$*" >&2
}

bool_string() {
    if [[ "$1" -eq 1 ]]; then
        printf 'true'
    else
        printf 'false'
    fi
}

print_summary() {
    printf 'SUMMARY_BEGIN\n'
    printf 'status=%q\n' "$SUMMARY_STATUS"
    printf 'message=%q\n' "$SUMMARY_MESSAGE"
    printf 'repo=%q\n' "$REPO_ROOT"
    printf 'base_branch=%q\n' "$BASE_BRANCH"
    printf 'branch=%q\n' "$BRANCH_NAME"
    printf 'worktree=%q\n' "$WORKTREE_PATH"
    printf 'agent=%q\n' "$AGENT_NAME"
    printf 'task_file=%q\n' "$TASK_FILE"
    printf 'push=%q\n' "$(bool_string "$PUSH_CHANGES")"
    printf 'create_pr=%q\n' "$(bool_string "$CREATE_PR")"
    printf 'dry_run=%q\n' "$(bool_string "$DRY_RUN")"
    printf 'pr_url=%q\n' "$PR_URL"
    printf 'log_dir=%q\n' "$LOG_DIR"
    printf 'opencode_log=%q\n' "$OPENCODE_LOG_FILE"
    printf 'verification_status=%q\n' "$VERIFICATION_STATUS"
    printf 'verification_log=%q\n' "$VERIFICATION_LOG_FILE"
    printf 'SUMMARY_END\n'
}

fail() {
    SUMMARY_STATUS="BLOCKED"
    SUMMARY_MESSAGE="$1"
    log_error "$1"
    print_summary
    exit 1
}

handle_unexpected_error() {
    local exit_code="$1"
    local line_number="$2"
    local command_text="$3"
    trap - ERR

    if [[ "$SUMMARY_STATUS" != "SUCCESS" ]]; then
        SUMMARY_STATUS="BLOCKED"
        SUMMARY_MESSAGE="Unhandled error (exit ${exit_code}) at line ${line_number}: ${command_text}"
    fi

    log_error "$SUMMARY_MESSAGE"
    print_summary
    exit "$exit_code"
}

trap 'handle_unexpected_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

require_command() {
    local name="$1"
    command -v "$name" >/dev/null 2>&1 || fail "Missing required command: $name"
}

resolve_existing_path() {
    local input="$1"
    if [[ -z "$input" ]]; then
        fail "Expected a file path but received an empty value"
    fi

    if [[ "$input" = /* ]]; then
        [[ -e "$input" ]] || fail "Path does not exist: $input"
        printf '%s\n' "$input"
        return 0
    fi

    [[ -e "$PWD/$input" ]] || fail "Path does not exist: $input"
    local dir
    dir="$(cd "$(dirname "$PWD/$input")" && pwd -P)"
    printf '%s/%s\n' "$dir" "$(basename "$input")"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo)
                [[ $# -ge 2 ]] || fail "Missing value for --repo"
                REPO_INPUT="$2"
                shift 2
                ;;
            --base)
                [[ $# -ge 2 ]] || fail "Missing value for --base"
                BASE_BRANCH="$2"
                shift 2
                ;;
            --type)
                [[ $# -ge 2 ]] || fail "Missing value for --type"
                CHANGE_TYPE="$2"
                shift 2
                ;;
            --slug)
                [[ $# -ge 2 ]] || fail "Missing value for --slug"
                TASK_SLUG="$2"
                shift 2
                ;;
            --agent)
                [[ $# -ge 2 ]] || fail "Missing value for --agent"
                AGENT_NAME="$2"
                shift 2
                ;;
            --task-file)
                [[ $# -ge 2 ]] || fail "Missing value for --task-file"
                TASK_FILE_INPUT="$2"
                shift 2
                ;;
            --pr-title)
                [[ $# -ge 2 ]] || fail "Missing value for --pr-title"
                PR_TITLE="$2"
                shift 2
                ;;
            --pr-body-file)
                [[ $# -ge 2 ]] || fail "Missing value for --pr-body-file"
                PR_BODY_FILE_INPUT="$2"
                shift 2
                ;;
            --push)
                PUSH_CHANGES=1
                shift
                ;;
            --create-pr)
                CREATE_PR=1
                shift
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                fail "Unknown option: $1"
                ;;
        esac
    done
}

validate_inputs() {
    [[ -n "$REPO_INPUT" ]] || fail "--repo is required"
    [[ -n "$TASK_SLUG" ]] || fail "--slug is required"
    [[ -n "$TASK_FILE_INPUT" ]] || fail "--task-file is required"

    if [[ ! "$TASK_SLUG" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
        fail "Invalid --slug '$TASK_SLUG'. Use lowercase letters, numbers, dots, underscores, or dashes."
    fi

    if [[ ! "$CHANGE_TYPE" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
        fail "Invalid --type '$CHANGE_TYPE'"
    fi

    if [[ "$AGENT_NAME" == "$INVALID_AGENT_ALIAS" ]]; then
        fail "Unsupported agent alias '$INVALID_AGENT_ALIAS'. Use the exact agent '$DEFAULT_AGENT' to avoid silent fallback to build."
    fi

    if [[ "$AGENT_NAME" != "$DEFAULT_AGENT" ]]; then
        fail "Unsupported agent '$AGENT_NAME'. This workflow only allows '$DEFAULT_AGENT'."
    fi

    if [[ "$CREATE_PR" -eq 1 ]]; then
        PUSH_CHANGES=1
        [[ -n "$PR_TITLE" ]] || fail "--pr-title is required when --create-pr is set"
        [[ -n "$PR_BODY_FILE_INPUT" ]] || fail "--pr-body-file is required when --create-pr is set"
    fi
}

resolve_paths() {
    REPO_ROOT="$(git -C "$REPO_INPUT" rev-parse --show-toplevel 2>/dev/null)" || fail "Unable to resolve git repository from --repo '$REPO_INPUT'"
    TASK_FILE="$(resolve_existing_path "$TASK_FILE_INPUT")"

    if [[ -n "$PR_BODY_FILE_INPUT" ]]; then
        PR_BODY_FILE="$(resolve_existing_path "$PR_BODY_FILE_INPUT")"
    fi

    BRANCH_NAME="${CHANGE_TYPE}/${TASK_SLUG}"
    WORKTREE_PATH="$REPO_ROOT/.worktrees/${CHANGE_TYPE}-${TASK_SLUG}"
}

init_log_paths() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        LOG_DIR=""
        OPENCODE_LOG_FILE=""
        VERIFICATION_LOG_FILE=""
        return 0
    fi

    local timestamp
    timestamp="$(date +%Y%m%d-%H%M%S)"
    LOG_DIR="$HERMES_HOME/logs/${WORKFLOW_NAME}/${TASK_SLUG}/${timestamp}"
    mkdir -p "$LOG_DIR"
    OPENCODE_LOG_FILE="$LOG_DIR/opencode-run.log"
    VERIFICATION_LOG_FILE="$LOG_DIR/verification.log"
}

detect_base_branch() {
    if [[ -n "$BASE_BRANCH" ]]; then
        return 0
    fi

    local origin_head
    origin_head="$(git -C "$REPO_ROOT" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null || true)"
    origin_head="${origin_head#origin/}"
    if [[ -n "$origin_head" ]]; then
        BASE_BRANCH="$origin_head"
    else
        BASE_BRANCH="main"
    fi
}

validate_base_branch() {
    if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/remotes/origin/${BASE_BRANCH}"; then
        BASE_START_REF="origin/${BASE_BRANCH}"
        return 0
    fi

    if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/${BASE_BRANCH}"; then
        BASE_START_REF="${BASE_BRANCH}"
        return 0
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        BASE_START_REF="origin/${BASE_BRANCH}"
        log_warn "Base branch '${BASE_BRANCH}' was not found locally; dry-run skips fetch validation"
        return 0
    fi

    log_info "Fetching origin/${BASE_BRANCH}"
    git -C "$REPO_ROOT" fetch origin "$BASE_BRANCH" >/dev/null 2>&1 || fail "Unable to fetch base branch 'origin/${BASE_BRANCH}'"
    BASE_START_REF="origin/${BASE_BRANCH}"
    git -C "$REPO_ROOT" show-ref --verify --quiet "refs/remotes/origin/${BASE_BRANCH}" || fail "Base branch 'origin/${BASE_BRANCH}' is unavailable"
}

validate_agent() {
    require_command opencode

    local agent_list
    agent_list="$(opencode agent list 2>&1)" || fail "Unable to list OpenCode agents"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        printf '%s\n' "$agent_list" >"$LOG_DIR/opencode-agent-list.log"
    fi

    if ! printf '%s\n' "$agent_list" | grep -Eq '(^|[^[:alnum:]_-])sdd-orchestrator([^[:alnum:]_-]|$)'; then
        fail "OpenCode agent '$DEFAULT_AGENT' is not available. Refusing to continue because fallback would be unsafe."
    fi

    if printf '%s\n' "$agent_list" | grep -Eq '(^|[^[:alnum:]_-])sdd-orquesador([^[:alnum:]_-]|$)'; then
        log_warn "Agent list unexpectedly contains '$INVALID_AGENT_ALIAS'; this helper still pins '$DEFAULT_AGENT' exactly."
    fi
}

validate_prerequisites() {
    require_command git
    git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "Resolved repository is not a git worktree: $REPO_ROOT"

    if [[ -f "$REPO_ROOT/.git" ]]; then
        fail "Resolved repository root '$REPO_ROOT' is a linked git worktree root (.git is a file). Start this helper from the main repository checkout only."
    fi

    if [[ ! -d "$REPO_ROOT/.git" ]]; then
        fail "Resolved repository root '$REPO_ROOT' does not contain a main-checkout .git directory. Start this helper from the main repository checkout only."
    fi

    local repo_status
    repo_status="$(git -C "$REPO_ROOT" status --porcelain)"
    if [[ -n "$repo_status" ]]; then
        fail "Repository has uncommitted changes. Commit/stash first to keep orchestration fail-closed."
    fi

    local git_dir
    git_dir="$(git -C "$REPO_ROOT" rev-parse --git-dir)"
    for in_progress_path in "MERGE_HEAD" "CHERRY_PICK_HEAD" "REVERT_HEAD" "REBASE_HEAD" "rebase-apply" "rebase-merge"; do
        if [[ -e "$git_dir/$in_progress_path" ]]; then
            fail "Repository has an in-progress git operation ($in_progress_path). Resolve it before running this helper."
        fi
    done

    if [[ "$CREATE_PR" -eq 1 ]]; then
        require_command gh
        gh auth status >/dev/null 2>&1 || fail "GitHub CLI auth is required for PR creation"
    fi
}

validate_worktree_clean_and_idle() {
    local worktree_path="$1"
    local context_label="$2"

    git -C "$worktree_path" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "${context_label} is not a git worktree: $worktree_path"

    local worktree_status
    worktree_status="$(git -C "$worktree_path" status --porcelain)"
    if [[ -n "$worktree_status" ]]; then
        fail "${context_label} has uncommitted changes. Commit/stash/reset them before continuing."
    fi

    local git_dir
    git_dir="$(git -C "$worktree_path" rev-parse --git-dir)"
    for in_progress_path in "MERGE_HEAD" "CHERRY_PICK_HEAD" "REVERT_HEAD" "REBASE_HEAD" "rebase-apply" "rebase-merge"; do
        if [[ -e "$git_dir/$in_progress_path" ]]; then
            fail "${context_label} has an in-progress git operation ($in_progress_path). Resolve it before continuing."
        fi
    done
}

extract_verification_commands() {
    awk '
        BEGIN { in_verification = 0; in_block = 0 }
        /^##[[:space:]]+Verification commands/ { in_verification = 1; next }
        in_verification && /^##[[:space:]]+/ { if (!in_block) exit; in_verification = 0 }
        in_verification && /^```bash[[:space:]]*$/ { in_block = 1; next }
        in_verification && in_block && /^```[[:space:]]*$/ { exit }
        in_verification && in_block {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
            if (length($0) > 0) print $0
        }
    ' "$TASK_FILE"
}

validate_task_brief() {
    grep -Eq '^##[[:space:]]+Problem statement' "$TASK_FILE" || fail "Task brief must include a '## Problem statement' section"
    grep -Eq '^##[[:space:]]+In scope' "$TASK_FILE" || fail "Task brief must include a '## In scope' section"
    grep -Eq '^##[[:space:]]+Out of scope' "$TASK_FILE" || fail "Task brief must include a '## Out of scope' section"
    grep -Eq '^##[[:space:]]+Acceptance criteria' "$TASK_FILE" || fail "Task brief must include an '## Acceptance criteria' section"
    grep -Eq '^##[[:space:]]+Verification commands' "$TASK_FILE" || fail "Task brief must include a '## Verification commands' section"
    grep -Eq '^##[[:space:]]+File constraints' "$TASK_FILE" || fail "Task brief must include a '## File constraints' section"
    grep -Eq '^##[[:space:]]+Required final output format' "$TASK_FILE" || fail "Task brief must include a '## Required final output format' section"

    if grep -Eq '<[^>]+>' "$TASK_FILE"; then
        fail "Task brief still contains template placeholders (<...>). Provide a populated brief file."
    fi

    local criteria_count
    criteria_count="$(awk '/^- \[ \] .+/ {count++} END {print count + 0}' "$TASK_FILE")"
    if [[ "$criteria_count" -eq 0 ]]; then
        fail "Task brief must contain at least one explicit acceptance criterion checklist item"
    fi

    local verification_count
    verification_count="$(extract_verification_commands | wc -l | tr -d '[:space:]')"
    if [[ "$verification_count" -eq 0 ]]; then
        fail "Task brief must define at least one explicit verification command in the bash block"
    fi
}

validate_pr_body_file() {
    if [[ -z "$PR_BODY_FILE" ]]; then
        return 0
    fi

    if ! grep -Eq '[^[:space:]]' "$PR_BODY_FILE"; then
        fail "PR body file is empty or whitespace-only. Provide a populated PR body before continuing."
    fi

    if grep -Eq '<[^>]+>' "$PR_BODY_FILE"; then
        fail "PR body file still contains template placeholders (<...>). Provide a populated PR body before continuing."
    fi

    local required_pr_headers=(
        '^##[[:space:]]+Summary'
        '^##[[:space:]]+Why'
        '^##[[:space:]]+Orchestration details'
        '^##[[:space:]]+Scope'
        '^###[[:space:]]+In scope'
        '^###[[:space:]]+Out of scope'
        '^##[[:space:]]+Validation'
        '^##[[:space:]]+Acceptance criteria'
        '^##[[:space:]]+Risks / Follow-ups'
    )

    local header_pattern
    for header_pattern in "${required_pr_headers[@]}"; do
        if ! grep -Eq "$header_pattern" "$PR_BODY_FILE"; then
            fail "PR body file is missing a required template section header. Provide a fully populated PR body before continuing."
        fi
    done
}

ensure_worktree() {
    if [[ "$DRY_RUN" -eq 0 ]]; then
        mkdir -p "$REPO_ROOT/.worktrees"
    fi

    if [[ -e "$WORKTREE_PATH" ]]; then
        fail "Target worktree path already exists: $WORKTREE_PATH. Refusing to reuse existing worktrees."
    fi

    if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
        fail "Target branch already exists locally: $BRANCH_NAME. Refusing to reuse existing branches."
    fi

    if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/remotes/origin/${BRANCH_NAME}"; then
        fail "Target branch already exists on origin: $BRANCH_NAME. Refusing to reuse existing branches."
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_info "[dry-run] Would create worktree '$WORKTREE_PATH' on branch '$BRANCH_NAME' from '$BASE_START_REF'"
        return 0
    fi

    log_info "Creating worktree '$WORKTREE_PATH' from $BASE_START_REF"
    git -C "$REPO_ROOT" worktree add "$WORKTREE_PATH" -b "$BRANCH_NAME" "$BASE_START_REF" >/dev/null
}

run_opencode() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_info "[dry-run] Would run opencode in '$WORKTREE_PATH' with agent '$AGENT_NAME' and task file '$TASK_FILE'"
        return 0
    fi

    local output
    if ! output="$(cd "$WORKTREE_PATH" && opencode run "Implement the attached brief exactly. Return changed files + validations and acceptance criteria evidence." --agent "$AGENT_NAME" --file "$TASK_FILE" --format json 2>&1)"; then
        printf '%s\n' "$output" >"$OPENCODE_LOG_FILE"
        fail "OpenCode execution failed. Inspect $OPENCODE_LOG_FILE"
    fi

    OPENCODE_RAW_OUTPUT="$output"
    printf '%s\n' "$output" >"$OPENCODE_LOG_FILE"

    if printf '%s\n' "$output" | grep -Eiq 'unknown agent|falling back|fallback|default agent|requested agent unavailable'; then
        fail "Aborted: requested agent unavailable or OpenCode fallback detected. Inspect $OPENCODE_LOG_FILE"
    fi

    if ! printf '%s\n' "$output" | grep -Eq '"agent"[[:space:]]*:[[:space:]]*"sdd-orchestrator"'; then
        fail "Aborted: requested agent unavailable or OpenCode fallback detected (missing exact agent metadata). Inspect $OPENCODE_LOG_FILE"
    fi

    if printf '%s\n' "$output" | grep -Eq '"agent"[[:space:]]*:[[:space:]]*"build"'; then
        fail "Aborted: OpenCode reported the build agent instead of '$DEFAULT_AGENT'. Inspect $OPENCODE_LOG_FILE"
    fi
}

verify_opencode_evidence_contract() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_info "[dry-run] Would verify OpenCode output includes verification results and acceptance criteria evidence"
        return 0
    fi

    if ! printf '%s\n' "$OPENCODE_RAW_OUTPUT" | grep -Eiq 'files changed'; then
        fail "OpenCode output is missing 'Files changed' evidence. Inspect $OPENCODE_LOG_FILE"
    fi

    if ! printf '%s\n' "$OPENCODE_RAW_OUTPUT" | grep -Eiq 'verification results'; then
        fail "OpenCode output is missing 'Verification results' evidence. Inspect $OPENCODE_LOG_FILE"
    fi

    if ! printf '%s\n' "$OPENCODE_RAW_OUTPUT" | grep -Eiq 'exit status|exit code'; then
        fail "OpenCode output is missing explicit verification exit evidence. Inspect $OPENCODE_LOG_FILE"
    fi

    if ! printf '%s\n' "$OPENCODE_RAW_OUTPUT" | grep -Eiq 'acceptance criteria mapping'; then
        fail "OpenCode output is missing 'Acceptance criteria mapping' evidence. Inspect $OPENCODE_LOG_FILE"
    fi
}

run_verification_phase() {
    mapfile -t verification_commands < <(extract_verification_commands)
    if [[ "${#verification_commands[@]}" -eq 0 ]]; then
        fail "No verification commands found in task brief"
    fi

    VERIFICATION_RAN=1

    if [[ "$DRY_RUN" -eq 1 ]]; then
        local idx
        for idx in "${!verification_commands[@]}"; do
            log_info "[dry-run] Would run verification command $((idx + 1)): ${verification_commands[$idx]}"
        done
        VERIFICATION_STATUS="dry-run-not-executed"
        return 0
    fi

    : >"$VERIFICATION_LOG_FILE"
    local idx=0
    local cmd
    for cmd in "${verification_commands[@]}"; do
        idx=$((idx + 1))
        log_info "Running verification command ${idx}/${#verification_commands[@]}: $cmd"
        printf '### Command %d\n%s\n' "$idx" "$cmd" >>"$VERIFICATION_LOG_FILE"

        if (cd "$WORKTREE_PATH" && bash -o pipefail -lc "$cmd") >>"$VERIFICATION_LOG_FILE" 2>&1; then
            printf 'exit_code=0\n\n' >>"$VERIFICATION_LOG_FILE"
        else
            local cmd_exit=$?
            printf 'exit_code=%d\n\n' "$cmd_exit" >>"$VERIFICATION_LOG_FILE"
            VERIFICATION_STATUS="failed"
            fail "Verification command failed (exit $cmd_exit): $cmd. Inspect $VERIFICATION_LOG_FILE"
        fi
    done

    VERIFICATION_STATUS="passed"
}

ensure_commits_exist() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_info "[dry-run] Would verify commits exist on '$BRANCH_NAME' relative to '$BASE_START_REF'"
        return 0
    fi

    local commit_count
    commit_count="$(git -C "$WORKTREE_PATH" rev-list --count "${BASE_START_REF}..HEAD")"
    if [[ "$commit_count" -eq 0 ]]; then
        fail "No commits found on '$BRANCH_NAME' relative to '$BASE_START_REF'; refusing to push or create a PR"
    fi
}

validate_post_run_worktree_state() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_info "[dry-run] Would verify worktree '$WORKTREE_PATH' is clean before push/PR success"
        return 0
    fi

    validate_worktree_clean_and_idle "$WORKTREE_PATH" "Worktree '$WORKTREE_PATH' after OpenCode/verification"
}

push_branch() {
    if [[ "$PUSH_CHANGES" -eq 0 ]]; then
        return 0
    fi

    ensure_commits_exist

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_info "[dry-run] Would push '$BRANCH_NAME' to origin"
        return 0
    fi

    log_info "Pushing branch '$BRANCH_NAME'"
    git -C "$WORKTREE_PATH" push -u origin "$BRANCH_NAME"
}

create_or_reuse_pr() {
    if [[ "$CREATE_PR" -eq 0 ]]; then
        return 0
    fi

    [[ "$VERIFICATION_RAN" -eq 1 ]] || fail "Refusing to create PR before verification phase runs"
    if [[ "$DRY_RUN" -eq 0 && "$VERIFICATION_STATUS" != "passed" ]]; then
        fail "Refusing to create PR because verification did not pass"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_info "[dry-run] Would create PR for '$BRANCH_NAME' against '$BASE_BRANCH'"
        return 0
    fi

    local create_cmd=(gh pr create --base "$BASE_BRANCH" --head "$BRANCH_NAME" --title "$PR_TITLE")
    if [[ -n "$PR_BODY_FILE" ]]; then
        create_cmd+=(--body-file "$PR_BODY_FILE")
    fi

    log_info "Creating pull request"
    local pr_create_output
    if ! pr_create_output="$(cd "$WORKTREE_PATH" && "${create_cmd[@]}" 2>&1)"; then
        if printf '%s\n' "$pr_create_output" | grep -Eiq 'already exists|pull request.*exists|a pull request for branch'; then
            fail "A pull request already exists for '$BRANCH_NAME'. Update or close the existing PR explicitly before rerunning. gh output: $pr_create_output"
        fi
        fail "gh pr create failed for '$BRANCH_NAME'. gh output: $pr_create_output"
    fi

    PR_URL="$pr_create_output"
}

main() {
    parse_args "$@"
    validate_inputs
    resolve_paths
    validate_prerequisites
    validate_task_brief
    validate_pr_body_file
    detect_base_branch
    validate_base_branch
    init_log_paths
    validate_agent
    ensure_worktree
    run_opencode
    verify_opencode_evidence_contract
    run_verification_phase
    validate_post_run_worktree_state
    push_branch
    create_or_reuse_pr

    SUMMARY_STATUS="SUCCESS"
    SUMMARY_MESSAGE="Workflow completed"
    print_summary
}

main "$@"
