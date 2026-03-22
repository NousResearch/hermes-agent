#!/usr/bin/env bash
#
# fork_push.sh - Automate forking, pushing, and creating PRs for hermes-agent
#
# This script handles the common workflow when you have commits that can't be
# pushed to the upstream repository due to lack of write access. It:
#   1. Creates a fork on GitHub (via gh CLI)
#   2. Adds your fork as a remote named "fork"
#   3. Pushes your branch to the fork
#   4. Creates a pull request to upstream
#
# Usage:
#   ./scripts/fork_push.sh [OPTIONS]
#
# Options:
#   -b, --branch BRANCH    Branch to push (default: current branch)
#   -t, --title TITLE      PR title (default: auto-generated from commits)
#   -d, --draft            Create a draft PR
#   -n, --dry-run          Show what would be done without executing
#   -f, --force            Force push to fork (overwrite remote history)
#   -r, --remote NAME      Name for fork remote (default: fork)
#   -h, --help             Show this help message
#
# Prerequisites:
#   - GitHub CLI (gh) installed and authenticated: https://cli.github.com/
#   - Git configured with your identity
#
# Examples:
#   # Basic usage - push current branch and create PR
#   ./scripts/fork_push.sh
#
#   # Push specific branch with custom PR title
#   ./scripts/fork_push.sh -b feature/my-feature -t "Add new feature"
#
#   # Create a draft PR
#   ./scripts/fork_push.sh --draft
#
#   # Preview without making changes
#   ./scripts/fork_push.sh --dry-run
#

set -e

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

UPSTREAM_REPO="NousResearch/hermes-agent"
FORK_REMOTE_NAME="fork"
DEFAULT_BASE_BRANCH="main"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

show_help() {
    sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# //'
    exit 0
}

check_prerequisites() {
    local errors=0
    
    # Check for gh CLI
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) is not installed."
        echo ""
        echo "Install instructions:"
        echo "  macOS:   brew install gh"
        echo "  Linux:   See https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
        echo "  Windows: winget install GitHub.cli"
        echo "           or: scoop install gh"
        echo "           or: choco install gh"
        echo ""
        errors=$((errors + 1))
    else
        # Check authentication
        if ! gh auth status &> /dev/null; then
            log_error "GitHub CLI is not authenticated."
            echo ""
            echo "Run: gh auth login"
            echo ""
            errors=$((errors + 1))
        fi
    fi
    
    # Check for git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed."
        errors=$((errors + 1))
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --is-inside-work-tree &> /dev/null; then
        log_error "Not in a git repository."
        errors=$((errors + 1))
    fi
    
    if [ $errors -gt 0 ]; then
        exit 1
    fi
}

get_current_branch() {
    git branch --show-current
}

get_commits_ahead() {
    git rev-list --count origin/${DEFAULT_BASE_BRANCH}..HEAD 2>/dev/null || echo "0"
}

get_github_username() {
    if ! command -v gh &> /dev/null; then
        echo "YOUR_USERNAME"
        return 1
    fi
    gh api user --jq '.login'
}

generate_pr_title() {
    # Try to generate a title from the first commit message
    local first_commit
    first_commit=$(git log origin/${DEFAULT_BASE_BRANCH}..HEAD --pretty=format:"%s" 2>/dev/null | head -1)
    
    if [ -n "$first_commit" ]; then
        echo "$first_commit"
    else
        echo "Update from fork"
    fi
}

generate_pr_body() {
    local commits_ahead
    commits_ahead=$(get_commits_ahead)
    
    cat << EOF
## Summary
This PR includes ${commits_ahead} commit(s) from a fork.

## Changes
\`\`\`
$(git log origin/${DEFAULT_BASE_BRANCH}..HEAD --oneline 2>/dev/null || echo "Unable to list commits")
\`\`\`

## Test plan
- [ ] Verify changes work as expected
- [ ] Run existing tests
- [ ] Review code quality
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Main Functions
# ─────────────────────────────────────────────────────────────────────────────

create_fork() {
    local dry_run="$1"
    
    log_info "Checking if fork already exists..."
    
    # Check if fork exists
    local username
    username=$(get_github_username)
    
    if gh repo view "${username}/hermes-agent" &> /dev/null; then
        log_success "Fork already exists: ${username}/hermes-agent"
        return 0
    fi
    
    log_info "Creating fork of ${UPSTREAM_REPO}..."
    
    if [ "$dry_run" = "true" ]; then
        echo "  [DRY RUN] Would run: gh repo fork ${UPSTREAM_REPO} --clone=false"
        return 0
    fi
    
    if gh repo fork "${UPSTREAM_REPO}" --clone=false; then
        log_success "Fork created: ${username}/hermes-agent"
    else
        log_error "Failed to create fork"
        return 1
    fi
}

add_fork_remote() {
    local dry_run="$1"
    local remote_name="$2"
    local username
    
    username=$(get_github_username)
    
    # Check if remote already exists
    if git remote | grep -q "^${remote_name}$"; then
        log_info "Remote '${remote_name}' already exists. Updating URL..."
        if [ "$dry_run" = "true" ]; then
            echo "  [DRY RUN] Would run: git remote set-url ${remote_name} https://github.com/${username}/hermes-agent.git"
            return 0
        fi
        git remote set-url "${remote_name}" "https://github.com/${username}/hermes-agent.git"
    else
        log_info "Adding fork as remote '${remote_name}'..."
        if [ "$dry_run" = "true" ]; then
            echo "  [DRY RUN] Would run: git remote add ${remote_name} https://github.com/${username}/hermes-agent.git"
            return 0
        fi
        git remote add "${remote_name}" "https://github.com/${username}/hermes-agent.git"
    fi
    
    log_success "Remote '${remote_name}' configured: https://github.com/${username}/hermes-agent.git"
}

push_to_fork() {
    local dry_run="$1"
    local remote_name="$2"
    local branch="$3"
    local force="$4"
    
    local force_flag=""
    if [ "$force" = "true" ]; then
        force_flag="--force"
    fi
    
    log_info "Pushing branch '${branch}' to fork..."
    
    if [ "$dry_run" = "true" ]; then
        echo "  [DRY RUN] Would run: git push ${force_flag} ${remote_name} ${branch}"
        return 0
    fi
    
    if git push ${force_flag} "${remote_name}" "${branch}"; then
        log_success "Pushed to fork: ${remote_name}/${branch}"
    else
        log_error "Failed to push to fork"
        return 1
    fi
}

create_pr() {
    local dry_run="$1"
    local branch="$2"
    local title="$3"
    local draft="$4"
    
    local draft_flag=""
    if [ "$draft" = "true" ]; then
        draft_flag="--draft"
    fi
    
    log_info "Creating pull request..."
    
    if [ "$dry_run" = "true" ]; then
        echo "  [DRY RUN] Would run:"
        echo "    gh pr create --repo ${UPSTREAM_REPO} \\"
        echo "                 --base ${DEFAULT_BASE_BRANCH} \\"
        echo "                 --head $(get_github_username):${branch} \\"
        echo "                 --title \"${title}\" \\"
        echo "                 ${draft_flag}"
        return 0
    fi
    
    local body
    body=$(generate_pr_body)
    
    local pr_url
    pr_url=$(gh pr create --repo "${UPSTREAM_REPO}" \
                          --base "${DEFAULT_BASE_BRANCH}" \
                          --head "$(get_github_username):${branch}" \
                          --title "${title}" \
                          --body "${body}" \
                          ${draft_flag})
    
    if [ $? -eq 0 ]; then
        log_success "Pull request created: ${pr_url}"
        echo ""
        echo "View your PR at: ${pr_url}"
    else
        log_error "Failed to create pull request"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Main Script
# ─────────────────────────────────────────────────────────────────────────────

main() {
    local branch=""
    local title=""
    local draft="false"
    local dry_run="false"
    local force="false"
    local remote_name="${FORK_REMOTE_NAME}"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--branch)
                branch="$2"
                shift 2
                ;;
            -t|--title)
                title="$2"
                shift 2
                ;;
            -d|--draft)
                draft="true"
                shift
                ;;
            -n|--dry-run)
                dry_run="true"
                shift
                ;;
            -f|--force)
                force="true"
                shift
                ;;
            -r|--remote)
                remote_name="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run prerequisite checks (skip in dry-run for demo purposes)
    if [ "$dry_run" = "false" ]; then
        check_prerequisites
    fi
    
    # Determine branch
    if [ -z "$branch" ]; then
        branch=$(get_current_branch)
        log_info "Using current branch: ${branch}"
    fi
    
    # Check for commits ahead
    local commits_ahead
    commits_ahead=$(get_commits_ahead)
    
    if [ "$commits_ahead" -eq 0 ]; then
        log_warning "No commits ahead of origin/${DEFAULT_BASE_BRANCH}"
        echo "Nothing to push. Make sure you have commits ready."
        if [ "$dry_run" = "false" ]; then
            exit 0
        fi
    else
        log_info "Found ${commits_ahead} commit(s) ahead of origin/${DEFAULT_BASE_BRANCH}"
    fi
    
    # Generate title if not provided
    if [ -z "$title" ]; then
        title=$(generate_pr_title)
        log_info "Auto-generated PR title: ${title}"
    fi
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo " Fork & Push Automation"
    echo "════════════════════════════════════════════════════════════════"
    echo "  Upstream:      ${UPSTREAM_REPO}"
    echo "  Branch:        ${branch}"
    echo "  Commits ahead: ${commits_ahead}"
    echo "  PR Title:      ${title}"
    echo "  Draft:         ${draft}"
    echo "  Force push:    ${force}"
    echo "  Remote name:   ${remote_name}"
    if [ "$dry_run" = "true" ]; then
        echo "  Mode:          DRY RUN (no changes will be made)"
    fi
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Execute steps
    create_fork "$dry_run" || exit 1
    add_fork_remote "$dry_run" "$remote_name" || exit 1
    push_to_fork "$dry_run" "$remote_name" "$branch" "$force" || exit 1
    create_pr "$dry_run" "$branch" "$title" "$draft" || exit 1
    
    echo ""
    log_success "All done! 🎉"
    
    if [ "$dry_run" = "true" ]; then
        echo ""
        echo "This was a dry run. Remove --dry-run to execute the changes."
    fi
}

# Run main with all arguments
main "$@"
