#!/usr/bin/env bash
# Update the current Hermes checkout while preserving the current branch.
#
# This is intended for fork/patch workflows where local commits should be
# replayed on top of the latest official main. It never switches to main and
# never hard-resets your branch. If Git hits conflicts, the rebase stops with
# your backup branch and any autostash left intact for manual recovery.

set -euo pipefail

OFFICIAL_REPO_SLUG="NousResearch/hermes-agent"
OFFICIAL_REPO_URL="https://github.com/NousResearch/hermes-agent.git"
DEFAULT_EXPECTED_BRANCH="local/hindsight-embedded-profile-env"

usage() {
    cat <<'EOF'
Usage: scripts/update-current-branch.sh [options]

Rebase the current branch onto the latest official Hermes main while preserving
local commits and uncommitted work.

Options:
  --no-push                Do not update the branch's upstream remote after
                           a successful rebase. By default, the script pushes
                           with --force-with-lease so local and fork stay in sync.
  --push-fork              Deprecated alias for the default push behavior.
  --expected-branch NAME   Require this local branch before updating. Defaults
                           to local/hindsight-embedded-profile-env, or the
                           HERMES_UPDATE_BRANCH env var when set.
  --any-branch             Disable the expected-branch guard for one run.
  --upstream REF           Rebase onto this ref instead of auto-detected
                           official main (for example origin/main).
  --dry-run                Show what would be updated without changing refs.
  -h, --help               Show this help.

Examples:
  scripts/update-current-branch.sh
  scripts/update-current-branch.sh --no-push
  scripts/update-current-branch.sh --expected-branch my/local-branch
  scripts/update-current-branch.sh --upstream origin/main
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

info() {
    printf '→ %s\n' "$*"
}

ok() {
    printf '✓ %s\n' "$*"
}

warn() {
    printf '⚠ %s\n' "$*" >&2
}

run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

remote_url() {
    git remote get-url "$1" 2>/dev/null || true
}

remote_is_official() {
    local url normalized
    url="$(remote_url "$1")"
    [ -n "$url" ] || return 1
    normalized="${url%.git}"
    case "$normalized" in
        *github.com[:/]$OFFICIAL_REPO_SLUG) return 0 ;;
        https://github.com/$OFFICIAL_REPO_SLUG) return 0 ;;
        git@github.com:$OFFICIAL_REPO_SLUG) return 0 ;;
        *) return 1 ;;
    esac
}

sanitize_ref_component() {
    printf '%s' "$1" | tr '/:~^?*[\\ ' '----------'
}

official_remote() {
    local remote
    for remote in upstream origin fork; do
        if git remote get-url "$remote" >/dev/null 2>&1 && remote_is_official "$remote"; then
            printf '%s\n' "$remote"
            return 0
        fi
    done
    return 1
}

ensure_no_git_operation_in_progress() {
    local git_dir
    git_dir="$(git rev-parse --git-dir)"
    for marker in \
        "$git_dir/rebase-merge" \
        "$git_dir/rebase-apply" \
        "$git_dir/MERGE_HEAD" \
        "$git_dir/CHERRY_PICK_HEAD" \
        "$git_dir/REVERT_HEAD"
    do
        if [ -e "$marker" ]; then
            die "another Git operation is in progress. Resolve or abort it before updating."
        fi
    done
}

stash_worktree_if_needed() {
    local stash_name stash_ref
    if [ -z "$(git status --porcelain)" ]; then
        return 0
    fi

    stash_name="hermes-current-branch-update-$(date -u +%Y%m%d-%H%M%S)"
    info "Uncommitted changes detected; stashing as $stash_name"
    run git stash push --include-untracked -m "$stash_name"

    if [ "$DRY_RUN" = "1" ]; then
        STASH_REF="dry-run-stash"
        return 0
    fi

    stash_ref="$(git rev-parse --verify refs/stash)"
    STASH_REF="$stash_ref"
}

restore_stash_if_needed() {
    [ -n "${STASH_REF:-}" ] || return 0
    if [ "$DRY_RUN" = "1" ]; then
        info "Would restore stashed uncommitted changes"
        return 0
    fi

    info "Restoring stashed uncommitted changes"
    if git stash apply "$STASH_REF"; then
        local selector
        selector="$(git stash list --format='%gd %H' | awk -v ref="$STASH_REF" '$2 == ref { print $1; exit }')"
        if [ -n "$selector" ]; then
            git stash drop "$selector" >/dev/null
        else
            warn "restored changes, but could not find the stash entry to drop"
            warn "stash commit: $STASH_REF"
        fi
        return 0
    fi

    warn "stashed changes conflicted while restoring"
    warn "your stash is preserved at commit: $STASH_REF"
    warn "resolve conflicts, or recover with: git stash apply $STASH_REF"
    exit 1
}

DRY_RUN=0
PUSH_FORK=1
UPSTREAM_REF=""
EXPECTED_BRANCH="${HERMES_UPDATE_BRANCH:-$DEFAULT_EXPECTED_BRANCH}"
ALLOW_ANY_BRANCH=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --push-fork)
            PUSH_FORK=1
            shift
            ;;
        --no-push)
            PUSH_FORK=0
            shift
            ;;
        --expected-branch)
            [ "$#" -ge 2 ] || die "--expected-branch requires a branch name"
            EXPECTED_BRANCH="$2"
            ALLOW_ANY_BRANCH=0
            shift 2
            ;;
        --any-branch)
            ALLOW_ANY_BRANCH=1
            shift
            ;;
        --upstream)
            [ "$#" -ge 2 ] || die "--upstream requires a ref"
            UPSTREAM_REF="$2"
            shift 2
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
            die "unknown option: $1"
            ;;
    esac
done

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a Git repository"
cd "$repo_root"

ensure_no_git_operation_in_progress

current_branch="$(git branch --show-current)"
[ -n "$current_branch" ] || die "detached HEAD is not supported; check out a branch first"

if [ "$ALLOW_ANY_BRANCH" != "1" ]; then
    [ -n "$EXPECTED_BRANCH" ] || die "expected branch guard is empty; use --any-branch to override"
    if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
        die "refusing to update branch '$current_branch'; expected '$EXPECTED_BRANCH'. Use --expected-branch NAME for a planned branch rename, or --any-branch to override once."
    fi
fi

if [ -z "$UPSTREAM_REF" ]; then
    if official="$(official_remote)"; then
        info "Official upstream remote: $official ($(remote_url "$official"))"
    else
        if git remote get-url upstream >/dev/null 2>&1; then
            official="upstream"
        else
            info "Adding official upstream remote"
            run git remote add upstream "$OFFICIAL_REPO_URL"
            official="upstream"
        fi
    fi

    info "Fetching latest official main"
    run git fetch "$official" main
    UPSTREAM_REF="$official/main"
else
    info "Using requested upstream ref: $UPSTREAM_REF"
    remote_name="${UPSTREAM_REF%%/*}"
    if [ "$remote_name" != "$UPSTREAM_REF" ] && git remote get-url "$remote_name" >/dev/null 2>&1; then
        info "Fetching $remote_name"
        run git fetch "$remote_name"
    fi
fi

git rev-parse --verify "$UPSTREAM_REF" >/dev/null 2>&1 || die "cannot resolve upstream ref: $UPSTREAM_REF"

upstream_branch="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
info "Current branch: $current_branch"
if [ -n "$upstream_branch" ]; then
    info "Branch upstream: $upstream_branch"
else
    warn "current branch has no configured upstream remote"
fi

backup_branch="backup/$(sanitize_ref_component "$current_branch")-before-update-$(date -u +%Y%m%d-%H%M%S)"
info "Creating backup branch: $backup_branch"
run git branch "$backup_branch" HEAD

ahead="$(git rev-list --count "$UPSTREAM_REF..HEAD")"
behind="$(git rev-list --count "HEAD..$UPSTREAM_REF")"
info "Branch is $ahead commit(s) ahead and $behind commit(s) behind $UPSTREAM_REF"

STASH_REF=""
stash_worktree_if_needed

info "Rebasing $current_branch onto $UPSTREAM_REF"
if ! run git rebase "$UPSTREAM_REF"; then
    warn "rebase stopped because of conflicts"
    warn "backup branch: $backup_branch"
    if [ -n "${STASH_REF:-}" ]; then
        warn "uncommitted changes are preserved in stash commit: $STASH_REF"
    fi
    warn "after resolving conflicts, run: git rebase --continue"
    warn "to go back, run: git rebase --abort && git switch $backup_branch"
    exit 1
fi

restore_stash_if_needed

ok "Updated $current_branch onto $UPSTREAM_REF"
ok "Backup branch kept at $backup_branch"


print_post_update_upstream_guidance() {
    local tracking ahead_remote behind_remote remote_name remote_branch
    tracking="$1"
    [ -n "$tracking" ] || return 0

    ahead_remote="$(git rev-list --count "$tracking..HEAD" 2>/dev/null || printf '0')"
    behind_remote="$(git rev-list --count "HEAD..$tracking" 2>/dev/null || printf '0')"
    if [ "$ahead_remote" = "0" ] && [ "$behind_remote" = "0" ]; then
        ok "Branch upstream is in sync: $tracking"
        return 0
    fi

    remote_name="${tracking%%/*}"
    remote_branch="${tracking#*/}"
    warn "local branch and $tracking are not in sync after the rebase"
    warn "status: ahead $ahead_remote, behind $behind_remote"
    warn "this is expected after rebasing until the fork branch is updated"
    warn "push safely with: git push --force-with-lease $remote_name HEAD:$remote_branch"
}

if [ "$PUSH_FORK" = "1" ]; then
    [ -n "$upstream_branch" ] || die "--push-fork requested, but current branch has no upstream"
    push_remote="${upstream_branch%%/*}"
    push_branch="${upstream_branch#*/}"
    info "Pushing $current_branch to $push_remote/$push_branch with --force-with-lease"
    run git push --force-with-lease "$push_remote" "HEAD:$push_branch"
    ok "Pushed $push_remote/$push_branch"
else
    print_post_update_upstream_guidance "$upstream_branch"
fi

git status --short --branch
