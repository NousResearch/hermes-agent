#!/usr/bin/env bash
# scripts/install-local-patch-hooks.sh
#
# Install the local-patch-recovery infrastructure into ~/.hermes/.
# Idempotent — safe to run multiple times.  Read-only on the source
# checkout (only writes to ~/.hermes/ and the local .git/hooks/).
#
# What gets installed:
#   ~/.hermes/patches/                    — patch archive (created if missing)
#   ~/.hermes/bin/hermes-post-merge-hook.sh — hook source (the template
#                                            shipped in the repo is the
#                                            same file, just kept in
#                                            sync by re-running this)
#   ~/.hermes/bin/hermes-preupdate.sh     — pre-update preflight script
#   ~/.hermes/patches/manifest.txt        — manifest of tracked local
#                                            patches (created if missing
#                                            with a placeholder entry
#                                            pointing at the well-known
#                                            minimax-oauth fix)
#   ~/.hermes/patches/README.md           — documentation
#   /usr/local/lib/hermes-agent/.git/hooks/post-merge — the actual hook
#
# Usage:
#   ./scripts/install-local-patch-hooks.sh         # install
#   ./scripts/install-local-patch-hooks.sh --check # verify
#   ./scripts/install-local-patch-hooks.sh --uninstall

set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
REPO_ROOT="${HERMES_REPO_ROOT:-/usr/local/lib/hermes-agent}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//; s/^#//'
    exit 0
}

CHECK_ONLY=0
UNINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --uninstall) UNINSTALL=1 ;;
        -h|--help) usage ;;
    esac
done

PATCH_DIR="$HERMES_HOME/patches"
HOOK_DEST="$REPO_ROOT/.git/hooks/post-merge"
HOOK_SRC="$SCRIPT_DIR/local-patch-recovery-hook.sh"
PREUPDATE_SCRIPT="$SCRIPT_DIR/hermes-preupdate.sh"
README_SRC="$SCRIPT_DIR/local-patches-README.md"
MANIFEST="$PATCH_DIR/manifest.txt"

say()  { printf '%s\n' "$*"; }
err()  { printf '✗ %s\n' "$*" >&2; }
warn() { printf '⚠ %s\n' "$*"; }

if [[ $UNINSTALL -eq 1 ]]; then
    say "Uninstalling local-patch-recovery infrastructure..."
    if [[ -f "$HOOK_DEST" ]]; then
        rm -f "$HOOK_DEST"
        say "  removed $HOOK_DEST"
    fi
    say "  ~/.hermes/patches/ and ~/.hermes/bin/ left in place (manual cleanup if needed)"
    say "Done."
    exit 0
fi

# Pre-flight: source files exist
for f in "$HOOK_SRC" "$PREUPDATE_SCRIPT" "$README_SRC"; do
    if [[ ! -f "$f" ]]; then
        err "missing source file: $f"
        exit 1
    fi
done

if [[ $CHECK_ONLY -eq 1 ]]; then
    say "Checking local-patch-recovery install..."
    ERRORS=0
    for f in "$HERMES_HOME/bin/hermes-post-merge-hook.sh" "$HERMES_HOME/bin/hermes-preupdate.sh" "$PATCH_DIR/README.md"; do
        if [[ -f "$f" ]]; then
            say "  ✓ $f"
        else
            err "$f"
            ERRORS=$((ERRORS+1))
        fi
    done
    if [[ -x "$HOOK_DEST" ]]; then
        say "  ✓ $HOOK_DEST (executable)"
    else
        err "$HOOK_DEST (missing or not executable)"
        ERRORS=$((ERRORS+1))
    fi
    if [[ -f "$MANIFEST" ]]; then
        say "  ✓ $MANIFEST"
    else
        err "$MANIFEST (missing)"
        ERRORS=$((ERRORS+1))
    fi
    if [[ $ERRORS -eq 0 ]]; then
        say "All checks passed."
        exit 0
    else
        err "$ERRORS check(s) failed"
        exit 1
    fi
fi

say "Installing local-patch-recovery infrastructure..."

# Create dirs
mkdir -p "$HERMES_HOME/patches" "$HERMES_HOME/bin" "$HERMES_HOME/logs"
say "  ✓ ~/.hermes/patches/"
say "  ✓ ~/.hermes/bin/"
say "  ✓ ~/.hermes/logs/"

# Install scripts
install -m 0755 "$HOOK_SRC"         "$HERMES_HOME/bin/hermes-post-merge-hook.sh"
install -m 0755 "$PREUPDATE_SCRIPT" "$HERMES_HOME/bin/hermes-preupdate.sh"
install -m 0644 "$README_SRC"       "$PATCH_DIR/README.md"
say "  ✓ ~/.hermes/bin/hermes-post-merge-hook.sh"
say "  ✓ ~/.hermes/bin/hermes-preupdate.sh"
say "  ✓ ~/.hermes/patches/README.md"

# Install hook
install -m 0755 "$HOOK_SRC" "$HOOK_DEST"
say "  ✓ $HOOK_DEST"

# Create manifest placeholder if missing
if [[ ! -f "$MANIFEST" ]]; then
    cat > "$MANIFEST" <<'EOF'
# Local patches manifest
# Format: <full-sha>  <branch-name>  [description]
# See ~/.hermes/patches/README.md for usage.
#
# Example entry:
# 25222e49068daa243a45850a43e92a6498e6abf5  fix/minimax-oauth-auxiliary-routing  PR #36779
EOF
    say "  ✓ $MANIFEST (placeholder created)"
else
    say "  ✓ $MANIFEST (already exists, leaving alone)"
fi

say ""
say "Done. To verify:  ./scripts/install-local-patch-hooks.sh --check"
say "To uninstall:  ./scripts/install-local-patch-hooks.sh --uninstall"
say "To run preflight:  ~/.hermes/bin/hermes-preupdate.sh"
