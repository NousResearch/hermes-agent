#!/usr/bin/env bash
# =============================================================================
# launch-hermes-docs.sh — Local launcher for the Hermes Docs workspace
# =============================================================================
#
# Starts the Hermes Dashboard on the chosen port and prints the direct URL
# for the Docs workspace.  Does not touch credentials, profiles, or workspaces.
# No extra dependencies beyond a working Hermes install.
#
# Usage:
#   bash scripts/launch-hermes-docs.sh              # port 9119, opens browser
#   bash scripts/launch-hermes-docs.sh --port 8787  # custom port
#   bash scripts/launch-hermes-docs.sh --no-open    # no auto-open
#   bash scripts/launch-hermes-docs.sh --help
#
# The Docs tab lives at:  http://localhost:<port>/docs-workspace
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PORT=9119
NO_OPEN=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="${2:?--port requires a value}"
            shift 2
            ;;
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --no-open)
            NO_OPEN=1
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            printf 'Run with --help for usage.\n' >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Prerequisite: hermes must be on PATH
# ---------------------------------------------------------------------------
if ! command -v hermes >/dev/null 2>&1; then
    printf '[hermes-docs] ERROR: "hermes" not found on PATH.\n' >&2
    printf '  Install Hermes first:\n' >&2
    printf '    curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash\n' >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Print launch info
# ---------------------------------------------------------------------------
DOCS_URL="http://localhost:${PORT}/docs-workspace"

printf '\n'
printf '  Hermes Docs\n'
printf '  ──────────────────────────────────────────\n'
printf '  Dashboard  →  http://localhost:%s\n' "${PORT}"
printf '  Docs tab   →  %s\n' "${DOCS_URL}"
printf '  ──────────────────────────────────────────\n'
printf '  Press Ctrl-C to stop.\n'
printf '\n'

# ---------------------------------------------------------------------------
# Build hermes dashboard args
# ---------------------------------------------------------------------------
DASHBOARD_ARGS=(dashboard --port "${PORT}")

if [[ "${NO_OPEN}" -eq 1 ]]; then
    DASHBOARD_ARGS+=(--no-open)
fi

# When --no-open is not set, open directly to the Docs workspace URL instead
# of the default dashboard root.  We delay one second so the server is ready.
if [[ "${NO_OPEN}" -eq 0 ]]; then
    (
        sleep 1.5
        if command -v open >/dev/null 2>&1; then       # macOS
            open "${DOCS_URL}"
        elif command -v xdg-open >/dev/null 2>&1; then # Linux
            xdg-open "${DOCS_URL}"
        fi
    ) &
    # Tell the dashboard not to open the generic root — we handle it above.
    DASHBOARD_ARGS+=(--no-open)
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
exec hermes "${DASHBOARD_ARGS[@]}"
