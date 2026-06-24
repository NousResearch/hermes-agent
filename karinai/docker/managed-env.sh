#!/bin/sh
# Shared Docker bootstrap helpers for KarinAI managed-runtime containers.
#
# This file is sourced by the upstream Docker stage2 hook and main-wrapper when
# KARINAI_MANAGED_RUNTIME is enabled. Keep it POSIX-sh compatible: it runs under
# s6-overlay's /command/with-contenv sh, not bash.

karinai_bool_truthy() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

karinai_managed_runtime_enabled() {
    karinai_bool_truthy "${KARINAI_MANAGED_RUNTIME:-}"
}

karinai_envdir_set() {
    name=$1
    value=$2
    envdir="${KARINAI_S6_ENVDIR:-/run/s6/container_environment}"
    if [ -d "$envdir" ] || mkdir -p "$envdir" 2>/dev/null; then
        printf '%s' "$value" > "$envdir/$name" 2>/dev/null || true
    fi
}

karinai_apply_managed_bootstrap_env() {
    karinai_managed_runtime_enabled || return 0

    if [ -n "${KARINAI_RUNTIME_STATE_DIR:-}" ]; then
        HERMES_HOME="$KARINAI_RUNTIME_STATE_DIR"
        export HERMES_HOME
        karinai_envdir_set HERMES_HOME "$HERMES_HOME"

        HOME="$HERMES_HOME/home"
        export HOME
        karinai_envdir_set HOME "$HOME"
    fi

    if [ -n "${KARINAI_WORKSPACE_DIR:-}" ]; then
        TERMINAL_CWD="$KARINAI_WORKSPACE_DIR"
        export TERMINAL_CWD
        karinai_envdir_set TERMINAL_CWD "$TERMINAL_CWD"

        HERMES_WRITE_SAFE_ROOT="$KARINAI_WORKSPACE_DIR"
        export HERMES_WRITE_SAFE_ROOT
        karinai_envdir_set HERMES_WRITE_SAFE_ROOT "$HERMES_WRITE_SAFE_ROOT"
    fi

    # Managed beta containers should not expose the dashboard. This is written
    # before s6 starts services, so docker/s6-rc.d/dashboard/run sees it too.
    HERMES_DASHBOARD=false
    export HERMES_DASHBOARD
    karinai_envdir_set HERMES_DASHBOARD "$HERMES_DASHBOARD"
}
