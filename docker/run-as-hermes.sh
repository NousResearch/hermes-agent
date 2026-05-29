#!/bin/sh
# Helpers for running commands as the container's hermes user.
#
# s6-setuidgid is only safe when the caller has enough privilege to rebuild
# the supplementary group list. If Docker already started the container as the
# hermes UID (for example with compose `user: "10000:10000"` plus group_add),
# calling s6-setuidgid again fails with "unable to set supplementary group
# list". In that case the process is already at the target UID, so run the
# command directly and preserve Docker's injected groups.

_hermes_current_uid() {
    id -u 2>/dev/null
}

_hermes_target_uid() {
    id -u hermes 2>/dev/null
}

_hermes_already_target_user() {
    current_uid="$(_hermes_current_uid || true)"
    target_uid="$(_hermes_target_uid || true)"
    [ -n "$current_uid" ] && [ -n "$target_uid" ] && [ "$current_uid" = "$target_uid" ]
}

run_as_hermes() {
    if _hermes_already_target_user; then
        "$@"
    else
        s6-setuidgid hermes "$@"
    fi
}

exec_as_hermes() {
    if _hermes_already_target_user; then
        exec "$@"
    else
        exec s6-setuidgid hermes "$@"
    fi
}
