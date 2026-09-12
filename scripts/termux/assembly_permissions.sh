#!/usr/bin/env bash
# Host-side ownership handoff for the disposable environment assembly mount.

prepare_assembly() {
    ASSEMBLY="$(mktemp -d "$PAYLOAD_ABS/.environments-XXXXXX")"
    # Docker would create the parent of the nested tool mounts as root:root.
    # Pre-create it and hand off the private scratch tree to Termux's system uid.
    mkdir "$ASSEMBLY/tools"
    docker run --rm --user 0:0 --network none --entrypoint chown \
        -v "$ASSEMBLY:/assembly" "$IMAGE" 1000:1000 /assembly /assembly/tools
}

restore_assembly_owner() {
    # No input mounts are present here: never recurse through mounted tools/app.
    # Bypass Termux's entrypoint, which otherwise drops root to uid 1000.
    docker run --rm --user 0:0 --network none --entrypoint chown \
        -v "$ASSEMBLY:/assembly" "$IMAGE" -R "$(id -u):$(id -g)" /assembly
}
