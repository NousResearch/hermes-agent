#!/usr/bin/env bash
# Lint the container inputs: `scripts/ci/lint_docker.sh [hadolint|shellcheck]` (default: both).
#
# hadolint runs from its image, pinned by digest, on the Dockerfiles; rules and ignores live in
# .hadolint.yaml. shellcheck runs from PATH at error severity on the shell scripts under docker/
# the way CI's ludeeus/action-shellcheck selects them: `*.sh` files, plus executable files with
# no extension whose first line is a sh/bash/ksh/zsh shebang. CI runs only the hadolint half
# here; its shellcheck comes from that action.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
HADOLINT_IMAGE=ghcr.io/hadolint/hadolint:v2.12.0-debian@sha256:27173fe25e062448490a32de410c08491c626a0bef360aa2ce5d5bdd9384b50d

run_hadolint() {
    docker run --rm --network none -v "$PWD:/repo:ro" -w /repo "$HADOLINT_IMAGE" \
        hadolint --config .hadolint.yaml --failure-threshold warning \
        Dockerfile docker/sandbox-desktop.Dockerfile
}

run_shellcheck() {
    local files=() file
    while IFS= read -r -d '' file; do
        case "${file##*/}" in
            *.sh) files+=("$file") ;;
            *.*) ;;
            *) if [ -x "$file" ] && head -n1 "$file" | grep -Eqs '^#! */[^ ]*/(env *)?[abk]*sh'; then
                   files+=("$file")
               fi ;;
        esac
    done < <(find docker -type f -print0)
    shellcheck --severity=error "${files[@]}"
}

case "${1:-all}" in
    hadolint) run_hadolint ;;
    shellcheck) run_shellcheck ;;
    all) status=0; run_hadolint || status=1; run_shellcheck || status=1; exit "$status" ;;
    *) echo "usage: $0 [hadolint|shellcheck]" >&2; exit 2 ;;
esac
