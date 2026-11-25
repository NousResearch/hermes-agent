#!/usr/bin/env bash
# Build + push the derived termux builder image (toolchain pre-baked).
#
# The image tag hashes the full pinned base digest and Dockerfile bytes,
# so either a lock bump or a toolchain recipe change produces a new image.
# Pushes to GHCR with the repo's CI identity (GITHUB_TOKEN); idempotent --
# an existing identical tag is left alone.
#
# Usage: build_builder_image.sh [--no-push]   (uses the published image, else builds + pushes it; with --no-push or TERMUX_BUILDER_NO_PUSH=1 a miss falls back to the digest-pinned base image)
set -euo pipefail

NO_PUSH="${TERMUX_BUILDER_NO_PUSH:-0}"
for arg in "$@"; do
    case "$arg" in
        --no-push) NO_PUSH=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

DIGEST="$(python3 -c 'import sys; sys.path.insert(0, "."); from pm.lock import termux_docker_digest; print(termux_docker_digest())')"
[ -n "$DIGEST" ] || { echo "termux-docker digest missing" >&2; exit 1; }

BASE="termux/termux-docker@${DIGEST}"
DOCKERFILE="scripts/termux/termux-builder.Dockerfile"
SHORT="$(python3 -c '
import hashlib, pathlib, sys
identity = sys.argv[1].encode() + b"\0" + pathlib.Path(sys.argv[2]).read_bytes()
print(hashlib.sha256(identity).hexdigest()[:12])
' "$BASE" "$DOCKERFILE")"
REGISTRY="ghcr.io"
# macOS still ships Bash 3.2, which predates ${value,,} case conversion.
OWNER="$(printf '%s' "$GITHUB_REPOSITORY_OWNER" | tr '[:upper:]' '[:lower:]')"
IMAGE="${REGISTRY}/${OWNER}/hermes-termux-builder:${SHORT}"

if docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
    echo "builder image already published: $IMAGE"
    echo "$IMAGE"
    exit 0
fi

if [ "$NO_PUSH" = "1" ]; then
    # termux_build.sh pulls the image and reads RepoDigests[0]; a job-local tag
    # has neither. Falling back to the digest-pinned base image works on both
    # current and pre-claim checkouts because termux_build.sh provisions the
    # toolchain inside the container when clang/git are absent.
    echo "builder image not published (push disabled); falling back to base: $BASE"
    echo "$BASE"
    exit 0
fi

echo "building $IMAGE from $BASE"
docker build \
    -f "$DOCKERFILE" \
    --build-arg "BASE=${BASE}" \
    -t "$IMAGE" \
    scripts/termux \
    || { echo "builder image build failed" >&2; exit 1; }

# Smoke: the baked image must answer the runtime probes termux_build.sh
# performs (clang + rustc + cargo + make) before we publish it. /bin/sh is
# linked at RUNTIME inside the wheelhouse phase's --tmpfs /bin, so it is
# deliberately absent from the baked image.
docker run --rm --platform linux/arm64 "$IMAGE" \
    /data/data/com.termux/files/usr/bin/bash -c '
        export PREFIX=/data/data/com.termux/files/usr
        for tool in clang rustc cargo make; do
            command -v "$tool" >/dev/null 2>&1 || { echo "smoke FAIL: $tool"; exit 1; }
        done
        echo "builder image smoke OK"
    ' || { echo "builder image failed its smoke test" >&2; exit 1; }

echo "$GITHUB_TOKEN" | docker login "$REGISTRY" -u "${GITHUB_ACTOR:-x}" --password-stdin
docker push "$IMAGE" || { echo "builder image push failed" >&2; exit 1; }
echo "published $IMAGE"
echo "$IMAGE"
