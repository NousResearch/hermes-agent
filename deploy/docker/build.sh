#!/usr/bin/env bash
# Build the NOVA control-plane image with an immutable tag.
#
# "Immutable" here means the tag names one commit and one tree state, so a tag can never
# be moved to different content later:
#
#     nova-control-plane:0.1.0-g<short sha>
#
# A dirty working tree gets `-dirty.<hash of the diff>` appended rather than being
# refused, because refusing forces people to commit before they can test. The tag still
# identifies exactly one tree.
#
# Nothing here talks to AWS. Pushing to ECR is a separate, deliberate step — the same
# position deploy/aws/README.md already takes.
#
# Usage:
#   deploy/docker/build.sh                 # build and tag
#   IMAGE_NAME=nova-control-plane deploy/docker/build.sh
#   deploy/docker/build.sh --print-tag     # just print the tag it would use

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

IMAGE_NAME="${IMAGE_NAME:-nova-control-plane}"

version="$(python3 -c 'import re,pathlib; print(re.search(r"__version__ = \"([^\"]+)\"", pathlib.Path("nova/__init__.py").read_text()).group(1))')"
sha="$(git rev-parse --short=12 HEAD)"
tag="${version}-g${sha}"

# Everything the image actually copies. Untracked files count: a file that is not
# committed still lands in the layer, so a tag that ignored them would name content that
# is not in the image.
# Named file by file rather than as directories where a directory would over-reach:
# deploy/docker also holds validate-local.sh, which tests the image and is not in it, and a
# tag that changed when the test changed would be lying about what it identifies.
image_paths=(nova hermes_cli cron agent tools gateway plugins providers *.py
             deploy/docker/Dockerfile.nova deploy/docker/entrypoint.sh
             deploy/docker/healthcheck.py deploy/docker/build.sh .dockerignore)

# Modified tracked files contribute their diff; untracked files contribute their path and
# a hash of their contents. Listing untracked paths alone was not enough — editing an
# uncommitted file changed the image without changing the tag, which is exactly the
# property the tag is supposed to have.
dirty="$( {
    git diff HEAD -- "${image_paths[@]}"
    git ls-files --others --exclude-standard -- "${image_paths[@]}" | sort | while read -r f; do
        printf '%s %s\n' "$f" "$(sha256sum "$f" | cut -d" " -f1)"
    done
} )"
if [ -n "$dirty" ]; then
    tag="${tag}-dirty.$(printf '%s' "$dirty" | sha256sum | cut -c1-8)"
fi

if [ "${1:-}" = "--print-tag" ]; then
    printf '%s:%s\n' "$IMAGE_NAME" "$tag"
    exit 0
fi

# Reproducible-ish: the build date is recorded as a label, not used to invalidate layers.
built_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Building behind a TLS-intercepting proxy: point NOVA_BUILD_CA at the proxy's CA and it
# is mounted for the single pip step. BuildKit secrets are not layers, so it never ships.
secret_args=()
if [ -n "${NOVA_BUILD_CA:-}" ]; then
    [ -r "$NOVA_BUILD_CA" ] || { echo "NOVA_BUILD_CA is set but $NOVA_BUILD_CA is not readable" >&2; exit 1; }
    secret_args=(--secret "id=proxy_ca,src=${NOVA_BUILD_CA}")
    echo "using build-time CA ${NOVA_BUILD_CA} (mounted, not copied)"
fi

echo "building ${IMAGE_NAME}:${tag}"
DOCKER_BUILDKIT=1 docker build \
    "${secret_args[@]}" \
    --file deploy/docker/Dockerfile.nova \
    --build-arg "PYTHON_BASE=${PYTHON_BASE:-python:3.12-slim-bookworm}" \
    --tag "${IMAGE_NAME}:${tag}" \
    --label "org.opencontainers.image.version=${version}" \
    --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" \
    --label "org.opencontainers.image.created=${built_at}" \
    "$repo_root"

# A floating convenience tag for local work only. It is deliberately NOT the thing the
# deployment references: deploy/aws takes an image_uri, and that URI should carry the
# immutable tag above so a host restart cannot silently pull different code.
docker tag "${IMAGE_NAME}:${tag}" "${IMAGE_NAME}:local"

echo
echo "built:      ${IMAGE_NAME}:${tag}"
echo "also tagged ${IMAGE_NAME}:local  (local convenience only — do not deploy this tag)"
echo "digest:     $(docker image inspect --format '{{index .Id}}' "${IMAGE_NAME}:${tag}")"
