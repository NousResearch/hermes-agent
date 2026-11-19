#!/usr/bin/env bash
# The x64 runner is Ubuntu 24.04. Run the app build in 22.04 without sharing
# the host's libraries, tools, home, Docker socket, or release credentials.
set -euo pipefail
phase=${1:?expected setup, stage-gpu, build or accept}
shift
case "$phase" in
  setup)
    context=$(mktemp -d)
    trap 'rm -rf -- "$context"' EXIT
    cp -- "$1" "$context/install.sh"
    cp scripts/ci/linux_x64_userland.Dockerfile "$context/Dockerfile"
    for target in floor builder gpu; do
      docker build --target "$target" --build-arg "UID_BUILDER=$(id -u)" --build-arg "GID_BUILDER=$(id -g)" \
        --iidfile "$context/id" "$context"
      printf 'X64_%s_IMAGE=%s\n' "${target^^}" "$(<"$context/id")" >> "$GITHUB_ENV"
    done ;;
  build)
    # GH file commands are individual mounts, not the runner's temp directory.
    args=(--user "$(id -u):$(id -g)" --volume "$GITHUB_WORKSPACE:$GITHUB_WORKSPACE" --workdir "$GITHUB_WORKSPACE"
          --env HOME=/build-home --env GITHUB_WORKSPACE --env HERMES_PAYLOAD_TAG --env HERMES_PAYLOAD_VERSION
          --env HERMES_RELEASE_COMMIT --env RELEASE_CLAIM_TAG --env RELEASE_CLAIM_OBJECT --env HERMES_DESKTOP_VARIANT
          --env HERMES_PYTHON --env HERMES_NODE --env HERMES_RUNTIME_DIR --env HERMES_HOME --env PYTHONUTF8)
    for key in GITHUB_ENV GITHUB_PATH; do
      args+=(--volume "${!key}:/commands/$key" --env "$key=/commands/$key")
    done
    # desktop_commands exports the job-local PM commands after preparation.
    prepared="$GITHUB_WORKSPACE/.build/desktop-job/commands"
    if [[ -n ${HERMES_NODE:-} && -n ${HERMES_PYTHON:-} ]]; then
      args+=(--env "PATH=$prepared:$(dirname "$HERMES_NODE"):$(dirname "$HERMES_PYTHON"):/usr/local/bin:/usr/bin:/bin")
    else
      [[ -f ${X64_BOOTSTRAP_PYTHON:-} && -x ${X64_BOOTSTRAP_PYTHON:-} ]] || { printf '%s\n' 'Missing PM x64 bootstrap Python' >&2; exit 1; }
      args+=(--env "PATH=$(dirname "$X64_BOOTSTRAP_PYTHON"):/usr/local/bin:/usr/bin:/bin")
    fi
    docker run --rm -i "${args[@]}" "$X64_BUILDER_IMAGE" bash -eo pipefail -s < "$1" ;;
  stage-gpu)
    docker run --rm --user "$(id -u):$(id -g)" --env HOME=/build-home --env PYTHONDONTWRITEBYTECODE=1 \
      --env HERMES_HOME=/build-home/state --env HERMES_RUNTIME_DIR=/build-home/tools \
      --volume "$GITHUB_ENV:/commands/GITHUB_ENV" --env GITHUB_ENV=/commands/GITHUB_ENV \
      --volume "$GITHUB_WORKSPACE:$GITHUB_WORKSPACE" --workdir "$GITHUB_WORKSPACE" \
      "$X64_GPU_IMAGE" python3 -B -m scripts.bundles.stage_x64_gpu ;;
  accept)
    shopt -s nullglob
    images=("$GITHUB_WORKSPACE"/apps/desktop/release/*.AppImage)
    [[ ${#images[@]} == 1 ]] || { printf '%s\n' 'Expected exactly one packaged x64 AppImage' >&2; exit 1; }
    digest=$(sha256sum -- "${images[0]}")
    digest=${digest%% *}
    for ubuntu in 22.04 24.04; do
      image=$X64_GPU_IMAGE
      start=()
      if [[ $ubuntu == 22.04 ]]; then image=$X64_FLOOR_IMAGE; start=(xvfb-run -a); fi
      docker run --rm --init --network none --cap-drop ALL --user "$(id -u):$(id -g)" --workdir /work \
        --shm-size=512m --env HOME=/work/home --env HERMES_HOME=/work/home/.hermes --env XDG_RUNTIME_DIR=/work/runtime \
        --env UV_OFFLINE=1 --env HERMES_DISABLE_LAZY_INSTALLS=1 --env npm_config_offline=true \
        --volume "${images[0]}:/image.AppImage:ro" \
        --volume "$GITHUB_WORKSPACE/scripts/bundles/appimage_checks.py:/checks.py:ro" \
        --volume "$GITHUB_WORKSPACE/scripts/smoke-payload.sh:/smoke-payload.sh:ro" \
        "$image" bash -ec '
          . /etc/os-release
          [[ $ID == ubuntu && $VERSION_ID == "$1" && $(uname -m) == x86_64 ]] || exit 1
          printf "%s  /image.AppImage\n" "$2" | sha256sum --check --strict
          mkdir -m 700 /work/home /work/runtime
          ubuntu=$1; shift 2
          exec "$@" python3 -I /checks.py "$ubuntu"
        ' bash "$ubuntu" "$digest" "${start[@]}"
      printf 'Accepted Ubuntu %s x64 AppImage sha256=%s\n' "$ubuntu" "$digest"
    done ;;
  *) printf 'Unknown x64 userland phase: %s\n' "$phase" >&2; exit 2 ;;
esac
