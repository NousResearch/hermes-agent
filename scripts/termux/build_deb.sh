#!/usr/bin/env bash
# Fat .deb assembly for the Termux hermes-agent bundle (Task 4 of
# .hermes/plans/2026-08-31_termux-deb.md). Runs AFTER termux_build.sh
# (wheelhouse) and build_cpython.sh / build_node.sh have populated the
# payload dir. No opt-out flags: a skipped step is a different artifact.
#
# Inputs (all required):
#   --repo <dir>          hermes-agent checkout (tag must exist; provenance)
#   --tag <tag>           immutable release tag (vX.Y.Z or vX.Y.Z-canary.<ts>)
#   --payload <dir>       dir containing python/, node/, app/ (git archive of
#                         the tag) and wheelhouse/ (from termux_build.sh)
#   --out <dir>           output dir; <out>/hermes-agent_<v>_aarch64.deb lands here
#
# No opt-out flags: the .deb is ALWAYS installed into a fresh run of the
# pinned termux-docker image (digest pinned in pm/lock.json) and smoke-tested.
# docker must be available. The channel is derived from the tag by
# deb_version.py (--channel), not passed in.
#
# Staged payload layout: python/ and node/ are pm-staged termux .deb
# trees ($PREFIX-shaped: data/data/com.termux/files/usr/...). The
# installed layout is $PREFIX/lib/hermes-agent/{tools,app,venv,pm-runtime,bin} with
# exactly one leak: $PREFIX/bin/hermes -> lib/hermes-agent/bin/hermes.

set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# The container digest is a pm pin (the termux-docker package); read it
# from the single lock beside every other third-party artifact pin.
REPO_ROOT="$(cd "$HERE/../.." && pwd)"


REPO=""
TAG=""
COMMIT_MODE=""
PAYLOAD=""
OUT=""

usage() { printf 'usage: build_deb.sh --repo <dir> (--tag <tag> | --commit <full-sha>) --payload <dir> --out <dir>\n' >&2; exit 2; }
log()  { printf '\n==> %s\n' "$*"; }
fail() { printf 'build_deb: FAILED: %s\n' "$*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo) REPO="${2:?}"; shift 2 ;;
        --tag) TAG="${2:?}"; shift 2 ;;
        --commit) COMMIT_MODE="${2:?}"; shift 2 ;;
        --payload) PAYLOAD="${2:?}"; shift 2 ;;
        --out) OUT="${2:?}"; shift 2 ;;
        *) usage ;;
    esac
done
[ -n "$REPO" ] && [ -n "$PAYLOAD" ] && [ -n "$OUT" ] || usage
{ [ -n "$TAG" ] || [ -n "$COMMIT_MODE" ]; } && { [ -z "$TAG" ] || [ -z "$COMMIT_MODE" ]; } || usage

for tool in python3 git docker dpkg-deb jq; do
    command -v "$tool" >/dev/null || fail "missing tool: $tool"
done

DIGEST="$(cd "$REPO_ROOT" && python3 -c 'from pm.lock import termux_docker_digest; print(termux_docker_digest())')"
[ -n "$DIGEST" ] || fail "termux-docker digest missing from pm/lock.json"
IMAGE="${TERMUX_BUILDER_IMAGE:-termux/termux-docker@$DIGEST}"

REPO_ABS="$(cd "$REPO" && pwd)"
PAYLOAD_ABS="$(cd "$PAYLOAD" && pwd)"

# Resolve source identity before writing output or changing payload files.
# Commit mode requires the checkout HEAD and staged version to agree.
if [ -n "$COMMIT_MODE" ]; then
    [[ "$COMMIT_MODE" =~ ^[a-f0-9]{40}$ ]] || fail "--commit requires an exact full 40-character SHA"
    COMMIT="$(git -C "$REPO_ABS" rev-parse HEAD)" || fail "not a git checkout: $REPO_ABS"
    [ "$COMMIT" = "$COMMIT_MODE" ] || fail "checkout HEAD $(echo "$COMMIT" | cut -c1-12) is not the requested commit"
    PY_VERSION="$(python3 - "$REPO_ROOT" "$REPO_ABS" "$COMMIT" "$PAYLOAD_ABS/app/pyproject.toml" <<'PY'
import sys, tomllib
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.releases.commit_build import version_at
version = version_at(Path(sys.argv[2]), sys.argv[3])
staged = tomllib.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))["project"]["version"]
if staged != version:
    raise ValueError("payload version does not match the admitted commit")
print(version)
PY
    )" || fail "commit version validation failed"
    DEB_VERSION="${PY_VERSION}+commit${COMMIT_MODE:0:12}"
    export HERMES_PAYLOAD_TAG=""
    export HERMES_BUILD_COMMIT="$COMMIT_MODE"
else
    unset HERMES_BUILD_COMMIT
    COMMIT="$(git -C "$REPO_ABS" rev-parse --verify "refs/tags/$TAG^{commit}")" \
        || fail "tag $TAG not found in $REPO_ABS"
fi
for d in python node uv npm ffmpeg ripgrep runtime-libs app wheelhouse; do
    [ -d "$PAYLOAD_ABS/$d" ] || fail "payload missing $d/ -- run termux_build.sh + build_cpython.sh + build_node.sh first"
done
PYBIN_REL="data/data/com.termux/files/usr/bin/python3.14"
[ -f "$PAYLOAD_ABS/python/$PYBIN_REL" ] || fail "payload python tree lacks $PYBIN_REL"
NODEBIN_REL="data/data/com.termux/files/usr/bin/node"
[ -f "$PAYLOAD_ABS/node/$NODEBIN_REL" ] || fail "payload node tree lacks $NODEBIN_REL"

PKG="hermes-agent"

# [1] Version derivation: tag mode uses the pure function in deb_version.py
# (tested separately). Commit mode derives it from pyproject above.
if [ -z "$COMMIT_MODE" ]; then
    log "Deriving Debian version from tag $TAG"
    DEB_VERSION="$(python3 "$HERE/deb_version.py" "$TAG")" || fail "version derivation failed for tag $TAG"
fi
log "Package version: $DEB_VERSION"
OUT_ABS="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"

# [2] Assemble the venv offline, INSIDE the pinned container: the staged
# interpreter is bionic/arm64 and cannot run on this host. Completeness is
# enforced by construction: --no-index means a missing wheel fails loudly.
# The container sees the payload at its real PREFIX path; the venv is built
# staged trees so the shipped venv's absolute shebangs point at the REAL
# $PREFIX path they will occupy on-device ($PREFIX is contractual).
log "Creating application and PM environments with the bundled CPython (inside the container)"
if [ -d "$PAYLOAD_ABS/venv" ]; then rm -rf "$PAYLOAD_ABS/venv"; fi
if [ -d "$PAYLOAD_ABS/pm-runtime" ]; then rm -rf "$PAYLOAD_ABS/pm-runtime"; fi
# The venv's dep list: the resolved graph with markers intact (the installer
# evaluates them on bionic) and documented android build misses skipped --
# uv pip check tolerates the app importing without them (its relay exporter
# is the only casualty). Generated host-side; consumed in-container.
python3 - "$HERE" "$PAYLOAD_ABS/.work/resolved.txt" "$PAYLOAD_ABS/.work/resolved-reqs.txt" <<'PYREQS' \
    || fail "deb-venv reqs generation failed"
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from build_wheels import write_reqs_file
write_reqs_file(Path(sys.argv[2]), Path(sys.argv[3]))
PYREQS
# The bind mount is runner-owned: the container (any uid) can only write
# into a dir the HOST pre-created with open perms (same as the wheelhouse).
mkdir -p "$PAYLOAD_ABS/venv" "$PAYLOAD_ABS/pm-runtime"
chmod 0777 "$PAYLOAD_ABS/venv" "$PAYLOAD_ABS/pm-runtime"
# Mount the payload at its REAL on-device path: the venv records
# absolute paths (interpreter symlink, pyvenv.cfg) that must be correct
# on-device from birth -- a /payload alias would bake container paths in.
docker run --rm --platform linux/arm64 \
    --user root --network none \
    -v "$PAYLOAD_ABS/python:/data/data/com.termux/files/usr/lib/hermes-agent/tools/python" \
    -v "$PAYLOAD_ABS/node:/data/data/com.termux/files/usr/lib/hermes-agent/tools/node" \
    -v "$PAYLOAD_ABS/uv:/data/data/com.termux/files/usr/lib/hermes-agent/tools/uv" \
    -v "$PAYLOAD_ABS/runtime-libs:/data/data/com.termux/files/usr/lib/hermes-agent/runtime-libs" \
    -v "$PAYLOAD_ABS/wheelhouse:/data/data/com.termux/files/usr/lib/hermes-agent/wheelhouse" \
    -v "$PAYLOAD_ABS/.work:/data/data/com.termux/files/usr/lib/hermes-agent/.work" \
    -v "$PAYLOAD_ABS/venv:/data/data/com.termux/files/usr/lib/hermes-agent/venv" \
    -v "$PAYLOAD_ABS/pm-runtime:/data/data/com.termux/files/usr/lib/hermes-agent/pm-runtime" \
    -v "$PAYLOAD_ABS/app:/data/data/com.termux/files/usr/lib/hermes-agent/app:ro" \
    "$IMAGE" bash -c '
        set -euo pipefail
        export PREFIX=/data/data/com.termux/files/usr
        export PATH="$PREFIX/bin:${PATH:-/usr/bin:/bin}"
        # The staged binary is dynamically linked against its OWN tree lib;
        # the container linker needs to be told where it lives (same fix as
        # the wheelhouse container half).
        export LD_LIBRARY_PATH="$PREFIX/lib/hermes-agent/tools/python$PREFIX/lib:$PREFIX/lib/hermes-agent/tools/node$PREFIX/lib:$PREFIX/lib/hermes-agent/runtime-libs/lib:$PREFIX/lib"
        # The staged tree is mounted at its REAL $PREFIX path so the venv
        # recorded absolute paths are correct on-device from birth.
        mkdir -p "$PREFIX" 2>/dev/null || true
        PY="$PREFIX/lib/hermes-agent/tools/python$PREFIX/bin/python3.14"
        UV="$PREFIX/lib/hermes-agent/tools/uv$PREFIX/bin/uv"
                # Desktop payload canon: the venv holds the DEPENDENCY tree only;
        # the app runs from its own directory via PYTHONPATH (the wheel
        # build is deliberately blocked in setup.py -- Hermes is not a
        # pip-installable package by design).
        # The payload is mounted at its ON-DEVICE path ($PREFIX/lib/
        # hermes-agent) so every absolute path the venv records --
        # interpreter symlink, pyvenv.cfg home -- is correct after
        # dpkg installs the tree to exactly that location.
        "$UV" venv --python "$PY" "$PREFIX/lib/hermes-agent/venv"
        # The dep graph with markers intact (the installer evaluates
        # them on bionic); documented android build misses skipped --
        # nemo-relay is the only casualty (the relay exporter).
        "$UV" pip install --python "$PREFIX/lib/hermes-agent/venv/bin/python" \
            --offline --no-index --only-binary :all: --find-links "$PREFIX/lib/hermes-agent/wheelhouse" \
            -r "$PREFIX/lib/hermes-agent/.work/resolved-reqs.txt"
        "$UV" pip check --python "$PREFIX/lib/hermes-agent/venv/bin/python"

        # PM uses the shared locked builder with the verified bionic wheelhouse.
        ROOT="$PREFIX/lib/hermes-agent"
        "$PY" -I -B -c "
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from pm.runtime_stage import stage_runtime
from scripts.bundles.payload import seal_pm_runtime
root, python, uv = map(Path, sys.argv[2:])
stage_runtime(uv, python, root / \"pm-runtime\", project=root / \"app/pm\",
              wheelhouse=root / \"wheelhouse\", offline=True)
seal_pm_runtime(root, python)
" "$ROOT/app" "$ROOT" "$PY" "$UV"
    ' || fail "venv assembly failed inside the container (offline wheelhouse install)"

# The install-method stamp (code-scoped, next to hermes_cli/): the deb IS
# the Termux apt distribution, and detect_install_method reads this marker
# to route hermes update -> pkg upgrade remediation.
printf 'apt\n' > "$PAYLOAD_ABS/app/.install_method"

# [3] Entry functions come from the archived project's script declarations.
log "Writing trampolines"
python3 "$HERE/launchers.py" --payload "$PAYLOAD_ABS"

# The shared stamp writer records the apt-termux update owner.
# Commit mode exports HERMES_BUILD_COMMIT and leaves the tag empty.
log "Writing app/install-stamp.json"
HERMES_PAYLOAD_TAG="$TAG" \
HERMES_DESKTOP_VARIANT=bundled \
python3 "$REPO_ABS/scripts/write_install_stamp.py" \
    --output "$PAYLOAD_ABS/app/install-stamp.json" \
    --commit "$COMMIT" \
    --distribution apt-termux \
    --update-mechanism external \
    --source bundle \
    || fail "stamp write failed"

# [5]+[6] Staging dir: DEBIAN/ control + payload under lib/hermes-agent/.
log "Staging the package tree"
STAGE="$OUT_ABS/.stage-$DEB_VERSION"
rm -rf "$STAGE"
# Termux debs store their payload at the ANDROID-fs-rooted on-device
# path (data/data/com.termux/files/usr/...): on-device dpkg extracts
# at / and only /data/data is writable -- a ./lib staging would make
# dpkg try to create /lib and fail on the read-only root.
ROOT_IN_DEB=data/data/com.termux/files/usr
DEST="$STAGE/$ROOT_IN_DEB/lib/hermes-agent"
mkdir -p "$STAGE/DEBIAN" "$DEST/tools"
for tool in python node uv npm ffmpeg ripgrep; do
    cp -a "$PAYLOAD_ABS/$tool" "$DEST/tools/"
done
cp -a "$PAYLOAD_ABS/runtime-libs" "$PAYLOAD_ABS/app" "$PAYLOAD_ABS/venv" "$PAYLOAD_ABS/pm-runtime" "$PAYLOAD_ABS/bin" "$DEST/"
python3 "$HERE/payload_facts.py" "$DEST" "$PAYLOAD_ABS/.work/build_set.txt"

python3 "$HERE/launchers.py" --payload "$DEST" --control "$STAGE/DEBIAN"

cat > "$STAGE/DEBIAN/control" <<EOF
Package: $PKG
Version: $DEB_VERSION
Architecture: aarch64
Maintainer: Nous Research
Description: Hermes Agent CLI for Termux (self-contained bundled python/node/venv)
Installed-Size: $(du -sk "$STAGE/$ROOT_IN_DEB" | cut -f1)
EOF
# Self-contained: no Depends line at all. Our python, node and venv ship inside.

# [7] Validation hook: install into a FRESH container of the pinned image and
# smoke-test the exact binaries the phone will run. No opt-out.
log "Validating in a fresh pinned termux-docker container"
DEB="$OUT_ABS/${PKG}_${DEB_VERSION}_aarch64.deb"
rm -f "$DEB"
# Use xz for both archive members so Termux dpkg can extract the package.
dpkg-deb --build -Zxz --root-owner-group "$STAGE" "$DEB" || fail "dpkg-deb --build failed"
rm -rf "$STAGE"
[ -f "$DEB" ] || fail "dpkg-deb did not produce $DEB"

# The bare rootfs has no build toolchain to hide a missing payload library.
docker run --rm --platform linux/arm64 \
    --user 1000:1000 --network none \
    -v "$DEB:/tmp/pkg.deb:ro" \
    -v "$HERE/check_deb.sh:/tmp/check.sh:ro" \
    -v "$HERE/validate_installed.py:/tmp/validate_installed.py:ro" \
    "termux/termux-docker@$DIGEST" bash -c \
        'source /tmp/check.sh; "$root/venv/bin/python" -m pm.cli status' \
    || fail "container validation failed"

log "Built $DEB (validated)"
