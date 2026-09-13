#!/usr/bin/env bash
# Android/Termux wheelhouse builder -- the ONE entry point. Two halves:
#
#   HOST half (glibc runner): tag gates, archive, uv resolve, marker-aware
#   PyPI probe -> resolved.txt + build_set.txt. Pure data work; arch-free.
#
#   CONTAINER half (bionic termux-docker): toolchain pins, the 13-ish native
#   sdist builds (clang/rust against the container's own termux python),
#   PEP 738 retag, the --no-index completeness gate, and the import gate.
#   The wheels MUST be bionic: building them on the glibc host would ship
#   glibc binaries that cannot exec on any phone.
#
# The script re-invokes ITSELF with --in-container inside the digest-pinned
# image (from pm/lock.json's termux-docker package). No opt-out flags.
#
# Inputs (host mode, all required):
#   --repo <dir>   hermes-agent checkout to build from (must contain the tag)
#   --tag <tag>    immutable release tag (vX.Y.Z or vX.Y.Z-canary.<ts>)
#   --out <dir>    output dir (wheelhouse/ + index.json + SHA256SUMS land here)

set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/termux/build_config.sh
. "$HERE/build_config.sh"

log() { printf '\n==> %s\n' "$*"; }
fail() { printf 'termux_build: FAILED: %s\n' "$*" >&2; exit 1; }

if [ "${1:-}" = "--in-container" ]; then
    # =================== CONTAINER HALF (bionic) =======================
    RESOLVED="$2"; BUILD_SET="$3"; WHEELHOUSE="$4"
    # Readers on the host must retain failed-build evidence too.
    umask 022
    # $5 (optional) is the staged payload root (mounted at /payload):
    # the wheels are built with THE PAYLOAD'S OWN python -- the TUR .deb
    # pm staged from the lock -- so the ABI is the shipped ABI by
    # construction and the offline gate installs into a venv of the very
    # interpreter the phone will run.
    PAYLOAD_ROOT="${5:-}"
    export PREFIX=/data/data/com.termux/files/usr
    STAGED_PY="$PAYLOAD_ROOT/python$PREFIX/bin/python3.14"
    # The staged binaries' RUNPATHs point at the phone's $PREFIX layout
    # (linkerconfig on-device). Inside the container the tree lives at
    # $PAYLOAD_ROOT/python$PREFIX, so the dynamic linker needs to be told
    # where the payload's libs live before any staged binary runs.
    export LD_LIBRARY_PATH="$PAYLOAD_ROOT/python$PREFIX/lib:$PAYLOAD_ROOT/node$PREFIX/lib:$PAYLOAD_ROOT/runtime-libs/lib:$PREFIX/lib"
    if [ -n "$PAYLOAD_ROOT" ] && [ -x "$STAGED_PY" ]; then
        PY="$STAGED_PY"
        log "Using the staged payload Python ($PY)"
    else
        # No payload staged (or missing): refuse. Building against any
        # other interpreter (the container's pkg python is 3.14!) ships
        # wheels the payload venv cannot install. No silent fallback.
        fail "staged payload python missing at $STAGED_PY -- stage the payload (pm lock rows) before the wheelhouse"
    fi
    # BUILD tools come from termux's own apt (the image is a bare
    # bootstrap). The USER machine never does any of this.
    if ! command -v clang >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
        log "Provisioning the container build toolchain (termux apt)"
        export DEBIAN_FRONTEND=noninteractive
        # Pin the OFFICIAL mirror and use apt DIRECTLY: pkg (the wrapper)
        # re-runs mirror selection and rewrote our pin to a desynced
        # third-party mirror mid-run (live 404 on libexpat). apt respects
        # sources.list as written. Retry the update for propagation windows.
        printf '%s\n' "deb https://packages.termux.dev/apt/termux-main stable main" \
            > "$PREFIX/etc/apt/sources.list"
        rm -f "$PREFIX/etc/apt/sources.list.d"/*.list 2>/dev/null || true
        apt update || apt update \
            || fail "apt update failed in the container"
        apt install -y clang rust make git patchelf binutils pkg-config protobuf cmake ninja autoconf automake libtool \
            libandroid-posix-semaphore libandroid-support libbz2 libffi \
            libjpeg-turbo libpng freetype libtiff libwebp openjpeg littlecms \
            libyaml openssl readline zlib liblzma libsqlite ncurses \
            || fail "apt install of the build toolchain failed"
    fi
    # BINARIES, not package names: the rust package provides rustc/cargo
    # (there is no `rust` binary).
    for tool in clang rustc cargo make git; do
        command -v "$tool" >/dev/null 2>&1 \
            || fail "container lacks $tool after provisioning"
    done
    # Serial builds only -- parallel Rust/C builds OOM arm runners.
    export CARGO_BUILD_JOBS=1
    export MAKEFLAGS=-j1
    # Native extension links need the STAGED payload's libpython: the
    # container's own $PREFIX/lib (bootstrap only) is on the default
    # -L path, but libpython3.14.so lives in the staged tree. setuptools
    # honors LDFLAGS, so every sdist build's link step finds it.
    STAGED_PYLIB="$PAYLOAD_ROOT/python$PREFIX/lib"
    export LDFLAGS="-L$STAGED_PYLIB ${LDFLAGS:-}"
    export CFLAGS="-I$PAYLOAD_ROOT/python$PREFIX/include ${CFLAGS:-}"
    # maturin (rust-backend sdists: cryptography, pydantic-core, ...) needs
    # the Android API level explicitly on a non-phone host. Matches the
    # wheel platform tag (android_24_arm64_v8a).
    export ANDROID_API_LEVEL=24
    # The container has no /bin/sh (termux's sh lives at $PREFIX/bin/sh);
    # scripts with #!/bin/sh shebangs (uvloop's libuv configure) exec-127.
    # Link the standard path into the container's namespace.
    if [ ! -e /bin/sh ]; then
        ln -s "$PREFIX/bin/sh" /bin/sh \
        || fail "could not link /bin/sh to the termux sh"
    fi
    # cargo composes its OWN link line (ignores LDFLAGS); pyo3 finds the
    # python BINARY via PYO3_PYTHON but the -lpython lib search path must
    # come through RUSTFLAGS, which cargo forwards to the linker.
    export RUSTFLAGS="-L$STAGED_PYLIB ${RUSTFLAGS:-}"
    # protoc-bin-vendored ships no android binary; the termux protobuf
    # package provides one, and PROTOC_BIN_PATH points vendored crates
    # at it (nemo-relay's worker-proto otherwise fails codegen).
    export PROTOC="$PREFIX/bin/protoc"
    export PROTOC_BIN_PATH="$PREFIX/bin/protoc"
    log "Preparing the scratch build environment through PM"
    mkdir -p "$PREFIX/tmp"
    BUILD_ROOT="$(mktemp -d "$PREFIX/tmp/hermes-build-XXXXXX")"
    BUILD_VENV="$BUILD_ROOT/venv"
    export HERMES_HOME="$BUILD_ROOT/home"
    export HERMES_RUNTIME_DIR="$BUILD_ROOT/tools"
    PYTHONPATH="$(cd "$HERE/../.." && pwd)"
    export PYTHONPATH
    "$PY" "$HERE/build_environment.py" prepare-tools \
        --root "$BUILD_ROOT" --source-tools "$PAYLOAD_ROOT"
    requirements=(--requirement pip==26.2.1 --requirement packaging==26.0)
    for requirement in "${TOOLCHAIN_PINS[@]}"; do
        requirements+=(--requirement "$requirement")
    done
    "$PY" -m pm.build_env --out "$BUILD_VENV" --python "$PY" "${requirements[@]}" \
        || fail "scratch build environment preparation failed"
    log "Building the android wheel set from sdist (bionic, payload ABI)"
    "$BUILD_VENV/bin/python" -u "$HERE/build_wheels.py" \
        --resolved "$RESOLVED" --build-set "$BUILD_SET" \
        --wheelhouse "$WHEELHOUSE" --retag "$HERE/retag_wheel.py" \
        --platform-tag "$PLATFORM_TAG" \
        || fail "wheel building failed"
    log "Wheelhouse container phase complete"
    # Root-owned outputs: make world-readable BEFORE exiting so the
    # host runner (different uid) can read/hash/publish them. The bind
    # mount's own top dir resists chmod (host-owned); only the FILES
    # need to be readable.
    find "$WHEELHOUSE" -type f -exec chmod a+r {} + 2>/dev/null || true
    exit 0
fi

# ===================== HOST HALF (glibc runner) ==========================
REPO=""
TAG=""
COMMIT_MODE=""
OUT=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo) REPO="${2:?}"; shift 2 ;;
        --tag) TAG="${2:?}"; shift 2 ;;
        --commit) COMMIT_MODE="${2:?}"; shift 2 ;;
        --out) OUT="${2:?}"; shift 2 ;;
        *) printf 'usage: termux_build.sh --repo <dir> (--tag <tag> | --commit <full-sha>) --out <dir>\n' >&2; exit 2 ;;
    esac
done
[ -n "$REPO" ] && [ -n "$OUT" ] && { [ -n "$TAG" ] || [ -n "$COMMIT_MODE" ]; } && { [ -z "$TAG" ] || [ -z "$COMMIT_MODE" ]; } || {
    printf 'usage: termux_build.sh --repo <dir> (--tag <tag> | --commit <full-sha>) --out <dir>\n' >&2; exit 2; }

for tool in git curl docker python3; do
    command -v "$tool" >/dev/null 2>&1 \
        || fail "missing tool: $tool (CI must provision the pinned toolchain before running this script)"
done

ARCH="$(uname -m)"
case "$ARCH" in
    aarch64|arm64) ;;
    *) fail "refusing to build on non-aarch64 host (uname -m: $ARCH)" ;;
esac

# Check source identity before writing build output.
if [ -n "$COMMIT_MODE" ]; then
    [[ "$COMMIT_MODE" =~ ^[a-f0-9]{40}$ ]] || fail "--commit requires an exact full 40-character SHA"
    [ "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT_MODE" ] || fail "checkout does not match --commit"
    log "Commit-only build of $COMMIT_MODE -- skipping the tag/release gates"
    REF="$COMMIT_MODE"
else
    log "Verifying release tag $TAG exists on origin"
    git -C "$REPO" ls-remote --exit-code --tags origin "$TAG" >/dev/null \
        || fail "tag $TAG not found on origin; refusing to build a mutable release"
    if command -v gh >/dev/null 2>&1; then
        gh release view "$TAG" --repo "$(git -C "$REPO" remote get-url origin | sed -e 's#.*github.com[:/]##' -e 's#\.git$##')" >/dev/null 2>&1 \
            || fail "release $TAG not found; refusing to build before the release exists"
    fi
    REF="$TAG"
fi

REPO_ABS="$(cd "$REPO" && pwd)"
OUT_ABS="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
WORK="$OUT_ABS/.work"
WHEELHOUSE="$OUT_ABS/wheelhouse"
# Preserve cached requirements and the native-build list until the manifest
# verifies both. Only the application archive changes for every release tag.
rm -rf "$WORK/tree"
mkdir -p "$WORK/tree" "$WHEELHOUSE"

# [c] Stage the tag as a gitless tree.
log "Archiving $REF into $WORK/tree"
python3 - "$REPO_ABS" "$REF" "$WORK/tree" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.bundles.payload import snapshot
snapshot(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
PY
[ -f "$WORK/tree/pyproject.toml" ] || fail "archived tag tree has no pyproject.toml -- bad tag?"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DIGEST="$(cd "$REPO_ROOT" && python3 -c 'from pm.lock import termux_docker_digest; print(termux_docker_digest())')"
[ -n "$DIGEST" ] || fail "termux-docker digest missing from pm/lock.json"
IMAGE="${TERMUX_BUILDER_IMAGE:-termux/termux-docker@$DIGEST}"
docker pull "$IMAGE"
IMAGE="$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"
CACHE_ARGS=(--payload "$OUT_ABS" --repo "$WORK/tree" --builder "$IMAGE"
            --platform-tag "$PLATFORM_TAG" --python-abi "$PYTHON_ABI")
WHEELHOUSE_CACHE_OK=0
if python3 "$HERE/wheelhouse_cache.py" check "${CACHE_ARGS[@]}"; then
    WHEELHOUSE_CACHE_OK=1
    log "Wheelhouse cache verified -- skipping resolve, probe, and native builds"
fi

if [ "$WHEELHOUSE_CACHE_OK" -eq 0 ]; then
# An unproven wheel must not reach the per-package skip check.
rm -rf "$WHEELHOUSE"
mkdir -p "$WHEELHOUSE"
rm -f "$OUT_ABS/index.json" "$OUT_ABS/SHA256SUMS" "$WORK/resolved.txt" "$WORK/build_set.txt"
# [d] Resolve the real graph from the tag's own lock.
log "Resolving dependency graph from the tag's uv.lock"
( cd "$REPO_ROOT" && python3 -m pm.build_env --source "$WORK/tree" \
    --extra acp --export-requirements "$WORK/req.txt" ) \
    || fail "PM requirements export failed (frozen lock at $REF)"
# Host parsing needs packaging too. Use the release lock, not runner packages.
PACKAGING_SPEC="$(python3 - "$WORK/tree/uv.lock" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as stream:
    packages = tomllib.load(stream)["package"]
package, = [item for item in packages if item["name"] == "packaging"]
print("packaging==" + package["version"])
PY
)" || fail "locked packaging dependency missing"
HOST_ENV="$(mktemp -d "$WORK/host-parser-XXXXXX")"
( cd "$REPO_ROOT" && python3 -m pm.build_env --out "$HOST_ENV/venv" \
    --python "$(command -v python3)" --requirement "$PACKAGING_SPEC" ) \
    || fail "host parser environment preparation failed"
HOST_PY=("$HOST_ENV/venv/bin/python")
RESOLVED="$WORK/resolved.txt"
"${HOST_PY[@]}" "$HERE/build_wheels.py" --normalize "$WORK/req.txt" "$WORK/tree/uv.lock" "$RESOLVED" \
    || fail "failed to normalize requirements"
[ -s "$RESOLVED" ] || fail "resolved dependency list is empty"

# [e] Marker-aware PyPI wheel-coverage probe: build set = resolved deps
# whose marker admits android AND that have no installable none-any wheel.
log "Probing PyPI wheel coverage (android markers)"
BUILD_SET="$WORK/build_set.txt"
"${HOST_PY[@]}" - "$RESOLVED" "$BUILD_SET" <<'PYEOF' || fail "PyPI wheel-coverage probe failed"
import json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

# The TARGET environment the wheelhouse must satisfy: Termux's bionic
# python. termux-main 3.14 reports sys.platform "android" (the linux value
# applied to 3.11/3.12; 3.13 changed it), so markers keying on
# `sys_platform == 'linux'` no longer admit the termux target -- and
# `platform_system` stays "Linux" (Android kernel), admitting
# platform_system-gated deps. windows/darwin markers exclude it as before.
TARGET_ENV = {
    "implementation_name": "cpython",
    "implementation_version": "3.14.6",
    "os_name": "posix",
    "platform_machine": "aarch64",
    "platform_release": "",
    "platform_system": "Linux",
    "platform_version": "",
    "python_full_version": "3.14.6",
    "python_version": "3.14",
    "sys_platform": "android",
}

def locked_version(spec: str) -> str | None:
    m = re.search(r"==\s*([A-Za-z0-9._+!-]+)", spec or "")
    return m.group(1) if m else None

def marker_admits(marker: str) -> bool:
    if not marker:
        return True
    from packaging.markers import Marker
    try:
        return Marker(marker).evaluate(TARGET_ENV)
    except Exception:
        # An unevaluable marker is a build-set decision, not a silent
        # exclude: admit it so the wheel build surfaces the truth loudly.
        return True

entries = []
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.strip():
        continue
    name, spec, marker, source = (line.rstrip("\n").split("\t") + [""])[:4]
    entries.append((name, spec, marker, source))

def probe(item):
    name, spec, marker, source = item
    if not marker_admits(marker):
        return name, None, None
    if source:
        return name, False, None
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=30) as r:
            d = json.load(r)
        locked = locked_version(spec) or d["info"]["version"]
        files = d["releases"].get(locked, [])
        # Coverage means INSTALLABLE on android/bionic, not "a wheel exists":
        # only py3-none-any wheels satisfy a package; anything else installs
        # nowhere on termux and must be built from sdist here.
        covered = any(
            f["filename"].endswith(".whl") and "-none-any.whl" in f["filename"]
            for f in files
        )
        return name, covered, None
    except Exception as exc:  # noqa: BLE001 -- a probe miss means BUILD it
        return name, False, str(exc)

with ThreadPoolExecutor(max_workers=10) as ex:
    results = list(ex.map(probe, entries))

# Documented android build misses: upstream packages whose vendored
# toolchain excludes android and cannot be built without a fork.
# nemo-relay: worker-proto uses protoc-bin-vendored, which ships no
# android protoc and fails codegen regardless of PROTOC* env (verified
# live twice). The .deb ships without the relay exporter.
ANDROID_BUILD_MISSES = {
    "nemo-relay": "protoc-bin-vendored ships no android protoc (upstream)",
}

needs_build = []
excluded = 0
missed = []
for name, covered, err in results:
    if covered is None:
        excluded += 1
        continue
    if err is not None:
        print(f"  probe miss {name}: {err} -> building from sdist", file=sys.stderr)
    if not covered:
        if name in ANDROID_BUILD_MISSES:
            missed.append((name, ANDROID_BUILD_MISSES[name]))
            continue
        needs_build.append(name)
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(needs_build) + "\n")
print(f"  {excluded} of {len(entries)} deps are marker-excluded for android")
for name, why in missed:
    print(f"  documented build miss: {name} ({why})")
print(f"  {len(needs_build)} of {len(entries) - excluded} applicable packages need sdist builds")
PYEOF
[ -s "$BUILD_SET" ] || fail "build set is empty -- nothing to build (probe bug?)"
fi

# [g] The build itself runs in the pinned container: the wheels must be
# bionic, and they are built with THE PAYLOAD'S OWN staged python (the
# exact TUR .deb pm staged from the lock), so the ABI matches the shipped
# interpreter by construction. The payload must be staged before this.
log "Building the wheelhouse inside the pinned container (payload ABI)"
PAYLOAD_ABS="$OUT_ABS"
[ -f "$PAYLOAD_ABS/python/data/data/com.termux/files/usr/bin/python3.14" ] \
    || fail "staged payload python missing -- run build_cpython.sh first (the wheelhouse builds with the payload interpreter)"
# The container mounts OUT_ABS at /out; translate the host-side work
# paths before crossing the boundary (host absolutes do not exist inside).
# The wheelhouse must be container-WRITABLE: /out is runner-owned, so
# pre-create the dir with open perms (the build writes wheels there).
mkdir -p "$OUT_ABS/wheelhouse"
chmod 0777 "$OUT_ABS/wheelhouse"

# The cache verdict was computed before [d]; on a hit the resolve, probe,
# and this container phase are all skipped (same flag guards each).
if [ "$WHEELHOUSE_CACHE_OK" -eq 0 ]; then
    C_RESOLVED="/out/.work/resolved.txt"
    C_BUILD_SET="/out/.work/build_set.txt"
    C_WHEELHOUSE="/out/wheelhouse"
    # --user root: the build must create /bin/sh (autotools config.sub,
    # configure shebangs expect the standard path); termux binaries run
    # fine as root in the build container (no phone-uid semantics here).
    docker run --rm --platform linux/arm64 \
        --user root \
        --tmpfs /bin \
        -v "$REPO_ROOT:/repo" \
        -v "$OUT_ABS:/out" \
        "$IMAGE" bash /repo/scripts/termux/termux_build.sh \
            --in-container "$C_RESOLVED" "$C_BUILD_SET" "$C_WHEELHOUSE" "/out" \
        || fail "container wheelhouse build failed"
fi

# [h+] Stage the archived tag tree as the payload's app/ -- build_deb.sh
# assembles the .deb from payload/{python,node,app,venv,bin}.
rm -rf "$OUT_ABS/app"
cp -a "$WORK/tree" "$OUT_ABS/app"

# [h] Only a successful native build and both gates can publish cache proof.
log "Emitting index.json and SHA256SUMS"
provenance=(--tag "$TAG")
if [ -n "$COMMIT_MODE" ]; then provenance=(--commit "$COMMIT_MODE"); fi
python3 "$HERE/wheelhouse_cache.py" write "${CACHE_ARGS[@]}" "${provenance[@]}" \
    || fail "manifest emission failed"

log "Wheelhouse complete: $WHEELHOUSE"
