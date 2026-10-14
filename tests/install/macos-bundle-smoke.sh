#!/usr/bin/env bash
# Consumes receipt-verified bytes; never builds, repairs, or relaxes Gatekeeper.
set -euo pipefail
fail() { printf '%s\n' "bundle smoke: $*" >&2; exit 1; }
[[ ${GITHUB_ACTIONS:-} == true && ${RUNNER_ENVIRONMENT:-} == github-hosted && $(uname -s) == Darwin ]] ||
  fail 'Disposable native GitHub-hosted macOS runner required'

artifact= arch= commit= tag= channel_request= work= out=
while (($#)); do
  (($# >= 2)) || fail "Missing value for $1"
  case "$1" in
    --artifact) artifact=$2;; --arch) arch=$2;; --commit) commit=$2;;
    --channel-request) channel_request=$2;; --tag) tag=$2;; --work) work=$2;; --out) out=$2;;
    *) fail "Unknown argument: $1";;
  esac
  shift 2
done
[[ -f $artifact && $artifact == /* && -n $work && -n $out ]] || fail 'Absolute artifact, work and out are required'
[[ $arch == arm64 || $arch == x64 ]] || fail 'Expected arm64 or x64'
[[ $commit =~ ^[a-f0-9]{40}$ ]] || fail 'Expected exact lowercase full commit SHA'

# uname can report x86_64 under Rosetta; inspect the physical host as well.
host=x64
if [[ $(/usr/sbin/sysctl -in hw.optional.arm64) == 1 ]]; then host=arm64; fi
[[ $host == "$arch" ]] || fail "Requested $arch on native $host host"
[[ $(uname -m) == "${arch/x64/x86_64}" ]] || fail 'Adapter must not run under Rosetta'

assets=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/e2e-assets" && pwd -P)
node=$(command -v node)
[[ $("$node" -p 'process.arch') == "$arch" ]] || fail 'Driver Node must also use the native architecture'
metadata=$assets/bundle-smoke-metadata.mjs
identity_args=(--commit "$commit")
if [[ -n $tag ]]; then identity_args+=(--tag "$tag"); fi
if [[ -n $channel_request ]]; then identity_args+=(--channel-request "$channel_request"); fi
"$node" "$metadata" identity "${identity_args[@]}" >/dev/null
"$node" "$metadata" prepare --work "$work" --out "$out"
exec > >(tee "$out/native-install.log") 2>&1
mountpoint=
cleanup() {
  status=$?
  trap - EXIT
  if [[ -n $mountpoint ]]; then
    if ! /usr/bin/hdiutil detach "$mountpoint"; then status=1; fi
  fi
  printf '{"exitCode":%s}\n' "$status" > "$out/native-install-exit.json"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

/usr/bin/shasum -a 256 "$artifact" > "$out/artifact-before.sha256"
mkdir "$work/install" "$work/home" "$work/user-data"
shopt -s nullglob
case "$artifact" in
  *.dmg)
    mountpoint=$work/mount
    mkdir "$mountpoint"
    /usr/bin/hdiutil attach -readonly -nobrowse -noautoopen -mountpoint "$mountpoint" "$artifact"
    apps=("$mountpoint"/*.app)
    ((${#apps[@]} == 1)) || fail 'DMG must contain exactly one top-level .app'
    [[ -d ${apps[0]} && ! -L ${apps[0]} ]] || fail 'Application must be a directory, not a link'
    /usr/bin/ditto --rsrc --extattr --acl "${apps[0]}" "$work/install/$(basename "${apps[0]}")"
    /usr/bin/hdiutil detach "$mountpoint"
    mountpoint=
    ;;
  *.zip) /usr/bin/ditto -x -k --rsrc --extattr --acl "$artifact" "$work/install";;
  *) fail 'Expected a receipt-selected DMG or ZIP';;
esac
apps=("$work/install"/*.app)
((${#apps[@]} == 1)) || fail 'Installed artifact must contain exactly one top-level .app'
[[ -d ${apps[0]} && ! -L ${apps[0]} ]] || fail 'Installed application must not be a link'
"$node" "$metadata" verify-mac --app "${apps[0]}" --arch "$arch" "${identity_args[@]}" --out "$out/installed-identity.json"
exe=$("$node" -e 'console.log(require(process.argv[1]).exe)' "$out/installed-identity.json")
root=$("$node" -e 'console.log(require(process.argv[1]).root)' "$out/installed-identity.json")
/usr/bin/shasum -a 256 "$artifact" > "$out/artifact-after.sha256"
cmp "$out/artifact-before.sha256" "$out/artifact-after.sha256"
cd "$work"
"$node" "$assets/desktop-smoke.ts" --exe "$exe" --root "$root" --origin bundled \
  --home "$work/home" --user-data "$work/user-data" --out "$out" --phase installed --expect-commit "$commit"