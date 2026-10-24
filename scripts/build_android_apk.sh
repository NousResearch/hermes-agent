#!/usr/bin/env bash
# Build the experimental Hermes Android APK. Requires a desktop JDK + Android SDK.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ANDROID_DIR="$REPO_ROOT/android"
BUILD_TOOLS_VERSION="34.0.0"
ANDROID_SDK_VERSION="34"
JDK_VERSION=17

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { printf '%b\n' "${GREEN}[+] ${1}${NC}"; }
warn() { printf '%b\n' "${YELLOW}[!] ${1}${NC}"; }
error() { printf '%b\n' "${RED}[-] ${1}${NC}" >&2; }

usage() {
    cat <<'EOF'
Usage: scripts/build_android_apk.sh [OPTIONS]
  --version VERSION   APK versionName (default: git tag or timestamp)
  --debug             Build debug APK (default)
  --release           Build release APK (requires a configured private keystore)
  --clean             Clean before building
  --help              Show this help
EOF
}

VERSION=""
BUILD_TYPE=debug
CLEAN=false
while (($#)); do
    case "$1" in
        --version)
            if (($# < 2)) || [[ -z "$2" ]]; then error "--version requires a value"; exit 2; fi
            VERSION="$2"; shift 2 ;;
        --debug) BUILD_TYPE=debug; shift ;;
        --release) BUILD_TYPE=release; shift ;;
        --clean) CLEAN=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) error "Unknown option: $1"; usage; exit 2 ;;
    esac
done

[[ -d "$ANDROID_DIR" ]] || { error "Android project not found: $ANDROID_DIR"; exit 1; }
[[ -f "$ANDROID_DIR/gradlew" ]] || { error "Gradle wrapper not found"; exit 1; }

if [[ -z "$VERSION" ]]; then
    if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        VERSION="$(git -C "$REPO_ROOT" describe --tags --abbrev=0 2>/dev/null || true)"
        VERSION="${VERSION#v}"
    fi
    [[ -n "$VERSION" ]] || VERSION="1.0.0-$(date +%Y%m%d%H%M)"
fi
[[ "$VERSION" =~ ^[0-9A-Za-z][0-9A-Za-z.+_-]*$ ]] || {
    error "Invalid version string: $VERSION"; exit 2;
}

log "Building Hermes Android APK version $VERSION ($BUILD_TYPE)"

command -v java >/dev/null 2>&1 || { error "Install JDK $JDK_VERSION or later"; exit 1; }
JAVA_OUTPUT="$(java -version 2>&1 | head -n 1)"
if [[ "$JAVA_OUTPUT" =~ version[[:space:]]+\"?([0-9]+) ]] ; then
    JAVA_VERSION="${BASH_REMATCH[1]}"
elif [[ "$JAVA_OUTPUT" =~ openjdk[[:space:]]+([0-9]+) ]]; then
    JAVA_VERSION="${BASH_REMATCH[1]}"
else
    error "Cannot determine Java version from: $JAVA_OUTPUT"; exit 1
fi
if (( JAVA_VERSION < JDK_VERSION )); then
    error "Java $JAVA_VERSION found; JDK $JDK_VERSION or later is required"; exit 1
fi

ANDROID_HOME="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}"
export ANDROID_HOME ANDROID_SDK_ROOT="$ANDROID_HOME"
[[ -d "$ANDROID_HOME" ]] || {
    error "Android SDK not found at $ANDROID_HOME. Install Android SDK platform 34 and build-tools 34.0.0."; exit 1;
}

SDKMANAGER=""
for candidate in "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" "$ANDROID_HOME/tools/bin/sdkmanager"; do
    if [[ -x "$candidate" ]]; then SDKMANAGER="$candidate"; break; fi
done
if [[ -z "$SDKMANAGER" ]] && command -v sdkmanager >/dev/null 2>&1; then SDKMANAGER="$(command -v sdkmanager)"; fi

if [[ ! -d "$ANDROID_HOME/platform-tools" || ! -d "$ANDROID_HOME/platforms/android-$ANDROID_SDK_VERSION" || ! -d "$ANDROID_HOME/build-tools/$BUILD_TOOLS_VERSION" ]]; then
    [[ -n "$SDKMANAGER" ]] || {
        error "Required Android SDK components are missing and sdkmanager was not found. Install Android SDK Command-line Tools."; exit 1;
    }
    log "Installing required SDK components with sdkmanager..."
    set +o pipefail
    yes | "$SDKMANAGER" --sdk_root="$ANDROID_HOME" \
        "platform-tools" "platforms;android-$ANDROID_SDK_VERSION" "build-tools;$BUILD_TOOLS_VERSION"
    SDK_STATUS=$?
    set -o pipefail
    (( SDK_STATUS == 0 )) || { error "sdkmanager failed to install required components"; exit 1; }
fi

if [[ "$BUILD_TYPE" == release ]]; then
    KEYSTORE_PATH="${ANDROID_KEYSTORE_PATH:-$ANDROID_DIR/app/keystore/hermes-release.keystore}"
    [[ -f "$KEYSTORE_PATH" ]] || {
        error "Release keystore missing. Set ANDROID_KEYSTORE_PATH or place it at android/app/keystore/hermes-release.keystore."; exit 1;
    }
    [[ -n "${ANDROID_STORE_PASSWORD:-}" && -n "${ANDROID_KEY_PASSWORD:-}" ]] || {
        error "Set ANDROID_STORE_PASSWORD and ANDROID_KEY_PASSWORD for a release build"; exit 1;
    }
    export ANDROID_KEYSTORE_PATH="$KEYSTORE_PATH"
fi

VERSION_CODE="$(git -C "$REPO_ROOT" rev-list --count HEAD 2>/dev/null || date +%s)"
[[ "$VERSION_CODE" =~ ^[0-9]+$ ]] || VERSION_CODE=1
(( VERSION_CODE > 0 && VERSION_CODE < 2147483647 )) || VERSION_CODE=1

cd "$ANDROID_DIR"
chmod +x ./gradlew

if [[ "$CLEAN" == true ]]; then
    log "Cleaning previous build..."
    ./gradlew clean --no-daemon
fi

GRADLE_ARGS=(-PandroidVersionName="$VERSION" -PandroidVersionCode="$VERSION_CODE" --stacktrace --no-daemon)
if [[ "$BUILD_TYPE" == debug ]]; then
    log "Building debug APK..."
    ./gradlew "${GRADLE_ARGS[@]}" assembleDebug
    APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
    APK_NAME="Hermes-Agent-${VERSION}-debug.apk"
else
    log "Building release APK..."
    ./gradlew "${GRADLE_ARGS[@]}" assembleRelease
    APK_PATH="app/build/outputs/apk/release/app-release.apk"
    APK_NAME="Hermes-Agent-${VERSION}-release.apk"
fi

[[ -s "$APK_PATH" ]] || { error "Gradle reported success but APK was not found at $APK_PATH"; exit 1; }
cp "$APK_PATH" "$REPO_ROOT/$APK_NAME"
log "APK built: $REPO_ROOT/$APK_NAME"
du -h "$REPO_ROOT/$APK_NAME"
if command -v aapt >/dev/null 2>&1; then aapt dump badging "$REPO_ROOT/$APK_NAME"; fi
