#!/usr/bin/env bash
# Build script for Hermes Agent Android APK
# This script builds a standalone APK that can be downloaded and installed on Android devices

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
ANDROID_DIR="$REPO_ROOT/android"
BUILD_TOOLS_VERSION="34.0.0"
ANDROID_SDK_VERSION="34"
JDK_VERSION="17"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[+]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

error() {
    echo -e "${RED}[-]${NC} $1"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Build Hermes Agent Android APK

Options:
  --version VERSION    Set APK version (default: auto from git or timestamp)
  --debug             Build debug APK (default)
  --release           Build release APK (requires signing)
  --clean             Clean build before starting
  --help              Show this help message

Examples:
  $0 --version 1.0.0 --debug
  $0 --release --clean
EOF
}

# Parse arguments
VERSION=""
BUILD_TYPE="debug"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --debug)
            BUILD_TYPE="debug"
            shift
            ;;
        --release)
            BUILD_TYPE="release"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [ ! -d "$ANDROID_DIR" ]; then
    error "Android directory not found at $ANDROID_DIR"
    error "Please run this script from the repository root"
    exit 1
fi

# Set version if not provided
if [ -z "$VERSION" ]; then
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        # Try to get version from latest tag
        LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
        if [ -n "$LATEST_TAG" ]; then
            VERSION="${LATEST_TAG#v}"
        else
            # Use timestamp
            VERSION="1.0.0-$(date +%Y%m%d%H%M)"
        fi
    else
        VERSION="1.0.0-$(date +%Y%m%d%H%M)"
    fi
fi

log "Building Hermes Agent Android APK version: $VERSION"
log "Build type: $BUILD_TYPE"

# Check for Java
log "Checking Java..."
if ! command -v java &>/dev/null; then
    error "Java is not installed. Please install JDK $JDK_VERSION or later."
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | grep -E '^[openjdk|java] version' | awk '{print $3}' | tr -d '"' | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt "$JDK_VERSION" ]; then
    warn "Java version $JAVA_VERSION found. JDK $JDK_VERSION or later recommended."
fi

# Check for Android SDK
log "Checking Android SDK..."
if [ -z "$ANDROID_HOME" ]; then
    warn "ANDROID_HOME not set. Using default location."
    ANDROID_HOME="$HOME/Android/Sdk"
fi

if [ ! -d "$ANDROID_HOME" ]; then
    error "Android SDK not found at $ANDROID_HOME"
    error "Please install Android SDK with Android Studio or via command line"
    exit 1
fi

# Check for required Android SDK components
log "Checking Android SDK components..."

# Platform tools
if [ ! -d "$ANDROID_HOME/platform-tools" ]; then
    error "Android platform-tools not found"
    exit 1
fi

# Build tools
if [ ! -d "$ANDROID_HOME/build-tools/$BUILD_TOOLS_VERSION" ]; then
    warn "Android build-tools $BUILD_TOOLS_VERSION not found"
    warn "Attempting to install..."
    if ! yes | android update sdk --no-ui --all --filter "build-tools;$BUILD_TOOLS_VERSION"; then
        error "Failed to install build-tools"
        exit 1
    fi
fi

# Platform
if [ ! -d "$ANDROID_HOME/platforms/android-$ANDROID_SDK_VERSION" ]; then
    warn "Android platform $ANDROID_SDK_VERSION not found"
    warn "Attempting to install..."
    if ! yes | android update sdk --no-ui --all --filter "platforms;android-$ANDROID_SDK_VERSION"; then
        error "Failed to install platform"
        exit 1
    fi
fi

# Clean if requested
if [ "$CLEAN" = true ]; then
    log "Cleaning previous build..."
    cd "$ANDROID_DIR"
    ./gradlew clean --no-daemon
    rm -rf app/build
    rm -rf app/.cxx
fi

# Update version in build.gradle
log "Updating version to $VERSION..."
sed -i "s/versionCode [0-9]*/versionCode ${VERSION//./}/" "$ANDROID_DIR/app/build.gradle"
sed -i "s/versionName \"[^\"]*\"/versionName \"$VERSION\"/" "$ANDROID_DIR/app/build.gradle"

# Create keystore directory
log "Setting up keystore..."
mkdir -p "$ANDROID_DIR/app/keystore"

# Create debug keystore if it doesn't exist
if [ ! -f "$ANDROID_DIR/app/keystore/debug.keystore" ]; then
    log "Creating debug keystore..."
    keytool -genkey -v -keystore "$ANDROID_DIR/app/keystore/debug.keystore" \
        -storepass android -alias androiddebugkey -keypass android \
        -keyalg RSA -keysize 2048 -validity 10000 \
        -dname "CN=Android Debug,O=Android,C=US"
fi

# Build the APK
log "Building APK..."
cd "$ANDROID_DIR"

# Make gradle wrapper executable
chmod +x gradle/wrapper/gradle-wrapper.jar

# Build based on type
if [ "$BUILD_TYPE" = "debug" ]; then
    log "Building debug APK..."
    if ! ./gradlew assembleDebug --stacktrace --no-daemon; then
        error "Failed to build debug APK"
        exit 1
    fi
    APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
    APK_NAME="Hermes-Agent-${VERSION}-debug.apk"
elif [ "$BUILD_TYPE" = "release" ]; then
    log "Building release APK..."
    if [ ! -f "$ANDROID_DIR/app/keystore/release.keystore" ]; then
        error "Release keystore not found at $ANDROID_DIR/app/keystore/release.keystore"
        error "Please provide a valid release keystore for signing"
        exit 1
    fi
    if ! ./gradlew assembleRelease --stacktrace --no-daemon; then
        error "Failed to build release APK"
        exit 1
    fi
    APK_PATH="app/build/outputs/apk/release/app-release.apk"
    APK_NAME="Hermes-Agent-${VERSION}-release.apk"
fi

# Copy and rename APK
log "Copying APK to output directory..."
cp "$APK_PATH" "$REPO_ROOT/${APK_NAME}"

log "APK built successfully!"
echo ""
echo "APK location: $REPO_ROOT/${APK_NAME}"
echo "APK size: $(du -h "$REPO_ROOT/${APK_NAME}" | cut -f1)"
echo ""
echo "To install on Android device:"
echo "  1. Download the APK to your device"
echo "  2. Enable 'Install from unknown sources' in Settings"
echo "  3. Open the APK file and install"
echo ""
echo "Or use adb:"
echo "  adb install $REPO_ROOT/${APK_NAME}"

# Verify APK
if command -v aapt &>/dev/null; then
    log "Verifying APK..."
    aapt dump badging "$REPO_ROOT/${APK_NAME}"
fi

exit 0
