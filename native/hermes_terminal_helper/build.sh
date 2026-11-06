#!/bin/sh
# Build HermesTerminalHelper and install it next to the Hermes launcher
# (<repo>/.hermes/bin/) and into the user install bin (~/.hermes/bin/).
#
# The helper is a named, signed .app bundle so macOS TCC automation prompts
# attribute to "Hermes Terminal Helper" instead of the generic "osascript".
# See HermesTerminalHelper.swift for the full rationale.
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP_NAME="HermesTerminalHelper.app"
BIN_NAME="HermesTerminalHelper"
BUNDLE="$REPO_ROOT/.hermes/bin/$APP_NAME"
INSTALL_BUNDLE="$HOME/.hermes/bin/$APP_NAME"

SRC="$(dirname "$0")/$BIN_NAME.swift"
TMP_BIN="$(mktemp -d)/$BIN_NAME"

swiftc -O -o "$TMP_BIN" "$SRC"

# Rebuild the bundle in place (keeps Info.plist, replaces the binary).
mkdir -p "$BUNDLE/Contents/MacOS"
cp "$TMP_BIN" "$BUNDLE/Contents/MacOS/$BIN_NAME"
rm -f "$BUNDLE/Contents/MacOS/$BIN_NAME.swift"

# Ad-hoc sign: TCC persists the grant keyed on the code signature, and the
# bundle needs a signature for LaunchServices to report its display name.
codesign --force --deep --sign - "$BUNDLE"

# Install copy (shared across profiles; same family as ~/.hermes/bin/uv).
rm -rf "$INSTALL_BUNDLE"
cp -R "$BUNDLE" "$INSTALL_BUNDLE"
codesign --force --deep --sign - "$INSTALL_BUNDLE"

echo "installed: $INSTALL_BUNDLE"
echo "checkout:  $BUNDLE"
