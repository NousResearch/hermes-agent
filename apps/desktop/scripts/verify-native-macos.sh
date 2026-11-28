#!/bin/sh
# Audit the actual installed payload, including nested helper binaries.
set -eu
app=${1:?Pass the path to the native Hermes.app}
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
desktop="$app/Contents/MacOS/hermes-desktop"
gateway="$app/Contents/MacOS/hermes-gateway"
test -x "$desktop"
test -x "$gateway"
plutil -lint "$app/Contents/Info.plist" >/dev/null
test "$(plutil -extract CFBundleExecutable raw -o - "$app/Contents/Info.plist")" = hermes-desktop
test "$(plutil -extract CFBundleDisplayName raw -o - "$app/Contents/Info.plist")" = Hermes
cmp "$app/Contents/Resources/Hermes.icns" "$root/assets/icon.icns"
codesign --verify --deep --strict "$app"

foreign=$(find "$app/Contents" \( -name '*.asar' -o -iname '*electron*' -o -iname '*chromium*' -o -iname '*tauri*' -o -iname '*webkit*' -o -iname '*python*' -o -iname node -o -iname node_modules -o -name '*.node' -o -name '*.py' -o -name '*.pyc' -o -name '*.html' -o -name '*.js' -o -name '*.wasm' \) -print)
if [ -n "$foreign" ]; then
    echo "Unexpected interpreter, web runtime, or script in application bundle" >&2
    echo "$foreign" >&2
    exit 1
fi
for executable in "$app/Contents/MacOS/"*; do
    file "$executable" | grep -q 'Mach-O'
    # The HTML artifact preview uses the operating system's own WKWebView
    # (/System/Library/Frameworks/WebKit.framework). Nothing web-related is bundled.
    if otool -L "$executable" | grep -v '^\s*/System/Library/Frameworks/WebKit.framework/' | grep -Ei 'Electron|Chromium|WebKit|JavaScriptCore|Python|libnode|libjvm'; then
        echo "Unexpected runtime dependency in $executable" >&2
        exit 1
    fi
done
printf 'Verified native executables, signing, official icon, and runtime dependencies in %s\n' "$app"
