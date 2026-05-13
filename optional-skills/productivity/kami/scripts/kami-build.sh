#!/bin/bash
# Convenience wrapper for kami build commands on macOS Apple Silicon
# Sets DYLD_FALLBACK_LIBRARY_PATH for WeasyPrint system libraries

cd "$(dirname "$0")/.."

export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:/opt/homebrew/opt/glib/lib:/opt/homebrew/opt/pango/lib:/opt/homebrew/opt/harfbuzz/lib:/opt/homebrew/opt/cairo/lib:/opt/homebrew/opt/gdk-pixbuf/lib"

exec python3 scripts/build.py "$@"
