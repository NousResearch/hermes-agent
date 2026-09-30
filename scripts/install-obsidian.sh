#!/usr/bin/env bash
# install-obsidian.sh — Obsidian dla Agenta Cześka na macOS (odpowiednik install-obsidian.ps1).
#
# Sprawdza, czy Obsidian.app już jest; jeśli nie, instaluje go z OFICJALNEGO źródła:
#   1) Homebrew: `brew install --cask obsidian` (gdy brew jest dostępny),
#   2) awaryjnie: oficjalny DMG z wydania obsidianmd/obsidian-releases (GitHub) →
#      kopia do ~/Applications (bez uprawnień administratora).
# Nie pakuje ani nie redystrybuuje binarek Obsidiana (licencja na to nie pozwala).
#
# Kontrakt (jak w wersji Windows):
#   exit 0 -> Obsidian był już zainstalowany albo udało się go zainstalować,
#   exit 2 -> nie udało się (brak internetu, blokada) — to NIE błąd instalacji Cześka.
#
#   install-obsidian.sh [--log PLIK]
# Zmienne dla testów: OBSIDIAN_RELEASE_JSON (zapisana odpowiedź API), OBSIDIAN_APP_DIRS (lista katalogów rozdzielona ':').

set -uo pipefail

LOG_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --log) LOG_FILE="${2:-}"; shift 2 ;;
        *) shift ;;
    esac
done

log() {
    local line
    line="[$(date +%Y-%m-%dT%H:%M:%S)] $*"
    printf '%s\n' "$line"
    if [ -n "$LOG_FILE" ]; then
        mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null && printf '%s\n' "$line" >> "$LOG_FILE" 2>/dev/null || true
    fi
}

IFS=':' read -r -a APP_DIRS <<< "${OBSIDIAN_APP_DIRS:-/Applications:$HOME/Applications}"

installed_app() {
    local dir
    for dir in "${APP_DIRS[@]}"; do
        if [ -d "$dir/Obsidian.app" ]; then printf '%s/Obsidian.app' "$dir"; return 0; fi
    done
    return 1
}

if found="$(installed_app)"; then
    log "Obsidian znaleziony: $found"
    exit 0
fi

if [ "${OBSIDIAN_SKIP_BREW:-0}" != 1 ] && command -v brew >/dev/null 2>&1; then
    log "Instaluję Obsidiana przez Homebrew…"
    if brew install --cask obsidian >>"${LOG_FILE:-/dev/null}" 2>&1 && installed_app >/dev/null; then
        log "Obsidian zainstalowany (Homebrew)."
        exit 0
    fi
    log "Homebrew nie zadziałał — próbuję oficjalnego DMG."
fi

if [ -n "${OBSIDIAN_RELEASE_JSON:-}" ]; then
    release="$(cat "$OBSIDIAN_RELEASE_JSON")"
else
    release="$(curl -fsSL -H 'Accept: application/vnd.github+json' \
        https://api.github.com/repos/obsidianmd/obsidian-releases/releases/latest)" || {
        log "Nie udało się pobrać informacji o wydaniu Obsidiana (brak internetu?)."
        exit 2
    }
fi

# Uniwersalny DMG (Intel + Apple silicon); nazwa: Obsidian-<wersja>.dmg.
url="$(printf '%s' "$release" \
    | grep -o '"browser_download_url"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | sed 's/.*:[[:space:]]*"\([^"]*\)"$/\1/' \
    | grep -E '/Obsidian-[0-9.]+\.dmg$' | head -1)"
if [ -z "$url" ]; then
    log "Wydanie Obsidiana nie zawiera pliku DMG."
    exit 2
fi

tmp="$(mktemp -d)"
mount="$tmp/mount"
cleanup() { hdiutil detach -quiet "$mount" >/dev/null 2>&1 || true; rm -rf "$tmp"; }
trap cleanup EXIT

log "Pobieram $url"
curl -fL --silent --show-error -o "$tmp/Obsidian.dmg" "$url" || { log "Pobieranie nie powiodło się."; exit 2; }
mkdir -p "$mount"
hdiutil attach -nobrowse -quiet -mountpoint "$mount" "$tmp/Obsidian.dmg" || { log "Nie udało się zamontować DMG."; exit 2; }
src="$(find "$mount" -maxdepth 1 -name '*.app' -print -quit)"
[ -n "$src" ] || { log "DMG nie zawiera aplikacji."; exit 2; }

target_dir="$HOME/Applications"
mkdir -p "$target_dir"
rm -rf "$target_dir/Obsidian.app"
cp -R "$src" "$target_dir/Obsidian.app" || { log "Kopiowanie do $target_dir nie powiodło się."; exit 2; }
log "Obsidian zainstalowany w $target_dir."
exit 0
