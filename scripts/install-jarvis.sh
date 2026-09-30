#!/usr/bin/env bash
# AI Evolution Jarvis — instalacja jednym poleceniem (Linux i macOS).
#
#   curl -fsSL https://raw.githubusercontent.com/aievolutionpl/AGENT_CZESIEK/main/scripts/install-jarvis.sh | bash
#
# Pobiera najnowsze wydanie z GitHub Releases dla tego systemu i architektury,
# instaluje aplikację bez uprawnień administratora i ją uruchamia. Ikonę na
# pulpicie aplikacja zakłada sama przy pierwszym starcie (docs/product/
# AI_EVOLUTION_JARVIS_DESIGN.md §12), więc skrypt jej nie dubluje.
#
# Opcje:
#   --version TAG   konkretne wydanie zamiast najnowszego (np. v0.17.2)
#   --no-launch     nie uruchamiaj aplikacji po instalacji
#   --dry-run       pokaż, co zostałoby pobrane i gdzie, bez zmian na dysku
#
# Zmienne (dla testów i luster):
#   JARVIS_REPO           właściciel/repozytorium (domyślnie aievolutionpl/AGENT_CZESIEK)
#   JARVIS_RELEASE_JSON   ścieżka do zapisanej odpowiedzi API wydania zamiast zapytania sieciowego
#   JARVIS_OS / JARVIS_ARCH  wymuszenie platformy (linux|mac, x64|arm64)
#   JARVIS_ICON_URL       skąd pobrać ikonę do menu (Linux); domyślnie ikona z tego samego wydania

set -euo pipefail

REPO="${JARVIS_REPO:-aievolutionpl/AGENT_CZESIEK}"
VERSION="latest"
LAUNCH=1
DRY_RUN=0
APP_NAME="AI Evolution Jarvis"
APP_ID="pl.aievolution.jarvis"

say() { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
step() { printf '\n\033[1;35m[%s/4]\033[0m \033[1m%s\033[0m\n' "$1" "$2"; }
ok() { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

next_steps() {
    cat <<'NEXT'

  ┌──────────────────────────────────────────────────────────────┐
  │  Co dalej                                                    │
  │                                                              │
  │  1. Jarvis otworzy kreator: Jak działa → Silnik → Głos …     │
  │  2. Najprostszy start: klucz OpenRouter (openrouter.ai/keys) │
  │     — jeden klucz daje GPT, Claude, Gemini i DeepSeek.       │
  │  3. W kroku „Połączenia i API” wybierz Google, e-mail,       │
  │     komunikatory — Jarvis poprowadzi Cię krok po kroku.      │
  │                                                              │
  │  Ikona Jarvisa jest w menu aplikacji, a skrót na pulpicie    │
  │  pojawi się przy pierwszym uruchomieniu.                     │
  └──────────────────────────────────────────────────────────────┘
NEXT
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="${2:?--version wymaga wartości}"; shift 2 ;;
        --no-launch) LAUNCH=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) fail "Nieznana opcja: $1" ;;
    esac
done

detect_os() {
    if [ -n "${JARVIS_OS:-}" ]; then printf '%s' "$JARVIS_OS"; return; fi
    case "$(uname -s)" in
        Linux) printf 'linux' ;;
        Darwin) printf 'mac' ;;
        *) fail "Ten skrypt obsługuje Linux i macOS. Na Windowsie użyj install-jarvis.ps1." ;;
    esac
}

detect_arch() {
    if [ -n "${JARVIS_ARCH:-}" ]; then printf '%s' "$JARVIS_ARCH"; return; fi
    case "$(uname -m)" in
        x86_64|amd64) printf 'x64' ;;
        arm64|aarch64) printf 'arm64' ;;
        *) fail "Nieobsługiwana architektura: $(uname -m)" ;;
    esac
}

release_json() {
    if [ -n "${JARVIS_RELEASE_JSON:-}" ]; then
        cat "$JARVIS_RELEASE_JSON"
        return
    fi

    local url
    if [ "$VERSION" = "latest" ]; then
        url="https://api.github.com/repos/$REPO/releases/latest"
    else
        url="https://api.github.com/repos/$REPO/releases/tags/$VERSION"
    fi

    curl -fsSL -H 'Accept: application/vnd.github+json' "$url" \
        || fail "Nie udało się pobrać informacji o wydaniu ($url)."
}

# Every download URL in the release, one per line. The API answers with
# compact JSON (many assets on one line), so match each field on its own.
# grep + sed only: the script needs nothing beyond curl and coreutils.
asset_urls() {
    grep -o '"browser_download_url"[[:space:]]*:[[:space:]]*"[^"]*"' \
        | sed 's/.*:[[:space:]]*"\([^"]*\)"$/\1/' || true
}

# The first URL matching an extended regex, or nothing.
pick_asset() {
    grep -E -e "$1" | head -1 || true
}

step 1 "Sprawdzam system i wydanie"
OS="$(detect_os)"
ARCH="$(detect_arch)"
RELEASE="$(release_json)"
URLS="$(printf '%s' "$RELEASE" | asset_urls)"
[ -n "$URLS" ] || fail "Wydanie nie zawiera plików do pobrania."
TAG="$(printf '%s' "$RELEASE" | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*:[[:space:]]*"\([^"]*\)"$/\1/' || true)"
# The icon of the very release being installed, so menu and app never disagree.
ICON_URL="${JARVIS_ICON_URL:-https://raw.githubusercontent.com/$REPO/${TAG:-main}/apps/desktop/assets/icon.png}"

if [ "$OS" = "linux" ]; then
    # AppImage works on every distribution without root; x86_64 is
    # electron-builder's other spelling of x64.
    if [ "$ARCH" = "x64" ]; then arch_re='(x64|x86_64|amd64)'; else arch_re='(arm64|aarch64)'; fi
    ASSET="$(printf '%s\n' "$URLS" | pick_asset "-linux-${arch_re}\\.AppImage$")"
    [ -n "$ASSET" ] || fail "Brak pliku AppImage dla linux-$ARCH w tym wydaniu."
    TARGET="${XDG_DATA_HOME:-$HOME/.local/share}/ai-evolution-jarvis/AI-Evolution-Jarvis.AppImage"
    ICON_FILE="$(dirname "$TARGET")/icon.png"
    MENU_ENTRY="${XDG_DATA_HOME:-$HOME/.local/share}/applications/$APP_ID.desktop"
else
    ASSET="$(printf '%s\n' "$URLS" | pick_asset "-mac-${ARCH}\\.dmg$")"
    # One universal/x64 build still runs on Apple silicon through Rosetta.
    [ -n "$ASSET" ] || ASSET="$(printf '%s\n' "$URLS" | pick_asset '-mac-(universal|x64)\.dmg$')"
    [ -n "$ASSET" ] || fail "Brak pliku DMG dla macOS w tym wydaniu."
    if [ -w /Applications ]; then TARGET="/Applications/$APP_NAME.app"; else TARGET="$HOME/Applications/$APP_NAME.app"; fi
    MENU_ENTRY=""
fi

say "System: $OS-$ARCH${TAG:+ · wydanie $TAG}"
say "Pobieram: $ASSET"
say "Instaluję do: $TARGET"
[ -e "$TARGET" ] && say "Jarvis jest już zainstalowany — zaktualizuję go, Twoje dane zostają."

if [ "$DRY_RUN" = 1 ]; then
    if [ -n "$MENU_ENTRY" ]; then
        say "Wpis w menu: $MENU_ENTRY"
        say "Ikona: $ICON_URL → $ICON_FILE"
    fi
    say "Tryb próbny — nic nie zostało zmienione."
    exit 0
fi

step 2 "Pobieram Jarvisa"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FILE="$TMP/$(basename "$ASSET")"
curl -fL --progress-bar -o "$FILE" "$ASSET" || fail "Pobieranie nie powiodło się. Sprawdź połączenie z internetem i spróbuj ponownie."
ok "Pobrano."

step 3 "Instaluję"

if [ "$OS" = "linux" ]; then
    mkdir -p "$(dirname "$TARGET")" "$(dirname "$MENU_ENTRY")"
    mv "$FILE" "$TARGET"
    chmod +x "$TARGET"
    ok "Aplikacja: $TARGET"

    step 4 "Ikona i wpis w menu"
    # An absolute Icon= path: nothing else installs a themed icon for an
    # AppImage, and a menu entry without one shows a blank square.
    if curl -fsSL -o "$ICON_FILE" "$ICON_URL"; then
        ICON_REF="$ICON_FILE"
    else
        warn "Nie udało się pobrać ikony — wpis w menu użyje ikony motywu."
        ICON_REF="ai-evolution-jarvis"
    fi
    cat > "$MENU_ENTRY" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=AI Evolution Jarvis — asystent AI
Exec="$TARGET" %U
Icon=$ICON_REF
Terminal=false
Categories=Utility;Office;
Keywords=ai;asystent;assistant;jarvis;hermes;agent;
StartupWMClass=$APP_NAME
EOF
    chmod +x "$MENU_ENTRY"
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$(dirname "$MENU_ENTRY")" >/dev/null 2>&1 || true
    ok "Gotowe. Jarvis jest w menu aplikacji."
    next_steps
    if [ "$LAUNCH" = 1 ]; then
        nohup "$TARGET" >/dev/null 2>&1 &
        say "Uruchamiam $APP_NAME…"
    fi
else
    MOUNT="$TMP/mount"
    mkdir -p "$MOUNT" "$(dirname "$TARGET")"
    hdiutil attach -nobrowse -quiet -mountpoint "$MOUNT" "$FILE" || fail "Nie udało się zamontować obrazu DMG."
    SOURCE="$(find "$MOUNT" -maxdepth 1 -name '*.app' -print -quit)"
    [ -n "$SOURCE" ] || { hdiutil detach -quiet "$MOUNT" || true; fail "Obraz DMG nie zawiera aplikacji."; }
    rm -rf "$TARGET"
    cp -R "$SOURCE" "$TARGET"
    hdiutil detach -quiet "$MOUNT" || true
    # Plik pobrany przez curl nie dostaje flagi kwarantanny, ale kopia z innego źródła mogłaby ją mieć.
    xattr -dr com.apple.quarantine "$TARGET" 2>/dev/null || true
    step 4 "Launchpad i Dock"
    ok "Gotowe. Jarvis jest w $(dirname "$TARGET") i w Launchpadzie."
    say "Wskazówka: po uruchomieniu kliknij ikonę w Docku prawym przyciskiem → Opcje → Zachowaj w Docku."
    next_steps
    if [ "$LAUNCH" = 1 ]; then
        open "$TARGET"
        say "Uruchamiam $APP_NAME…"
    fi
fi
