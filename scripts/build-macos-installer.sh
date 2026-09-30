#!/usr/bin/env bash
# Buduje instalator Agenta Cześka dla macOS (DMG + ZIP) jedną komendą.
#
#   scripts/build-macos-installer.sh [--online] [--sign]
#
# Domyślnie DMG zawiera cały silnik (Python + zależności + kod): pierwszy start
# nie pobiera niczego i nie uruchamia pip. Architektura = architektura tego Maca
# (arm64 albo x64); zbuduj osobno na każdej. Wynik: apps/desktop/release/*.dmg
#
#   --online   mniejszy DMG bez silnika; silnik doinstaluje się przy pierwszym starcie (wymaga internetu)
#   --sign     podpisz certyfikatem Developer ID z kluczenika (bez tej flagi: podpis ad-hoc, do testów)
#   --dry-run  tylko sprawdź wymagania i pokaż plan
#
# Wymaga: macOS, Node.js >= 20, git, dostępu do internetu (uv pobiera przenośnego Pythona).
# Podpis i notaryzacja wydania publicznego: docs/RELEASE_SIGNING.md.

set -euo pipefail

BUNDLE=1
SIGN=0
DRY_RUN=0

say() { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

for arg in "$@"; do
    case "$arg" in
        --online) BUNDLE=0 ;;
        --sign) SIGN=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) fail "Nieznana opcja: $arg" ;;
    esac
done

[ "$(uname -s)" = "Darwin" ] || fail "Ten skrypt buduje instalator macOS i musi działać na Macu."
case "$(uname -m)" in
    arm64) ARCH=arm64 ;;
    x86_64) ARCH=x64 ;;
    *) fail "Nieobsługiwana architektura: $(uname -m)" ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$REPO/apps/desktop"

command -v node >/dev/null || fail "Brak Node.js (https://nodejs.org)."
command -v git >/dev/null || fail "Brak git (xcode-select --install)."
[ "$(node -p 'process.versions.node.split(".")[0]')" -ge 20 ] || fail "Potrzebny Node.js >= 20."

say "macOS-$ARCH · silnik: $([ "$BUNDLE" = 1 ] && echo wbudowany || echo 'pobierany przy pierwszym starcie') · podpis: $([ "$SIGN" = 1 ] && echo Developer\ ID || echo ad-hoc)"
if [ "$DRY_RUN" = 1 ]; then
    say "Tryb próbny — nic nie zostało zbudowane."
    exit 0
fi

cd "$REPO"
[ -d node_modules ] || { say "Instaluję zależności npm…"; npm ci; }

cd "$DESKTOP"
rm -rf build/runtime

if [ "$BUNDLE" = 1 ]; then
    "$REPO/scripts/stage-macos-runtime.sh"
fi

say "Buduję aplikację…"
npm run build

BUILDER_ARGS=(--mac dmg zip "--$ARCH")
if [ "$SIGN" = 1 ]; then
    export CSC_IDENTITY_AUTO_DISCOVERY=true
else
    # Apple silicon nie uruchomi całkiem niepodpisanej aplikacji; podpis ad-hoc wystarcza do testów lokalnych.
    export CSC_IDENTITY_AUTO_DISCOVERY=false
    BUILDER_ARGS+=(-c.mac.identity=-)
fi
npm run builder -- "${BUILDER_ARGS[@]}"

say "Gotowe:"
ls -1 "$DESKTOP"/release/*.dmg "$DESKTOP"/release/*.zip 2>/dev/null || true
[ "$SIGN" = 1 ] || say "Instalator jest podpisany ad-hoc: przy pierwszym uruchomieniu kliknij aplikację prawym → Otwórz (albo w Terminalu: xattr -dr com.apple.quarantine /Applications/*.app po skopiowaniu z DMG)."
