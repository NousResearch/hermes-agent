#!/usr/bin/env bash
# Copy what tells a failed macOS in-app update apart into one directory: the
# OLD app crashing on quit, NEW crashing at launch, or a swap that never ran.
# ShipIt's own logs, crash reports, the installed bundle's version and
# signature, the app's logs and the unified log. It only reads and copies.
#
#   mac-update-evidence.sh --out <dir> --since <file> --app <installed .app>
#     [--home <dir>]... [--hermes-home <dir>] [--user-data <dir>]
#
# --since is a file touched before the update began; only crash reports newer
# than it are copied. Each --home is searched for Library/Caches/*.ShipIt and
# Library/Logs/DiagnosticReports: pass the sandbox HOME and the runner's own,
# because Squirrel and ReportCrash may not honour an overridden HOME.

# No -u: macOS /bin/bash 3.2 treats an empty "${HOMES[@]}" as unbound.
set -o pipefail

OUT="" SINCE="" APP="" HERMES_HOME_DIR="" USER_DATA=""
HOMES=()
while [ "$#" -gt 0 ]; do
  [ "$#" -ge 2 ] || { echo "mac-update-evidence: $1 needs a value" >&2; exit 2; }
  case "$1" in
    --out) OUT="$2" ;;
    --since) SINCE="$2" ;;
    --app) APP="$2" ;;
    --home) HOMES+=("$2") ;;
    --hermes-home) HERMES_HOME_DIR="$2" ;;
    --user-data) USER_DATA="$2" ;;
    *) echo "mac-update-evidence: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift 2
done
[ -n "$OUT" ] && [ -f "$SINCE" ] && [ -n "$APP" ] \
  || { echo "usage: mac-update-evidence.sh --out <dir> --since <file> --app <.app> [--home <dir>]..." >&2; exit 2; }

mkdir -p "$OUT/caches" "$OUT/crash-reports"

plist_value() { # $1: key
  command -v plutil >/dev/null 2>&1 || return 0
  plutil -extract "$1" raw -o - "$APP/Contents/Info.plist" 2>&1
}

# The update ShipIt unpacked sits inside these dirs: list it, copy only the small files.
index=0
for home in "${HOMES[@]}"; do
  index=$((index + 1))
  printf 'home%s %s\n' "$index" "$home" >> "$OUT/caches/homes.txt"
  for dir in "$home"/Library/Caches/*.ShipIt "$home"/Library/Caches/*updater*; do
    [ -d "$dir" ] || continue
    dest="$OUT/caches/home$index/$(basename "$dir")"
    mkdir -p "$dest"
    find "$dir" -maxdepth 3 -exec ls -ld {} + > "$dest/listing.txt" 2>&1
    find "$dir" -maxdepth 1 -type f \( -name '*.log' -o -name '*.plist' -o -name '*.json' \) \
      -exec cp -p {} "$dest/" \;
  done
done

report_dirs=()
for home in "${HOMES[@]}"; do report_dirs+=("$home/Library/Logs/DiagnosticReports"); done
report_dirs+=("/Library/Logs/DiagnosticReports")
for dir in "${report_dirs[@]}"; do
  [ -d "$dir" ] || continue
  find "$dir" -maxdepth 2 -type f \( -name '*.ips' -o -name '*.crash' -o -name '*.diag' \) \
    -newer "$SINCE" -exec cp -p {} "$OUT/crash-reports/" \;
done

if [ -n "$HERMES_HOME_DIR" ] && [ -d "$HERMES_HOME_DIR/logs" ]; then
  mkdir -p "$OUT/hermes-logs"
  find "$HERMES_HOME_DIR/logs" -maxdepth 1 -type f -name '*.log*' -exec cp -p {} "$OUT/hermes-logs/" \;
fi
if [ -n "$USER_DATA" ] && [ -d "$USER_DATA/Crashpad" ]; then
  find "$USER_DATA/Crashpad" -exec ls -ld {} + > "$OUT/crashpad-listing.txt" 2>&1
fi

{
  echo "app: $APP"
  ls -la "$(dirname "$APP")" 2>&1
  for key in CFBundleIdentifier CFBundleShortVersionString CFBundleVersion CFBundleExecutable; do
    printf '%s: %s\n' "$key" "$(plist_value "$key")"
  done
  for stamp in "$APP/Contents/Resources/install-stamp.json" "$APP/Contents/Resources/app-update.yml"; do
    [ -f "$stamp" ] && { echo "--- $stamp"; cat "$stamp"; echo; }
  done
  if command -v codesign >/dev/null 2>&1; then
    codesign --display --verbose=2 "$APP" 2>&1
    codesign --verify --deep --strict "$APP" 2>&1; echo "codesign verify rc=$?"
  fi
  if command -v spctl >/dev/null 2>&1; then
    spctl --assess --type execute -vv "$APP" 2>&1; echo "spctl rc=$?"
  fi
} > "$OUT/installed-app.txt" 2>&1

ps -axo pid=,ppid=,lstart=,command= 2>/dev/null | awk 'tolower($0) ~ /[h]ermes|[s]hipit/' > "$OUT/processes.txt"

if [ "$(uname -s)" = Darwin ] && command -v log >/dev/null 2>&1; then
  exe="$(plist_value CFBundleExecutable)"
  start="$(date -r "$SINCE" '+%Y-%m-%d %H:%M:%S')"
  log show --style compact --start "$start" --predicate \
    "process == \"ShipIt\" OR process == \"$exe\" OR process == \"ReportCrash\" OR ((process == \"syspolicyd\" OR process == \"amfid\") AND eventMessage CONTAINS[c] \"hermes\")" \
    2>&1 | tail -n 20000 > "$OUT/unified-log.txt"
fi

echo "--- installed app"
cat "$OUT/installed-app.txt"
echo "--- crash reports since the update began"
for report in "$OUT"/crash-reports/*; do
  [ -f "$report" ] || continue
  printf '%s: %s\n' "$(basename "$report")" "$(head -n 1 "$report" | cut -c1-400)"
done
for shipit_log in "$OUT"/caches/home*/*/*.log; do
  [ -f "$shipit_log" ] || continue
  echo "--- ${shipit_log#"$OUT"/} (last 40 lines)"
  tail -n 40 "$shipit_log"
done
echo "--- evidence saved under $OUT"
