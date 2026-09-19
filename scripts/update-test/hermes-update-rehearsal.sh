#!/usr/bin/env bash
# hermes-update-rehearsal.sh — for an EXISTING Hermes install.
#
# Two steps:
#   pre   back up your ENTIRE HERMES_HOME and the desktop app's Electron
#         userData, then point the install's update source at a custom repo +
#         ref so `hermes update` pulls it. Prints what to do next.
#   post  wipe both trees and put the backup back exactly as it was.
#
# Plus `status`, which only prints. This script never judges your install: it
# reports what it did and stops. Whether the update worked is for you to see.
#
# The backup is one plain tar per tree with nothing filtered out, and `post`
# restores those tars over empty directories, so every file comes back as it was.
#
#   ./hermes-update-rehearsal.sh pre  --source <git-url> --ref <branch-or-tag>
#   # ... run `hermes update`, use Hermes, test whatever you need ...
#   ./hermes-update-rehearsal.sh post
#
# Options:
#   --source URL   repo to pull the update from (default: the rehearsal fork)
#   --ref REV      branch or tag in that repo (default: main)
#   --backup-root DIR   where the backup lives (default ~/hermes-update-rehearsal)
#   --yes          post: skip the confirmation
#
# Requires: tar and git. `pre` needs network access to --source.

set -euo pipefail

OFFICIAL_HTTPS="https://github.com/NousResearch/hermes-agent.git"
OFFICIAL_SSH="git@github.com:NousResearch/hermes-agent.git"
DEFAULT_SOURCE="https://github.com/ethernet8023/hermes-agent.git"
DEFAULT_REF="main"

SUBCMD=""
BACKUP_ROOT="${HOME}/hermes-update-rehearsal"
SOURCE="$DEFAULT_SOURCE"
REF="$DEFAULT_REF"
ASSUME_YES=0

say()  { printf '%s\n' "$*"; }
ok()   { printf '  OK %s\n' "$*"; }
warn() { printf '  WARN %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
step() { printf '\n=== %s ===\n' "$*"; }

usage() {
  if [ -n "$SELF" ] && [ -r "$SELF" ]; then sed -n '2,26p' "$SELF"; exit 0; fi
  cat <<'EOF'
hermes-update-rehearsal.sh -- run against an EXISTING Hermes install.

  pre     back up everything, then point the update source at a custom repo+ref
  post    wipe both trees and restore the backup exactly as it was
  status  print what is prepared (read-only; nothing is touched)

Options:
  --source URL        repo to pull the update from (default: the rehearsal fork)
  --ref REV           branch or tag in that repo (default: main)
  --backup-root DIR   where the backup lives (default ~/hermes-update-rehearsal)
  --yes               post: skip the confirmation

`pre` reports what it did and prints the next commands. It never judges your
install; whether the update worked is for you to see.
EOF
  exit 0
}

# How to name ourselves in the instructions we print. Piped into a shell
# (`curl ... | bash -s -- pre`) there is no path to point at and "$0" is just
# "bash", which would print a command that does not exist.
SELF="$0"
case "$(basename "$SELF")" in bash|sh|dash|zsh|-bash|-sh|'') SELF="" ;; esac
case "$SELF" in /dev/fd/*|/dev/stdin|/proc/self/fd/*) SELF="" ;; esac

while [ "$#" -gt 0 ]; do
  case "$1" in
    pre|post|status) SUBCMD="$1"; shift ;;
    --source)      [ "$#" -ge 2 ] || die "--source needs a value"; SOURCE="$2"; shift 2 ;;
    --ref)         [ "$#" -ge 2 ] || die "--ref needs a value";    REF="$2";    shift 2 ;;
    --backup-root) [ "$#" -ge 2 ] || die "--backup-root needs a value"; BACKUP_ROOT="$2"; shift 2 ;;
    --yes|-y)      ASSUME_YES=1; shift ;;
    -h|--help)     usage ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -n "$SUBCMD" ] || usage

command -v git >/dev/null 2>&1 || die "git is required"
command -v tar >/dev/null 2>&1 || die "tar is required"

SNAP=""

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

# Under git-bash/cygwin, POSIX paths are not translated for native tools
# (git.exe, tar.exe), so normalise them. On macOS cygpath is absent: no-op.
native_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$1" 2>/dev/null || printf '%s' "$1"
  else
    printf '%s' "$1"
  fi
}

resolve_paths() {
  local suffix="${HERMES_DATA_DIR_SUFFIX:-}"
  if [ -n "${HERMES_HOME:-}" ]; then
    HERMES_HOME="$(cd "$HERMES_HOME" 2>/dev/null && pwd || printf '%s' "$HERMES_HOME")"
  else
    HERMES_HOME="$HOME/.hermes${suffix}"
  fi
  INSTALL_DIR="$HERMES_HOME/hermes-agent"
  if [ -n "${HERMES_DESKTOP_USER_DATA_DIR:-}" ]; then
    USERDATA_DIR="$HERMES_DESKTOP_USER_DATA_DIR"; USERDATA_SOURCE="env"
  else
    USERDATA_DIR="$HOME/Library/Application Support/Hermes${suffix}"; USERDATA_SOURCE="default"
  fi
  BACKUP_ROOT="$(native_path "$BACKUP_ROOT")"
  HERMES_HOME="$(native_path "$HERMES_HOME")"
  INSTALL_DIR="$(native_path "$INSTALL_DIR")"
  USERDATA_DIR="$(native_path "$USERDATA_DIR")"
}

# `file:///C:/x` on Windows, `file:///home/x` elsewhere.
file_url() {
  case "$1" in
    [A-Za-z]:*) printf 'file:///%s' "$1" ;;
    *)          printf 'file://%s' "$1" ;;
  esac
}

latest_backup_root() {
  [ -d "$BACKUP_ROOT" ] || return 1
  local newest
  newest="$(ls -1d "$BACKUP_ROOT"/*/ 2>/dev/null | sort | tail -1 || true)"
  [ -n "$newest" ] || return 1
  printf '%s' "${newest%/}"
}

load_snapshot() {
  SNAP="$(latest_backup_root)" || die "no backup found under $BACKUP_ROOT — run 'pre' first"
  local recorded_path="$SNAP/hermes-home.txt"
  [ -f "$recorded_path" ] || die "$SNAP is not a rehearsal backup (no hermes-home.txt)"
  # Plain text, not JSON: this string is compared byte-for-byte to decide whether
  # the backup belongs to the home post is about to wipe.
  local recorded current
  recorded="$(tr -d '\r' < "$recorded_path")"
  current="$(printf '%s' "$HERMES_HOME" | tr '\\' '/')"
  [ "$(printf '%s' "$recorded" | tr '\\' '/')" = "$current" ] \
    || die "that backup belongs to HERMES_HOME=$recorded, not $HERMES_HOME; pass --backup-root to pick the right one"
}

# ---------------------------------------------------------------------------
# pre: back up, then point the install at the rehearsal source
# ---------------------------------------------------------------------------

cmd_pre() {
  resolve_paths
  step "your install"
  say "HERMES_HOME   $HERMES_HOME"
  say "install       $INSTALL_DIR"
  say "desktop data  $USERDATA_DIR ($USERDATA_SOURCE)"
  say "backup to     $BACKUP_ROOT"
  [ -d "$HERMES_HOME" ] || die "no HERMES_HOME at $HERMES_HOME"
  [ -d "$INSTALL_DIR/.git" ] || die "no git checkout at $INSTALL_DIR — this tool covers source installs"

  step "before we start (nothing here is a pass/fail, just read it)"
  if command -v pgrep >/dev/null 2>&1; then
    local procs
    procs="$(pgrep -fl hermes 2>/dev/null | grep -v 'hermes-update-rehearsal' || true)"
    if [ -n "$procs" ]; then
      warn "Hermes looks like it is running — close the desktop app and the gateway"
      warn "before you run 'hermes update', or the dependency sync may fail:"
      printf '    %s\n' "$procs"
    else
      ok "no Hermes processes running"
    fi
  fi
  local n
  n="$( { git config --global --get-regexp '^url\.' 2>/dev/null || true; git -C "$INSTALL_DIR" config --local --get-regexp '^url\.' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  if [ "$n" = 0 ]; then ok "global git config has no URL rewrites"
  else warn "$n existing url.* insteadOf entr(y/ies) in your git config; we add more and remove only ours"; fi

  SNAP="$BACKUP_ROOT/$(date -u +%Y%m%dT%H%M%SZ)"
  [ ! -e "$SNAP" ] || die "backup dir already exists: $SNAP"
  mkdir -p "$SNAP"

  step "backing up your entire HERMES_HOME"
  local started elapsed tar_pid
  started="$SECONDS"
  # No excludes: checkout, venv, PM store and node_modules come too, so `post`
  # is a true rollback rather than a re-download. SQLite sidecars travel WITH
  # their db on purpose (a raw copy of db+wal+shm is consistent).
  # No -z: the backup root is typically the same internal disk, so the size
  # saving buys nothing and gzip costs ~5x the wall time on a checkout-sized
  # tree (72s vs 14s measured on an M1). bsdtar (macOS) and GNU tar differ on
  # progress options, so poll the growing archive instead — portable, and it
  # doubles as a liveness signal.
  tar -cf "$SNAP/hermes-home.tar" -C "$HERMES_HOME" . &
  tar_pid=$!
  while kill -0 "$tar_pid" 2>/dev/null; do
    printf '\r    %s written...  ' "$(du -h "$SNAP/hermes-home.tar" 2>/dev/null | cut -f1)"
    sleep 2
  done
  printf '\r    \r'
  wait "$tar_pid" || die "backup failed (tar)"
  elapsed=$((SECONDS - started))
  ok "hermes-home.tar ($(du -h "$SNAP/hermes-home.tar" | cut -f1), ${elapsed}s)"

  step "backing up the desktop app's data"
  if [ -d "$USERDATA_DIR" ]; then
    tar -cf "$SNAP/electron-userdata.tar" -C "$USERDATA_DIR" . || die "userData backup failed"
    ok "electron-userdata.tar ($(du -h "$SNAP/electron-userdata.tar" | cut -f1))"
  else
    warn "no Electron userData at $USERDATA_DIR (desktop app not installed?)"
  fi

  step "recording what this backup is"
  printf '%s\n' "$HERMES_HOME" > "$SNAP/hermes-home.txt"
  cat > "$SNAP/manifest.json" <<EOF
{
  "schema": 3,
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hermes_home": "$HERMES_HOME",
  "install_dir": "$INSTALL_DIR",
  "userdata_dir": "$USERDATA_DIR",
  "userdata_dir_source": "$USERDATA_SOURCE",
  "rehearsal_source": "$SOURCE",
  "rehearsal_ref": "$REF"
}
EOF
  ok "manifest.json (backup of $HERMES_HOME)"

  # --- point the install at the rehearsal source ---------------------------
  step "fetching the rehearsal source"
  say "source        $SOURCE"
  say "ref           $REF"
  local serve="$SNAP/serve.git" target_sha=""
  rm -rf "$serve"
  # Prefer a single-branch clone (fast); fall back to a full bare clone so a
  # raw commit in --ref still resolves.
  if git clone --quiet --bare --branch "$REF" --single-branch "$SOURCE" "$serve" 2>/dev/null; then
    ok "cloned $REF"
  else
    rm -rf "$serve"
    git clone --quiet --bare "$SOURCE" "$serve" \
      || die "could not clone $SOURCE (network? permissions? bad --source?)"
    ok "cloned the whole repo (--ref '$REF' is not a branch/tag name)"
  fi
  target_sha="$(git -C "$serve" rev-parse --verify "${REF}^{commit}" 2>/dev/null || true)"
  [ -n "$target_sha" ] || die "--ref '$REF' was not found in $SOURCE"
  git -C "$serve" update-ref refs/heads/main "$target_sha"
  git -C "$serve" symbolic-ref HEAD refs/heads/main
  # The updater may ask for an exact SHA instead of a branch tip.
  git -C "$serve" config uploadpack.allowAnySHA1InWant true
  ok "the update will land on $target_sha"

  step "pointing your install at it"
  # insteadOf is a TRANSPORT rewrite. Your checkout's origin keeps the official
  # URL, which matters: `hermes update` resolves its channel from the archive
  # and validates it against `git config --get remote.origin.url`. Repointing
  # origin at a fork would make the update fail before any git work.
  local redirect url
  redirect="$(file_url "$serve")"
  # The redirect belongs in the REPO-LOCAL config: it lives inside the checkout
  # this kit already backs up, so `post`'s wipe+restore removes it for free. A
  # writable GLOBAL git config is a dependency we do not need -- requiring it
  # aborts on any machine whose ~/.config/git/config is read-only or ACL-denied.
  for url in "$OFFICIAL_HTTPS" "$OFFICIAL_SSH"; do
    # --add: the key is multi-valued; a plain write would drop the first URL.
    git -C "$INSTALL_DIR" config --local --add "url.$redirect.insteadOf" "$url" \
      || die "could not write the URL redirect into $INSTALL_DIR/.git/config"
  done
  mkdir -p "$HERMES_HOME"
  touch "$HERMES_HOME/.skip_upstream_prompt"
  printf '%s\n' "$target_sha" > "$SNAP/target-sha"
  ok "official repo URL now resolves to the rehearsal copy"
  ok "created $HERMES_HOME/.skip_upstream_prompt (stops the 'add upstream remote?' prompt)"

  step "ready"
  say "your install is unchanged so far. nothing has been updated yet."
  say ""
  say "continue with the instructions provided!"
  say "your backup is at $SNAP. keep it until post has run."
}

# ---------------------------------------------------------------------------
# status (read-only)
# ---------------------------------------------------------------------------

cmd_status() {
  resolve_paths
  step "your install"
  say "HERMES_HOME   $HERMES_HOME"
  say "install       $INSTALL_DIR"
  say "desktop data  $USERDATA_DIR ($USERDATA_SOURCE)"
  say "backup root   $BACKUP_ROOT"
  step "backup"
  if ! SNAP="$(latest_backup_root)"; then
    say "none — nothing has been set up yet (run 'pre')"
    return 0
  fi
  say "latest        $SNAP"
  if [ -f "$SNAP/target-sha" ]; then
    say "prepared for  $(tr -d '\r' < "$SNAP/target-sha")"
    say "source        $(sed -n 's/.*"rehearsal_source": "\(.*\)",/\1/p' "$SNAP/manifest.json" 2>/dev/null | head -1) @ $(sed -n 's/.*"rehearsal_ref": "\(.*\)"/\1/p' "$SNAP/manifest.json" 2>/dev/null | head -1)"
  else
    say "prepared      no"
  fi
  say "marker        $([ -f "$HERMES_HOME/.skip_upstream_prompt" ] && echo present || echo absent)"
  local n=0
  while IFS= read -r _; do n=$((n + 1)); done \
    < <( { git config --global --get-regexp '^url\.' 2>/dev/null || true; git -C "$INSTALL_DIR" config --local --get-regexp '^url\.' 2>/dev/null || true; } )
  say "git rewrites  $n insteadOf entr(y/ies)"
  if [ -d "$INSTALL_DIR/.git" ]; then
    say "checkout now  $(git -C "$INSTALL_DIR" rev-parse --short HEAD) ($(git -C "$INSTALL_DIR" branch --show-current))"
  fi
}

# ---------------------------------------------------------------------------
# post: wipe both trees, then put the backup back
# ---------------------------------------------------------------------------

confirm() {
  [ "$ASSUME_YES" = 1 ] && return 0
  local prompt="$1 [y/N] " reply=""
  # Piped into a shell (`curl ... | bash -s -- post`) stdin is THIS SCRIPT, so a
  # plain `read` hits EOF and aborts every time. Ask the terminal instead.
  # /dev/tty is still a device node with no controlling terminal, so probe by
  # OPENING it -- `[ -r /dev/tty ]` passes there and the open then fails.
  local from_tty=""
  if from_tty="$( { printf '%s' "$prompt" >/dev/tty && read -r a </dev/tty && printf '%s' "$a"; } 2>/dev/null )"; then
    reply="$from_tty"
  else
    printf '%s' "$prompt"
    read -r reply || reply=""
  fi
  case "$reply" in y|Y|yes|YES) return 0 ;; *) die "aborted — nothing was changed (re-run with --yes to skip this prompt)" ;; esac
}

# An earlier version of this kit wrote the insteadOf redirect into the GLOBAL git
# config. Those entries name THIS snapshot's serve.git, which post leaves behind,
# so they would keep hijacking `hermes update` forever. Remove only the entries
# that point at our own rehearsal copy.
remove_stale_global_redirect() {
  local prefix url removed=0
  prefix="$(file_url "$SNAP/serve.git")"
  for url in "$OFFICIAL_HTTPS" "$OFFICIAL_SSH"; do
    if git config --global --get "url.$prefix.insteadOf" >/dev/null 2>&1; then
      git config --global --unset-all "url.$prefix.insteadOf" 2>/dev/null || true
      removed=$((removed + 1))
    fi
  done
  [ "$removed" = 0 ] || ok "removed $removed stale global URL redirect(s) from an older run of this kit"
}

cmd_post() {
  resolve_paths
  load_snapshot
  step "this will delete and restore:"
  say "  $HERMES_HOME  (all of it, including the checkout)"
  say "  $USERDATA_DIR"
  say "  from $SNAP"
  confirm "Put everything back from $SNAP?"

  step "stopping Hermes"
  pkill -f "Hermes.app/Contents/MacOS" 2>/dev/null || true
  pkill -f "hermes gateway" 2>/dev/null || true
  ok "asked Hermes to stop (if anything was running)"

  step "clearing both trees"
  rm -rf "$HERMES_HOME";  ok "removed $HERMES_HOME"
  rm -rf "$USERDATA_DIR"; ok "removed $USERDATA_DIR"

  step "restoring your HERMES_HOME"
  mkdir -p "$HERMES_HOME"
  tar -xf "$SNAP/hermes-home.tar" -C "$HERMES_HOME" || die "restore failed — your backup is intact at $SNAP"
  ok "restored"

  step "restoring the desktop app's data"
  if [ -f "$SNAP/electron-userdata.tar" ]; then
    mkdir -p "$USERDATA_DIR"
    tar -xf "$SNAP/electron-userdata.tar" -C "$USERDATA_DIR" || die "userData restore failed (backup intact at $SNAP)"
    ok "restored"
  else
    warn "there was no desktop app data to restore"
  fi

  remove_stale_global_redirect

  step "done"
  say "Your HERMES_HOME and the desktop app's data are back exactly as they were."
  say "Open the desktop app once and run 'hermes doctor' to confirm."
  say "Nothing was judged or changed by this script; the backup at $SNAP is"
  say "yours to keep or delete."
}

case "$SUBCMD" in
  pre)    cmd_pre ;;
  post)   cmd_post ;;
  status) cmd_status ;;
esac
