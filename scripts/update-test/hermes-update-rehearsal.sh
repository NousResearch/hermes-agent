#!/usr/bin/env bash
# hermes-update-rehearsal.sh — for an EXISTING Hermes install.
#
# Two steps:
#   pre   take a `hermes backup` of your data, clone HERMES_HOME and the desktop
#         app's Electron userData into the backup dir, then point the install's
#         update source at a fork so `hermes update` pulls that fork's main.
#         Prints what to do next.
#   post  swap the clones back in, so both trees are exactly as they were.
#
# Plus `status`, which only prints. This script never judges your install: it
# reports what it did and stops. Whether the update worked is for you to see.
#
# On APFS (macOS) and btrfs/xfs the clone is copy-on-write: no file data is
# copied. Elsewhere it is a plain copy. post swaps by rename, so it is instant
# when the backup dir is on the same disk. HERMES_HOME/cache is left as it is.
#
#   ./hermes-update-rehearsal.sh pre  --source <git-url>
#   # ... run `hermes update`, use Hermes, test whatever you need ...
#   ./hermes-update-rehearsal.sh post
#
# Options:
#   --source URL   repo to pull the update from; updates follow its main
#                  (default: the rehearsal fork)
#   --backup-root DIR   where the backup lives (default ~/hermes-update-rehearsal)
#   --yes          post: skip the confirmation
#
# Requires: git. `pre` and `hermes update` need network access to --source.

set -euo pipefail

OFFICIAL_HTTPS="https://github.com/NousResearch/hermes-agent.git"
OFFICIAL_SSH="git@github.com:NousResearch/hermes-agent.git"
DEFAULT_SOURCE="https://github.com/ethernet8023/hermes-agent.git"

SUBCMD=""
BACKUP_ROOT="${HOME}/hermes-update-rehearsal"
SOURCE="$DEFAULT_SOURCE"
ASSUME_YES=0

say()  { printf '%s\n' "$*"; }
ok()   { printf '  OK %s\n' "$*"; }
warn() { printf '  WARN %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
step() { printf '\n=== %s ===\n' "$*"; }

usage() {
  if [ -n "$SELF" ] && [ -r "$SELF" ]; then sed -n '2,28p' "$SELF"; exit 0; fi
  cat <<'EOF'
hermes-update-rehearsal.sh -- run against an EXISTING Hermes install.

  pre     back up your data, clone both trees, point the update source at a fork
  post    swap the clones back in, exactly as they were
  status  print what is prepared (read-only; nothing is touched)

Options:
  --source URL        repo to pull the update from; updates follow its main
                      (default: the rehearsal fork)
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
    --backup-root) [ "$#" -ge 2 ] || die "--backup-root needs a value"; BACKUP_ROOT="$2"; shift 2 ;;
    --yes|-y)      ASSUME_YES=1; shift ;;
    -h|--help)     usage ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -n "$SUBCMD" ] || usage

command -v git >/dev/null 2>&1 || die "git is required"

SNAP=""

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

# Under git-bash/cygwin, POSIX paths are not translated for native tools
# (git.exe), so normalise them. On macOS cygpath is absent: no-op.
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
  [ -d "$SNAP/home" ] || die "$SNAP has no clone of HERMES_HOME (made by an older version of this script, or already restored)"
  # Plain text, not JSON: this string is compared byte-for-byte to decide whether
  # the backup belongs to the home post is about to restore.
  local recorded current
  recorded="$(tr -d '\r' < "$recorded_path")"
  current="$(printf '%s' "$HERMES_HOME" | tr '\\' '/')"
  [ "$(printf '%s' "$recorded" | tr '\\' '/')" = "$current" ] \
    || die "that backup belongs to HERMES_HOME=$recorded, not $HERMES_HOME; pass --backup-root to pick the right one"
}

# ---------------------------------------------------------------------------
# cloning
# ---------------------------------------------------------------------------

# Pick how to clone into $SNAP: copy-on-write where the filesystem can, a plain
# copy elsewhere. Probed on a real file, since support is per filesystem.
CLONE_MODE=""
detect_clone_mode() {
  local probe="$SNAP/.clone-probe"
  printf 'x' > "$probe"
  # GNU cp also accepts -c (a deprecated --preserve=context), so only ask
  # macOS's cp for clonefile.
  if [ "$(uname -s)" = Darwin ] && cp -c "$probe" "$probe.c" 2>/dev/null; then
    CLONE_MODE="clonefile"          # macOS APFS
  elif cp --reflink=always "$probe" "$probe.c" 2>/dev/null; then
    CLONE_MODE="reflink"            # btrfs, xfs, bcachefs
  elif cp --reflink=auto "$probe" "$probe.c" 2>/dev/null; then
    CLONE_MODE="copy-gnu"           # GNU cp, no CoW here (ext4, ...)
  else
    CLONE_MODE="copy"               # BSD cp without clonefile (non-APFS mac)
  fi
  rm -f "$probe" "$probe.c"
}

clone_entry() {
  case "$CLONE_MODE" in
    clonefile)        clonefile_entry "$1" "$2" || {
                        warn "fast clone of $(basename "$1") failed; cloning it file by file"
                        rm -rf "${2:?}/$(basename "$1")"; cp -cpR "$1" "$2/"; } ;;
    reflink|copy-gnu) cp -a --reflink=auto "$1" "$2/" ;;
    *)                cp -pR "$1" "$2/" ;;
  esac
}

# APFS can clone a whole directory tree in ONE clonefileat(2) call -- ~9x faster
# than cp -c, which clones file by file (2.3s vs 19.3s for 130k entries). No
# compiler on a stock Mac, but /usr/bin/perl can make the raw syscall. The
# kernel stamps cloned DIRECTORIES with the current time, so a second pass puts
# every directory's atime/mtime back; file times come through the clone as-is.
clonefile_entry() {
  /usr/bin/perl -MFile::Find -e '
    my ($src, $dst) = @ARGV;
    # SYS_clonefileat = 462 (xnu syscalls.master), AT_FDCWD = -2,
    # CLONE_NOFOLLOW = 1: a top-level symlink is cloned as a link.
    syscall(462, -2, $src, -2, $dst, 1) == 0 or exit 1;
    find({ no_chdir => 1, wanted => sub {
      lstat($_); return unless -d _;
      my @st = lstat(_);
      utime($st[8], $st[9], $dst . substr($File::Find::name, length $src)) or exit 1;
    } }, $src);
  ' "$1" "$2/$(basename "$1")" 2>/dev/null
}

# Clone every top-level entry of $1 into $2, except one named $3 (may be empty).
clone_tree() {
  local src="$1" dst="$2" skip="$3" e
  mkdir -p "$dst"
  while IFS= read -r -d '' e; do
    [ -n "$skip" ] && [ "$(basename "$e")" = "$skip" ] && continue
    clone_entry "$e" "$dst" || return 1
  done < <(find "$src" -mindepth 1 -maxdepth 1 -print0)
}

# post: move $live's top-level entries (except $skip) into $aside, then the
# clone's entries into $live. Renames only, so the live directory itself never
# moves -- it may be a mountpoint or a symlink target.
swap_tree() {
  local live="$1" clone="$2" aside="$3" skip="$4" e
  mkdir -p "$aside" "$live"
  while IFS= read -r -d '' e; do
    [ -n "$skip" ] && [ "$(basename "$e")" = "$skip" ] && continue
    mv "$e" "$aside/" || return 1
  done < <(find "$live" -mindepth 1 -maxdepth 1 -print0)
  while IFS= read -r -d '' e; do
    mv "$e" "$live/" || return 1
  done < <(find "$clone" -mindepth 1 -maxdepth 1 -print0)
  rmdir "$clone"
}

# The install's own hermes: pre-branch installs keep it in the venv.
resolve_hermes_exe() {
  local c
  for c in "$INSTALL_DIR/venv/bin/hermes" "$INSTALL_DIR/.hermes/bin/hermes" \
           "$INSTALL_DIR/venv/Scripts/hermes.exe" "$HERMES_HOME/bin/hermes"; do
    [ -x "$c" ] && { printf '%s' "$c"; return 0; }
  done
  return 1
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
  local hermes_exe
  hermes_exe="$(resolve_hermes_exe)" || die "no hermes executable found in $INSTALL_DIR (venv/bin, .hermes/bin) or $HERMES_HOME/bin"

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

  # Before anything is written: a bad --source or no network should abort with
  # nothing done, not after the backup and clone.
  local target_sha
  target_sha="$(git ls-remote "$SOURCE" refs/heads/main 2>/dev/null | cut -f1)"
  [ -n "$target_sha" ] || die "could not read main from $SOURCE (network? permissions? bad --source?) — nothing was done"
  ok "$SOURCE main is at $target_sha"

  SNAP="$BACKUP_ROOT/$(date -u +%Y%m%dT%H%M%SZ)"
  [ ! -e "$SNAP" ] || die "backup dir already exists: $SNAP"
  mkdir -p "$SNAP"

  step "backing up your data (hermes backup)"
  # Config, keys, sessions, memories, skills, with SQLite copied consistently:
  # a second, independent copy of what matters most. post never deletes it.
  local started code=0 out
  started="$SECONDS"
  out="$(HERMES_HOME="$HERMES_HOME" "$hermes_exe" backup -o "$SNAP/hermes-backup.zip" 2>&1)" || code=$?
  if [ "$code" = 1 ] && [ -f "$SNAP/hermes-backup.zip" ]; then
    printf '    %s\n' "$out"
    warn "hermes backup finished INCOMPLETE ($((SECONDS - started))s): the files listed above are not in $SNAP/hermes-backup.zip"
  elif [ "$code" != 0 ] || [ ! -f "$SNAP/hermes-backup.zip" ]; then
    printf '    %s\n' "$out"
    die "hermes backup failed (exit $code) — nothing else was done"
  else
    ok "hermes-backup.zip ($(du -h "$SNAP/hermes-backup.zip" | cut -f1), $((SECONDS - started))s)"
  fi

  step "cloning HERMES_HOME and the desktop app's data"
  detect_clone_mode
  case "$CLONE_MODE" in
    clonefile|reflink) ok "copy-on-write clones ($CLONE_MODE): no file data is copied" ;;
    *) ok "this filesystem cannot clone, so this is a plain copy" ;;
  esac
  started="$SECONDS"
  # Everything but cache/: checkout, venv, PM store and node_modules come too,
  # so post is a true rollback rather than a re-download. SQLite sidecars travel
  # WITH their db on purpose (a raw copy of db+wal+shm is consistent).
  clone_tree "$HERMES_HOME" "$SNAP/home" cache || die "cloning HERMES_HOME failed"
  ok "HERMES_HOME -> $SNAP/home ($((SECONDS - started))s, cache/ left out)"
  local userdata_existed=false
  if [ -d "$USERDATA_DIR" ]; then
    userdata_existed=true
    clone_tree "$USERDATA_DIR" "$SNAP/userdata" "" || die "cloning the desktop app's data failed"
    ok "desktop data -> $SNAP/userdata"
  else
    warn "no Electron userData at $USERDATA_DIR (desktop app not installed?)"
  fi

  step "recording what this backup is"
  printf '%s\n' "$HERMES_HOME" > "$SNAP/hermes-home.txt"
  cat > "$SNAP/manifest.json" <<EOF
{
  "schema": 4,
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hermes_home": "$HERMES_HOME",
  "install_dir": "$INSTALL_DIR",
  "userdata_dir": "$USERDATA_DIR",
  "userdata_dir_source": "$USERDATA_SOURCE",
  "userdata_existed": $userdata_existed,
  "clone_mode": "$CLONE_MODE",
  "rehearsal_source": "$SOURCE"
}
EOF
  ok "manifest.json (backup of $HERMES_HOME)"

  step "pointing your install at $SOURCE"
  # insteadOf is a TRANSPORT rewrite. Your checkout's origin keeps the official
  # URL, which matters: `hermes update` resolves its channel from the archive
  # and validates it against `git config --get remote.origin.url`. Repointing
  # origin at a fork would make the update fail before any git work.
  # Straight at the fork: the updater follows main, so the fork's main is what
  # lands (force-push it to the branch under test).
  local url
  # The redirect belongs in the REPO-LOCAL config: it lives inside the checkout
  # this kit already cloned, so `post`'s swap removes it for free. A writable
  # GLOBAL git config is a dependency we do not need -- requiring it aborts on
  # any machine whose ~/.config/git/config is read-only or ACL-denied.
  for url in "$OFFICIAL_HTTPS" "$OFFICIAL_SSH"; do
    # --add: the key is multi-valued; a plain write would drop the first URL.
    git -C "$INSTALL_DIR" config --local --add "url.$SOURCE.insteadOf" "$url" \
      || die "could not write the URL redirect into $INSTALL_DIR/.git/config"
  done
  touch "$HERMES_HOME/.skip_upstream_prompt"
  printf '%s\n' "$target_sha" > "$SNAP/target-sha"
  ok "official repo URL now resolves to $SOURCE"
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
    say "prepared for  $(tr -d '\r' < "$SNAP/target-sha") (main at pre time)"
    say "source        $(sed -n 's/.*"rehearsal_source": "\(.*\)"/\1/p' "$SNAP/manifest.json" 2>/dev/null | head -1)"
  else
    say "prepared      no"
  fi
  say "clone         $([ -d "$SNAP/home" ] && echo "present ($(sed -n 's/.*"clone_mode": "\(.*\)",/\1/p' "$SNAP/manifest.json" 2>/dev/null | head -1))" || echo 'none (restored already?)')"
  say "data backup   $([ -f "$SNAP/hermes-backup.zip" ] && echo hermes-backup.zip || echo none)"
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
# post: swap the clones back in
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

cmd_post() {
  resolve_paths
  load_snapshot
  local userdata_existed
  userdata_existed="$(sed -n 's/.*"userdata_existed": \([a-z]*\).*/\1/p' "$SNAP/manifest.json" | head -1)"
  step "this will restore:"
  say "  $HERMES_HOME  (everything except cache/, including the checkout)"
  say "  $USERDATA_DIR"
  say "  to how they were at $SNAP"
  confirm "Put everything back from $SNAP?"

  step "stopping this home's gateway"
  local hermes_exe
  hermes_exe="$(resolve_hermes_exe)" || die "no Hermes launcher for $INSTALL_DIR; stop this home's gateway before restoring"
  HERMES_HOME="$HERMES_HOME" "$hermes_exe" gateway stop \
    || die "could not stop this home's gateway; restore has not started"
  ok "stopped this home's gateway (close the desktop app before restoring its data)"

  local aside="$SNAP/replaced"
  [ ! -e "$aside" ] || die "$aside already exists (an interrupted post?) — nothing was changed; move it away and run post again"

  step "restoring your HERMES_HOME"
  swap_tree "$HERMES_HOME" "$SNAP/home" "$aside/home" cache \
    || die "restore stopped part-way: the post-update files are in $aside/home, the rest of the clone in $SNAP/home"
  ok "restored (cache/ left as it was)"

  step "restoring the desktop app's data"
  if [ "$userdata_existed" = true ]; then
    swap_tree "$USERDATA_DIR" "$SNAP/userdata" "$aside/userdata" "" \
      || die "restore stopped part-way: the post-update files are in $aside/userdata, the rest of the clone in $SNAP/userdata"
    ok "restored"
  elif [ -e "$USERDATA_DIR" ]; then
    # There was no desktop data at the snapshot: exact means none now either.
    mkdir -p "$aside"
    mv "$USERDATA_DIR" "$aside/userdata"
    ok "removed $USERDATA_DIR (it did not exist before)"
  else
    ok "nothing to restore (there was no desktop data before)"
  fi

  step "cleaning up"
  # Only the whole trees the update left behind. hermes-backup.zip stays.
  rm -rf "$aside"
  ok "deleted the post-update trees ($aside)"

  step "done"
  say "Your HERMES_HOME and the desktop app's data are back exactly as they were."
  say "Open the desktop app once and run 'hermes doctor' to confirm."
  say "Nothing was judged or changed by this script; the backup at $SNAP"
  say "(including hermes-backup.zip) is yours to keep or delete."
}

case "$SUBCMD" in
  pre)    cmd_pre ;;
  post)   cmd_post ;;
  status) cmd_status ;;
esac
