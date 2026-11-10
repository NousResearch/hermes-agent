#!/usr/bin/env bash
# Source-only helpers: never search PATH for a different installation.
source_hermes() {
  local root="$1" command="$1/.hermes/bin/hermes"
  if [ -e "$command" ] || [ -L "$command" ]; then
    [ -f "$command" ] && [ -x "$command" ] || {
      printf 'invalid published launcher: %s\n' "$command" >&2; return 1;
    }
  else
    # A PM source tree promises publication during install/update. Falling
    # back here would let --version complete an unfinished update for it.
    [ ! -f "$root/pm/lock.json" ] || {
      printf 'missing published launcher: %s\n' "$command" >&2; return 1;
    }
    command="$root/venv/bin/hermes"
    [ -f "$command" ] && [ -x "$command" ] || {
      printf 'no installed Hermes command under %s\n' "$root" >&2; return 1;
    }
  fi
  printf '%s\n' "$command"
}

# v2026.6.19's install.sh (also run by its DMG bootstrap) writes .install_method
# into the checkout without ignoring it, and that release's Desktop checker
# reports git status --porcelain verbatim. Teach only this disposable clone that
# the installer's own marker is not a source edit; keep the marker for
# install-method detection and refuse any other dirty state.
accept_installer_marker() {
  local root="$1" status
  status="$(git -C "$root" status --porcelain --untracked-files=all)" || return 1
  if [ "$status" = '?? .install_method' ] && [ "$(cat "$root/.install_method")" = git ]; then
    printf '\n/.install_method\n' >> "$(git -C "$root" rev-parse --absolute-git-dir)/info/exclude"
    status="$(git -C "$root" status --porcelain --untracked-files=all)" || return 1
  fi
  [ -z "$status" ] || { printf 'installed source has changes other than the installer marker:\n%s\n' "$status" >&2; return 1; }
}

# Hand out a command to DRIVE the next ordinary startup, even when the
# published launcher is not there yet.
#
# A pre-handoff release cannot flip during `hermes update` -- there is no
# retired-hook seam on its update path to reach, so the update ends with the
# tree at HEAD and no `.hermes/bin/*`. The NEXT ordinary startup is what
# completes it: hermes_bootstrap calls prepare_launch() before importing
# anything, which syncs PM and publishes the launchers.
#
# Deliberately NOT used for `--version` probes: those stay under
# HERMES_DISABLE_LAZY_INSTALLS so a probe can never complete an unfinished
# update. Only a real startup may heal.
source_hermes_for_startup() {
  local root="$1" command="$1/.hermes/bin/hermes"
  if [ -f "$command" ] && [ -x "$command" ]; then
    printf '%s\n' "$command"; return 0
  fi
  command="$root/venv/bin/hermes"
  [ -f "$command" ] && [ -x "$command" ] || {
    printf 'no installed Hermes command under %s\n' "$root" >&2; return 1
  }
  printf '%s\n' "$command"
}