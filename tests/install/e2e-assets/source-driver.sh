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