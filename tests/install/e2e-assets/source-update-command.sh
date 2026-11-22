#!/usr/bin/env bash
# Build the installed CLI's invocation for a source-to-source update.
# Older releases do not accept either flag, so probe their own help first.
build_source_update_command() {
  local hermes="$1" help="$2"
  update_cmd=("$hermes" update)
  if grep -qF -- --yes <<< "$help"; then
    update_cmd+=(--yes)
  fi
  # This fixture advances serve.git/main to an unpublished HEAD (or NEXT).
  # Newer updaters otherwise consult releases/channels/main.json, which may
  # not exist. Explicit --branch main follows the fixture's git transport.
  if grep -Eq -- '(^|[[:space:]]|\[)--branch([=[:space:]]|$)' <<< "$help"; then
    update_cmd+=(--branch main)
  fi
}

# Official source checkouts default to the stable channel (the latest published
# vX.Y.Z), and these fixtures publish commits on main, not releases: a fresh
# install at HEAD is newer than the latest release and correctly waits, so the
# HEAD -> NEXT legs would find no update. Record `main` for the install BEFORE
# the user-state snapshot (config.yaml is part of the baseline). Releases that
# predate --set-channel follow main already.
pin_source_main_channel() {
  local hermes="$1" help
  help="$("$hermes" update --help 2>&1)" || { printf 'update --help failed: %s\n' "$help" >&2; return 1; }
  grep -qF -- --set-channel <<< "$help" || return 0
  "$hermes" update --set-channel main < /dev/null
}
