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
