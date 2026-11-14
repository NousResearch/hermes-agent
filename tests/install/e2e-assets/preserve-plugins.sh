#!/usr/bin/env bash
# Shared plugin upgrade-preservation hooks for the install E2E drivers
# (POSIX + macOS). Sourced by tests/install/installer-script-e2e.sh and
# tests/install/macos-desktop-e2e.sh.
#
# The contract under test: a tagged Hermes upgrade must NOT delete or modify
# anything in the active home's plugins/** tree or any profile's plugins/**
# tree — including directory wrapper markers (mnemosyne-wrapper.json),
# symlinked runtimes, and the externally-owned sidecar witness file that
# lives OUTSIDE the home. Destructive flows (explicit uninstall, plugin
# removal, profile/user deletion) are out of contract and not exercised.
#
# The fixtures are deliberately non-dependency: directory wrappers with no
# pyproject anywhere in the scanned root, so the plugin scanner never
# recurses into a dependency graph and nothing is downloaded.
#
# State flow (called by the driver):
#   preserve_before_upgrade  seed fixtures + snapshot  ->  $PRESERVE_SNAPSHOT
#   preserve_after_upgrade   verify against snapshot    (exit 1 on violation)
#
# Requires: HERMES_HOME set (the leg's isolated home), WORK_ROOT, LOG_DIR,
# REPO_ROOT. Verifier: tests/install/e2e-assets/verify-plugin-preservation.py
# (stdlib-only; runs under python3 or $HERMES_E2E_PYTHON).

PRESERVE_SNAPSHOT="$WORK_ROOT/plugin-preservation-snapshot.json"
PRESERVE_REPORT="$LOG_DIR/plugin-preservation-report.json"
PRESERVE_EXTERNAL="$WORK_ROOT/external-mnemosyne-runtime"

_preserve_python() {
  if [ -n "${HERMES_E2E_PYTHON:-}" ]; then
    printf '%s' "$HERMES_E2E_PYTHON"
  else
    printf 'python3'
  fi
}

seed_plugin_preservation_fixtures() {
  "$(_preserve_python)" "$REPO_ROOT/tests/install/e2e-assets/verify-plugin-preservation.py" \
    seed --home "$1" --external "$2" || fail "could not seed fresh preservation fixtures"
}

preserve_before_upgrade() {
  step "plugin preservation: seeding fixtures and snapshotting pre-upgrade state"
  [ ! -e "$PRESERVE_SNAPSHOT" ] || fail "refusing to overwrite an existing preservation snapshot"
  seed_plugin_preservation_fixtures "$HERMES_HOME" "$PRESERVE_EXTERNAL"
  local py; py="$(_preserve_python)"
  "$py" "$REPO_ROOT/tests/install/e2e-assets/verify-plugin-preservation.py" \
    snapshot --home "$HERMES_HOME" --out "$PRESERVE_SNAPSHOT" 2>&1 | ts_prefix \
    || fail "plugin preservation snapshot failed"

  ok "pre-upgrade plugin snapshot at $PRESERVE_SNAPSHOT"
}

preserve_after_upgrade() {
  step "plugin preservation: verifying post-upgrade state"
  [ -f "$PRESERVE_SNAPSHOT" ] \
    || fail "no pre-upgrade plugin snapshot at $PRESERVE_SNAPSHOT; cannot verify preservation"
  local py; py="$(_preserve_python)"
  local rc=0
  "$py" "$REPO_ROOT/tests/install/e2e-assets/verify-plugin-preservation.py" \
    verify --home "$HERMES_HOME" --snapshot "$PRESERVE_SNAPSHOT" \
    --report "$PRESERVE_REPORT" > "$LOG_DIR/plugin-preservation-verify.log" 2>&1 || rc=$?
  log_group "plugin preservation verify transcript" "$LOG_DIR/plugin-preservation-verify.log"
  [ "$rc" -eq 0 ] \
    || fail "plugin preservation violated by the upgrade: deleted/modified entries above; report at $PRESERVE_REPORT"
  ok "plugins/** and profile plugin trees (markers, symlinked runtimes, external sidecar witness) survived the upgrade intact"
}
