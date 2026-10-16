#!/usr/bin/env bash
# Shared user-state preservation hooks for the install E2E drivers
# (POSIX + macOS + Windows-via-bash). Sourced by installer-script-e2e.sh and
# scripts under tests/install/.
#
# The contract under test: a tagged upgrade must not delete or modify the
# user's OWN durable state — config.yaml (additive rewrites only), .env,
# auth.json, state.db (row counts may not shrink), gateway_state.json,
# memories/, cron/, sessions/, profiles/, photon/, desktop-plugins/,
# tui-widgets/, skins/, pets/, and skills/.archive/. Additions are fine; an
# upgrade may add files, it may not take yours away or alter them.
#
# Everything this verifies is produced by the PRODUCT through the ordinary user
# path (see user-state-actions.sh) — never seeded by the harness. Asserting over
# fixtures we wrote ourselves would only prove the harness can write files.
#
# Not owned here, by design:
#   plugins/**            -> preserve-plugins.sh / verify-plugin-preservation.py
#   config migration      -> config.yaml and state.db are reported, not judged
#                            (their real contract is the row-count check)
#   skills/ bulk          -> re-synced from the bundled library by the product
#
# State flow (called by the driver):
#   user_state_before_upgrade   snapshot   ->  $USER_STATE_SNAPSHOT
#   user_state_after_upgrade    verify      (exit 1 on violation)
#
# Requires: HERMES_HOME set (the leg's isolated home), WORK_ROOT, LOG_DIR,
# REPO_ROOT. Verifier: tests/install/e2e-assets/verify-user-state.py
# (stdlib-only; runs under python3 or $HERMES_E2E_PYTHON).

USER_STATE_SNAPSHOT="$WORK_ROOT/user-state-snapshot.json"
USER_STATE_REPORT="$LOG_DIR/user-state-report.json"
USER_STATE_VERIFIER="$REPO_ROOT/tests/install/e2e-assets/verify-user-state.py"

_user_state_python() {
  if [ -n "${HERMES_E2E_PYTHON:-}" ]; then
    printf '%s' "$HERMES_E2E_PYTHON"
  else
    printf 'python3'
  fi
}

user_state_before_upgrade() {
  step "user state: snapshotting what the user already has"
  [ -f "$USER_STATE_VERIFIER" ] || fail "user-state verifier missing at $USER_STATE_VERIFIER"
  [ ! -e "$USER_STATE_SNAPSHOT" ] || fail "refusing to overwrite an existing user-state snapshot"
  local py rc=0
  py="$(_user_state_python)"
  "$py" "$USER_STATE_VERIFIER" snapshot --home "$HERMES_HOME" --out "$USER_STATE_SNAPSHOT" \
    2>&1 | ts_prefix > "$LOG_DIR/user-state-snapshot.log" || rc=$?
  log_group "user-state snapshot transcript" "$LOG_DIR/user-state-snapshot.log"
  # 3 = inconclusive (nothing to protect). That is a broken leg, not a pass:
  # an empty contract would make the post-upgrade verify meaningless.
  [ "$rc" -eq 0 ] || fail "user-state snapshot failed (exit $rc); see $LOG_DIR/user-state-snapshot.log"
  ok "user-state snapshot at $USER_STATE_SNAPSHOT"
}

user_state_after_upgrade() {
  step "user state: verifying the upgrade did not take anything away"
  [ -f "$USER_STATE_SNAPSHOT" ] \
    || fail "no pre-upgrade user-state snapshot; cannot claim preservation"
  local py rc=0
  py="$(_user_state_python)"
  "$py" "$USER_STATE_VERIFIER" verify --home "$HERMES_HOME" \
    --snapshot "$USER_STATE_SNAPSHOT" --report "$USER_STATE_REPORT" \
    > "$LOG_DIR/user-state-verify.log" 2>&1 || rc=$?
  log_group "user-state verify transcript" "$LOG_DIR/user-state-verify.log"
  [ "$rc" -eq 0 ] \
    || fail "the upgrade changed the user's own state; report at $USER_STATE_REPORT"
  ok "user state survived: $(grep -o 'user-state preservation OK.*' "$LOG_DIR/user-state-verify.log" | head -1)"
}
