/**
 * The gates' signatures are part of the suite's contract: each must match the
 * exact failing assertion its bug produces and nothing else, or a gate turns
 * an unrelated regression into a silent expected-failure.
 */

import { expect, test } from '@playwright/test'

import { gateMatches, KNOWN } from './gates'

test('gate signatures match their bug and not neighbouring failures', () => {
  const real =
    '"Update now" hands off to the updater and the app quits for it; the app logged:\n' +
    '2026-09-27 01:19:44,583 [hermes] [updates] state.db pre-flight failed: Python not found. Update cancelled before backend shutdown.'

  expect(gateMatches(KNOWN.preflightPython, real)).toBe(true)

  for (const decoy of [
    '[updates] refusing posix hand-off: An update is already running (PID 670, started 2s ago).',
    '[updates] state.db pre-flight failed: database disk image is malformed. Update cancelled before backend shutdown.',
    '[updates] detached update FAILED',
    'Timed out 120000ms waiting for the app to quit for the update hand-off',
    'Python not found'
  ]) {
    expect(gateMatches(KNOWN.preflightPython, decoy), decoy).toBe(false)
  }
})
