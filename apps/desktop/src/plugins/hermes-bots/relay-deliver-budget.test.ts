import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  RELAY_DELIVER_BACKEND_CEILING_MS,
  RELAY_DELIVER_SETTLEMENT_MARGIN_MS,
  RELAY_DELIVER_TIMEOUT_MS,
  RELAY_TURN_ATTEMPT_MS,
  RELAY_TURN_LOCK_WAIT_MS,
  RELAY_TURN_MAX_ATTEMPTS
} from './relay-budget'

// #93911 review follow-up: the Desktop deadline for bot_relay.deliver mirrors
// three backend numbers. Nothing in the type system links a TS constant to a
// Python default, so this file is the seam: it reads the backend defaults and
// fails when a mirror drifts or the settlement margin stops being positive.
// Without it, raising the backend turn timeout would silently reintroduce
// #93911 — the client giving up before a valid typed settlement arrives.

const repoRoot = join(process.cwd(), '..', '..')
const configDefaults = readFileSync(join(repoRoot, 'hermes_cli/config_defaults.py'), 'utf8')
const relayPlumbing = readFileSync(join(repoRoot, 'tools/bot_relay.py'), 'utf8')

function pyConstant(name: string): number {
  const match = relayPlumbing.match(new RegExp(`^${name}\\s*=\\s*(\\d+)`, 'm'))
  expect(match, `${name} must exist as a literal in tools/bot_relay.py`).toBeTruthy()

  return Number(match![1])
}

describe('bot_relay.deliver budget mirrors', () => {
  it('mirrors the backend turn-lock default', () => {
    const lockWaitMatch = configDefaults.match(/"turn_wait_seconds":\s*(\d+)/)

    expect(lockWaitMatch, 'bot_mode.turn_wait_seconds default must exist in config_defaults.py').toBeTruthy()
    expect(RELAY_TURN_LOCK_WAIT_MS).toBe(Number(lockWaitMatch![1]) * 1000)
  })

  it('mirrors the backend per-attempt turn timeout', () => {
    // The backend names both numbers explicitly (tools/bot_relay.py) so the mirror is a
    // constant-to-constant check, not a count of textual subprocess.run(...) call sites.
    expect(RELAY_TURN_ATTEMPT_MS).toBe(pyConstant('TURN_ATTEMPT_TIMEOUT_SECONDS') * 1000)
    expect(RELAY_TURN_MAX_ATTEMPTS).toBe(pyConstant('TURN_MAX_ATTEMPTS'))
  })

  it('shares its settlement margin with the sender-side waiter budget', () => {
    // tests/tools/test_bot_relay.py checks that REPLY_WAIT_SECONDS exceeds the rebuilt sum.
    expect(RELAY_DELIVER_SETTLEMENT_MARGIN_MS).toBe(pyConstant('DESKTOP_DELIVER_SETTLEMENT_MARGIN_SECONDS') * 1000)
  })

  it('keeps the client deadline strictly greater than the backend ceiling', () => {
    // Strictly greater, not equal: a backend that answers at its own limit
    // still has to serialize and transport that answer.
    expect(RELAY_DELIVER_SETTLEMENT_MARGIN_MS, 'settlement margin must be positive').toBeGreaterThan(0)
    expect(RELAY_DELIVER_TIMEOUT_MS).toBeGreaterThan(RELAY_DELIVER_BACKEND_CEILING_MS)

    // The ceiling is the backend's own worst case: lock wait + every attempt.
    const lockWaitMs = Number(configDefaults.match(/"turn_wait_seconds":\s*(\d+)/)![1]) * 1000
    const attemptsMs = pyConstant('TURN_ATTEMPT_TIMEOUT_SECONDS') * 1000 * pyConstant('TURN_MAX_ATTEMPTS')

    expect(RELAY_DELIVER_BACKEND_CEILING_MS).toBe(lockWaitMs + attemptsMs)
  })
})
