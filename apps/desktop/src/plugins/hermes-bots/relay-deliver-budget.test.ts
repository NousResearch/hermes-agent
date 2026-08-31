import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

// #93911 review follow-up: the Desktop deadline for bot_relay.deliver mirrors
// three backend numbers. Nothing in the type system links a TS constant to a
// Python default, so this file is the seam: it reads the backend sources and
// fails when a mirror drifts or the settlement margin stops being positive.
// Without it, raising the backend turn timeout would silently reintroduce
// #93911 — the client giving up before a valid typed settlement arrives.

const relaySource = readFileSync(join(process.cwd(), 'src/plugins/hermes-bots/relay.ts'), 'utf8')
const repoRoot = join(process.cwd(), '..', '..')
const configDefaults = readFileSync(join(repoRoot, 'hermes_cli/config_defaults.py'), 'utf8')
const relayHandler = readFileSync(join(repoRoot, 'tui_gateway/methods_bot_relay.py'), 'utf8')

function tsConstant(name: string): number {
  const match = relaySource.match(new RegExp(`const ${name} = ([0-9_]+)`))
  expect(match, `${name} must stay a literal so this test can read it`).toBeTruthy()

  return Number(match![1].replaceAll('_', ''))
}

describe('bot_relay.deliver budget mirrors', () => {
  it('mirrors the backend turn-lock default', () => {
    const lockWaitMatch = configDefaults.match(/"turn_wait_seconds":\s*(\d+)/)

    expect(lockWaitMatch, 'bot_mode.turn_wait_seconds default must exist in config_defaults.py').toBeTruthy()
    expect(tsConstant('RELAY_TURN_LOCK_WAIT_MS')).toBe(Number(lockWaitMatch![1]) * 1000)
  })

  it('mirrors the backend per-attempt turn timeout', () => {
    const attemptTimeouts = [...relayHandler.matchAll(/timeout=(\d+)/g)].map(m => Number(m[1]))

    expect(attemptTimeouts.length, 'expected the attempt and its policy-gated retry').toBeGreaterThanOrEqual(2)
    // Every attempt shares one bound; if they ever diverge, the mirror below is
    // no longer a faithful ceiling and this must be revisited deliberately.
    expect(new Set(attemptTimeouts).size, `attempt timeouts diverged: ${attemptTimeouts}`).toBe(1)
    expect(tsConstant('RELAY_TURN_ATTEMPT_MS')).toBe(attemptTimeouts[0] * 1000)
    expect(tsConstant('RELAY_TURN_MAX_ATTEMPTS')).toBe(attemptTimeouts.length)
  })

  it('keeps the client deadline strictly greater than the backend ceiling', () => {
    const margin = tsConstant('RELAY_DELIVER_SETTLEMENT_MARGIN_MS')

    // Strictly greater, not equal: a backend that answers at its own limit
    // still has to serialize and transport that answer.
    expect(margin, 'settlement margin must be positive').toBeGreaterThan(0)

    // The call site must pass the composed budget, not a bare literal.
    const drain = relaySource.slice(
      relaySource.indexOf('async function drainRelayOutboxes'),
      relaySource.indexOf('export function startBotRelay')
    )

    expect(drain).toMatch(/'bot_relay\.deliver'[\s\S]{0,400}RELAY_DELIVER_TIMEOUT_MS/)
    expect(drain).not.toMatch(/'bot_relay\.deliver'[\s\S]{0,400}\d{6,}/)
  })
})
