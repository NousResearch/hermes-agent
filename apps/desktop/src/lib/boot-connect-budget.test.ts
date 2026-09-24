import { describe, expect, it } from 'vitest'

import {
  DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
  resolvePortAnnounceTimeoutMs,
  resolveRendererBootWaitMs
} from '../../../shared/src/desktop-boot-budget'

import { BACKEND_BOOT_WAIT_TIMEOUT_MS } from './with-timeout'

describe('renderer boot connect budget', () => {
  it('covers the port-announce deadline and stays finite', () => {
    const announce = resolvePortAnnounceTimeoutMs({})
    const wait = resolveRendererBootWaitMs(announce)

    expect(announce).toBe(DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS)
    expect(wait).toBeGreaterThanOrEqual(announce)
    expect(wait).toBe(announce)
    expect(BACKEND_BOOT_WAIT_TIMEOUT_MS).toBe(wait)
    expect(Number.isFinite(BACKEND_BOOT_WAIT_TIMEOUT_MS)).toBe(true)
    expect(BACKEND_BOOT_WAIT_TIMEOUT_MS).toBeGreaterThan(0)
  })

  it('does not turn a bad announce deadline into an immediate fail or an infinite wait', () => {
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const wait = resolveRendererBootWaitMs(bad)

      expect(wait).toBe(DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS)
      expect(Number.isFinite(wait)).toBe(true)
      expect(wait).toBeGreaterThan(0)
    }
  })

  it('covers a longer finite announce deadline with one wait, not a retry loop', () => {
    const announce = resolvePortAnnounceTimeoutMs({ HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: '120000' })

    expect(resolveRendererBootWaitMs(announce)).toBe(announce)
    expect(Number.isFinite(announce)).toBe(true)
  })
})
