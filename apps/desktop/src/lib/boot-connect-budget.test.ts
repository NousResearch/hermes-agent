import { describe, expect, it } from 'vitest'

import {
  BOOT_CONNECT_MARGIN_MS,
  DEFAULT_BACKEND_READY_TIMEOUT_MS,
  DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
  resolvePortAnnounceTimeoutMs,
  resolveRendererBootWaitMs
} from '../../../shared/src/desktop-boot-budget'

import { BACKEND_BOOT_WAIT_TIMEOUT_MS } from './with-timeout'

describe('renderer boot connect budget', () => {
  it('outlasts main announcing late and then running the full health poll', () => {
    // Main's slowest healthy path: the port announces just before the
    // deadline, then waitForHermes polls health for its whole window.
    const mainChain = resolvePortAnnounceTimeoutMs({}) + DEFAULT_BACKEND_READY_TIMEOUT_MS

    expect(BACKEND_BOOT_WAIT_TIMEOUT_MS).toBeGreaterThan(mainChain)
    expect(BACKEND_BOOT_WAIT_TIMEOUT_MS - mainChain).toBe(BOOT_CONNECT_MARGIN_MS)
  })

  it('does not turn a bad announce deadline into an immediate fail or an infinite wait', () => {
    const fallback = resolveRendererBootWaitMs(DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS)

    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      expect(resolveRendererBootWaitMs(bad)).toBe(fallback)
    }

    expect(Number.isFinite(fallback)).toBe(true)
  })

  it('stretches with a longer announce override so the health phase stays covered', () => {
    const announce = resolvePortAnnounceTimeoutMs({ HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: '180000' })
    const wait = resolveRendererBootWaitMs(announce)

    expect(announce).toBe(180_000)
    expect(wait).toBeGreaterThan(announce + DEFAULT_BACKEND_READY_TIMEOUT_MS)
    expect(wait).toBeGreaterThan(BACKEND_BOOT_WAIT_TIMEOUT_MS)
  })
})
