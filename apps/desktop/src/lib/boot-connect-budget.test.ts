import { describe, expect, it, vi } from 'vitest'

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

  it('sizes the renderer wait from the announce deadline main publishes through preload', async () => {
    // Main resolves HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS=180000 and hands
    // it to the renderer on the launch-flags bridge before any script runs.
    const announce = resolvePortAnnounceTimeoutMs({ HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: '180000' })
    const previous = window.hermesDesktop

    window.hermesDesktop = { ...previous, portAnnounceTimeoutMs: announce } as typeof window.hermesDesktop
    vi.resetModules()

    try {
      const { BACKEND_BOOT_WAIT_TIMEOUT_MS: wait } = await import('./with-timeout')

      expect(announce).toBe(180_000)
      expect(wait).toBe(announce + DEFAULT_BACKEND_READY_TIMEOUT_MS + BOOT_CONNECT_MARGIN_MS)
    } finally {
      window.hermesDesktop = previous
      vi.resetModules()
    }
  })
})
