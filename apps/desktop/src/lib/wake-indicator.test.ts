import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('wake indicator lifecycle', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('moves from detection to capture and hides when the wake-started session ends', async () => {
    const setState = vi.fn()
    vi.stubGlobal('window', { hermesDesktop: { wakeIndicator: { setState } } })
    const { activateWakeIndicator, syncWakeIndicatorWithVoice } = await import('./wake-indicator')

    activateWakeIndicator()
    expect(syncWakeIndicatorWithVoice(false, 'idle')).toBe(false)
    syncWakeIndicatorWithVoice(true, 'idle')
    syncWakeIndicatorWithVoice(true, 'listening')
    syncWakeIndicatorWithVoice(true, 'thinking')
    syncWakeIndicatorWithVoice(false, 'idle')

    expect(setState.mock.calls.map(([state]) => state)).toEqual(['detected', 'capturing', 'detected', 'hidden'])
  })

  it('does not show for a manually started voice conversation', async () => {
    const setState = vi.fn()
    vi.stubGlobal('window', { hermesDesktop: { wakeIndicator: { setState } } })
    const { syncWakeIndicatorWithVoice } = await import('./wake-indicator')

    expect(syncWakeIndicatorWithVoice(true, 'listening')).toBe(false)
    expect(syncWakeIndicatorWithVoice(false, 'idle')).toBe(false)

    expect(setState).not.toHaveBeenCalled()
  })

  it('deduplicates repeated visual states', async () => {
    const setState = vi.fn()
    vi.stubGlobal('window', { hermesDesktop: { wakeIndicator: { setState } } })
    const { activateWakeIndicator, clearWakeIndicator } = await import('./wake-indicator')

    activateWakeIndicator()
    activateWakeIndicator()
    clearWakeIndicator()
    clearWakeIndicator()

    expect(setState.mock.calls.map(([state]) => state)).toEqual(['detected', 'hidden'])
  })
})
