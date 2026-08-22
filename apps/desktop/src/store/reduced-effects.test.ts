import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('reduced effects preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    document.documentElement.removeAttribute('data-reduced-effects')
    vi.resetModules()
  })

  it('applies and persists the device-local preference', async () => {
    const { setReducedEffects } = await import('./reduced-effects')

    setReducedEffects(true)

    expect(document.documentElement.hasAttribute('data-reduced-effects')).toBe(true)
    expect(window.localStorage.getItem('hermes.desktop.reducedEffects.v1')).toBe('true')
  })

  it('restores the preference when the renderer starts', async () => {
    window.localStorage.setItem('hermes.desktop.reducedEffects.v1', 'true')

    await import('./reduced-effects')

    expect(document.documentElement.hasAttribute('data-reduced-effects')).toBe(true)
  })
})
