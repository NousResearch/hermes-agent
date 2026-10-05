import { beforeEach, describe, expect, it, vi } from 'vitest'

import { REDUCE_EFFECTS_ATTRIBUTE } from './reduce-effects'

const KEY = 'hermes.desktop.reduce-effects.v1'

/** The store subscribes at module scope, so each case needs a fresh instance. */
async function load() {
  vi.resetModules()

  return await import('./reduce-effects')
}

describe('reduce effects', () => {
  beforeEach(() => {
    window.localStorage.clear()
    document.documentElement.removeAttribute(REDUCE_EFFECTS_ATTRIBUTE)
  })

  it('defaults to full effects, with no attribute on the root', async () => {
    const mod = await load()

    expect(mod.$reduceEffects.get()).toBe(false)
    expect(document.documentElement.hasAttribute(REDUCE_EFFECTS_ATTRIBUTE)).toBe(false)
  })

  it('applies the root attribute and persists when enabled', async () => {
    const mod = await load()

    mod.setReduceEffects(true)

    expect(document.documentElement.hasAttribute(REDUCE_EFFECTS_ATTRIBUTE)).toBe(true)
    expect(window.localStorage.getItem(KEY)).toBe('on')
  })

  it('removes BOTH the attribute and the stored key when disabled', async () => {
    const mod = await load()

    mod.setReduceEffects(true)
    mod.setReduceEffects(false)

    expect(document.documentElement.hasAttribute(REDUCE_EFFECTS_ATTRIBUTE)).toBe(false)
    expect(window.localStorage.getItem(KEY)).toBeNull()
  })

  it('restores a stored preference, attribute included, at module load', async () => {
    window.localStorage.setItem(KEY, 'on')

    const mod = await load()

    // The attribute matters more than the atom: it is what the stylesheet
    // reads, and it has to be right before the first paint.
    expect(mod.$reduceEffects.get()).toBe(true)
    expect(document.documentElement.hasAttribute(REDUCE_EFFECTS_ATTRIBUTE)).toBe(true)
  })

  it('ignores any value other than the opt-in', async () => {
    window.localStorage.setItem(KEY, 'nonsense')

    const mod = await load()

    expect(mod.$reduceEffects.get()).toBe(false)
    expect(document.documentElement.hasAttribute(REDUCE_EFFECTS_ATTRIBUTE)).toBe(false)
  })
})
