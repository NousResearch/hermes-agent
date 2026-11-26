import { afterEach, describe, expect, it, vi } from 'vitest'

import { HELD_PROMPT_KEY, HELD_PROMPT_TTL_MS } from './held-prompt'

/** One page load: the store reads what this browser kept, as after sign up's navigation back. */
async function load() {
  vi.resetModules()

  return import('./held-prompt')
}

afterEach(() => window.localStorage.clear())

describe('a message held until sign up', () => {
  it('survives leaving the page and coming back, and is taken exactly once', async () => {
    const before = await load()
    before.holdPrompt('  Research my 5 biggest competitors  ')

    const after = await load()
    expect(after.$heldPrompt.get()).toBe('Research my 5 biggest competitors')
    expect(after.takeHeldPrompt()).toBe('Research my 5 biggest competitors')
    expect(after.takeHeldPrompt()).toBeNull()

    // Another tab, or a reload after it ran: nothing left to run.
    const again = await load()
    expect(again.takeHeldPrompt()).toBeNull()
    expect(window.localStorage.getItem(HELD_PROMPT_KEY)).toBeNull()
  })

  it('keeps a stale tab from taking a prompt another tab already claimed', async () => {
    const first = await load()
    first.holdPrompt('claimed once')

    // The second tab loads the same browser storage, then claims it.
    const second = await load()
    expect(second.$heldPrompt.get()).toBe('claimed once')
    expect(second.takeHeldPrompt()).toBe('claimed once')

    // The first tab's in-memory atom is stale, but storage is the authority: it must not re-send.
    expect(first.$heldPrompt.get()).toBe('claimed once')
    expect(first.takeHeldPrompt()).toBeNull()
    expect(window.localStorage.getItem(HELD_PROMPT_KEY)).toBeNull()
  })

  it('keeps only the latest message', async () => {
    const store = await load()
    store.holdPrompt('first')
    store.holdPrompt('second')

    expect(store.takeHeldPrompt()).toBe('second')
  })

  it('drops a message held too long instead of running it unannounced', async () => {
    const store = await load()
    store.holdPrompt('stale', 1_000)

    expect(store.takeHeldPrompt(1_000 + HELD_PROMPT_TTL_MS)).toBeNull()
    expect(window.localStorage.getItem(HELD_PROMPT_KEY)).toBeNull()
  })

  it('treats a damaged stored value as nothing held', async () => {
    window.localStorage.setItem(HELD_PROMPT_KEY, JSON.stringify({ text: 42 }))

    const store = await load()
    expect(store.$heldPrompt.get()).toBeNull()
  })
})
