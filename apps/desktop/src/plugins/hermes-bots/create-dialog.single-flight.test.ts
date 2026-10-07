/**
 * The create dialog pre-creates the draft profile so the Capabilities tab has
 * a real backend to point at. Every tab that needs one races for it, so the
 * creation is shared through one in-flight slot — otherwise a user tabbing
 * quickly mints two profiles and the second one leaks.
 *
 * The other half is failure: a rejected creation must CLEAR the slot, or one
 * gateway blip permanently wedges the dialog on a promise that will never
 * resolve and the user can never finish creating the bot.
 */

import { describe, expect, it, vi } from 'vitest'

import { singleFlight } from './create-dialog'

// The dialog is a 1200-line surface pulling in most of the SDK; the helper
// under test touches none of it, so a self-returning stub keeps the module
// graph linkable without pinning any of that surface.
vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  const stub: unknown = new Proxy(function stubbed() {}, {
    apply: () => stub,
    get: (_target, key) => (key === 'then' ? undefined : stub)
  })

  return new Proxy({ atom } as Record<string, unknown>, {
    // A callable `then` would make the namespace look thenable and hang the
    // module loader.
    get: (target, key) =>
      typeof key === 'symbol' || key in target ? target[key as string] : key === 'then' ? undefined : stub,
    has: () => true
  })
})

describe('singleFlight', () => {
  it('shares one in-flight creation across concurrent callers', async () => {
    const ref: { current: null | Promise<string> } = { current: null }
    let calls = 0
    let release!: () => void

    const pending = new Promise<void>(resolve => {
      release = resolve
    })

    const create = async () => {
      calls += 1
      await pending

      return 'researcher'
    }

    const first = singleFlight(ref, create)
    const second = singleFlight(ref, create)

    expect(calls).toBe(1)
    expect(first).toBe(second)

    release()
    await expect(first).resolves.toBe('researcher')
    expect(ref.current).toBe(first)
  })

  it('clears the slot on failure so a retry can succeed', async () => {
    const ref: { current: null | Promise<string> } = { current: null }
    let calls = 0

    await expect(
      singleFlight(ref, async () => {
        calls += 1
        throw new Error('gateway unavailable')
      })
    ).rejects.toThrow(/gateway unavailable/)

    expect(ref.current).toBeNull()

    await expect(
      singleFlight(ref, async () => {
        calls += 1

        return 'researcher'
      })
    ).resolves.toBe('researcher')
    expect(calls).toBe(2)
  })

  it('treats a synchronous throw as a rejected flight, not a crash', async () => {
    const ref: { current: null | Promise<string> } = { current: null }

    await expect(
      singleFlight<string>(ref, () => {
        throw new Error('bad args')
      })
    ).rejects.toThrow(/bad args/)
    expect(ref.current).toBeNull()
  })
})
