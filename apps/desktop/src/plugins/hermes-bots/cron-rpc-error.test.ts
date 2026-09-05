/**
 * #94471: React Query stores whatever the queryFn throws, and React 19 then
 * formats it with `(error.name || '').trim()`. IPC / JSON-RPC rejections are
 * frequently plain objects whose `name` is the numeric error code, so that
 * `.trim()` threw "is not a function" and the Routines pane died instead of
 * rendering "Could not load cronjobs".
 *
 * `requestForBot` is the choke point every cron read and mutation goes
 * through, so the coercion is pinned there — through the real function, not
 * the private helper behind it.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const requestLocalProfileGateway = vi.fn()

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, requestLocalProfileGateway } }
})

const { loadRoutines } = await import('./cron')
const { requestForBot } = await import('./routing')

/** Exactly what React 19 does to a query error before painting it. */
function react19Format(error: { message?: unknown; name?: unknown }) {
  return `${((error.name as string) || '').trim()}: ${String(error.message)}`
}

/** The rejection `requestForBot` hands back, after coercion. */
async function rejectionFrom(thrown: unknown) {
  requestLocalProfileGateway.mockRejectedValue(thrown)

  return requestForBot({ name: 'research' }, 'cron.manage', {}).then(
    () => {
      throw new Error('requestForBot resolved; it must propagate the rejection')
    },
    (error: Error) => error
  )
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('a rejection always reaches React with a string name', () => {
  it('coerces a plain JSON-RPC object whose name is an error code', async () => {
    requestLocalProfileGateway.mockRejectedValue({ code: -32603, message: 'cron.manage failed', name: 32000 })

    const error = await loadRoutines('research').then(
      () => {
        throw new Error('loadRoutines resolved; the list read must propagate the failure')
      },
      (thrown: Error) => thrown
    )

    expect(typeof error.name).toBe('string')
    expect(() => react19Format(error)).not.toThrow()
    expect(error.message).toMatch(/cron\.manage failed/)
    // The original rejection is preserved for logs — only the surface is fixed.
    expect((error.cause as { name: number }).name).toBe(32000)
  })

  it('coerces a real Error whose name was overwritten with a number', async () => {
    const weird = new Error('down')

    Object.defineProperty(weird, 'name', { configurable: true, value: 13 })

    const error = await rejectionFrom(weird)

    expect(typeof error.name).toBe('string')
    expect(() => react19Format(error)).not.toThrow()
    expect(error.message).toMatch(/down/)
  })

  it('copies rather than mutates a frozen non-string name', async () => {
    const weird = new Error('down')

    Object.defineProperty(weird, 'name', { configurable: false, value: 13, writable: false })

    const error = await rejectionFrom(weird)

    expect(error).not.toBe(weird)
    expect(typeof error.name).toBe('string')
    expect(() => react19Format(error)).not.toThrow()
  })

  it('copies a sealed Error even where assignment would have worked', async () => {
    const weird = new Error('sealed')

    Object.defineProperty(weird, 'name', { configurable: true, value: 32000, writable: true })
    Object.seal(weird)

    const error = await rejectionFrom(weird)

    expect(error).not.toBe(weird)
    expect(typeof error.name).toBe('string')
    expect(error.message).toMatch(/sealed/)
    // Assignment would have succeeded on a sealed writable property; we still
    // copy so React 19 never sees a numeric name even if mutation is possible.
    expect(weird.name).toBe(32000)
  })

  it('passes an ordinary Error through untouched', async () => {
    const original = new Error('gateway rejected the pause')

    expect(await rejectionFrom(original)).toBe(original)
  })
})
