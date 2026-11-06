import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { COMPOSER_AREAS, type ComposerMiddleware, runComposerMiddleware } from './contrib'

const disposers: Array<() => void> = []

function addMiddleware(id: string, handler: ComposerMiddleware['handler'], order?: number) {
  disposers.push(
    registry.register({ id, area: COMPOSER_AREAS.middleware, order, data: { handler } satisfies ComposerMiddleware })
  )
}

afterEach(() => {
  disposers.splice(0).forEach(d => d())
})

describe('runComposerMiddleware', () => {
  it('passes the draft through untouched when nothing is registered', async () => {
    const draft = { text: 'hello' }

    expect(await runComposerMiddleware(draft)).toBe(draft)
  })

  it('chains rewrites in registry order', async () => {
    addMiddleware('b', d => ({ ...d, text: `${d.text}b` }), 20)
    addMiddleware('a', d => ({ ...d, text: `${d.text}a` }), 10)

    expect(await runComposerMiddleware({ text: 'x' })).toEqual({ text: 'xab' })
  })

  it('cancels the send when a handler returns null', async () => {
    addMiddleware('gate', () => null)
    addMiddleware('later', d => ({ ...d, text: 'never' }), 99)

    expect(await runComposerMiddleware({ text: 'x' })).toBeNull()
  })

  it('treats a throwing handler as pass-through', async () => {
    addMiddleware('boom', () => {
      throw new Error('broken plugin')
    })
    addMiddleware('after', d => ({ ...d, text: `${d.text}!` }), 99)

    expect(await runComposerMiddleware({ text: 'x' })).toEqual({ text: 'x!' })
  })

  it('supports async handlers', async () => {
    addMiddleware('async', async d => ({ ...d, text: d.text.toUpperCase() }))

    expect(await runComposerMiddleware({ text: 'quiet' })).toEqual({ text: 'QUIET' })
  })
})

describe('runComposerMiddleware — gating middleware', () => {
  it('cancels the send when a fail-closed handler throws', async () => {
    disposers.push(
      registry.register({
        id: 'gate',
        area: COMPOSER_AREAS.middleware,
        data: {
          failClosed: true,
          handler: () => {
            throw new Error('review host vanished')
          }
        } satisfies ComposerMiddleware
      })
    )
    addMiddleware('later', d => ({ ...d, text: 'never' }), 99)

    expect(await runComposerMiddleware({ text: 'x' })).toBeNull()
  })

  it('cancels the send when a fail-closed handler rejects asynchronously', async () => {
    disposers.push(
      registry.register({
        id: 'gate',
        area: COMPOSER_AREAS.middleware,
        data: { failClosed: true, handler: async () => Promise.reject(new Error('timeout')) } satisfies ComposerMiddleware
      })
    )

    expect(await runComposerMiddleware({ text: 'x' })).toBeNull()
  })

  it('re-stamps the app-set origin so a handler cannot forge it for the next one', async () => {
    const seen: unknown[] = []

    addMiddleware('forger', d => ({ ...d, origin: { kind: 'typed', target: 'tile:other' } }), 10)
    addMiddleware('reader', d => (seen.push(d.origin), d), 20)

    const origin = { kind: 'queued', target: 'main', composerScope: 'session-a' } as const
    const result = await runComposerMiddleware({ text: 'x', origin })

    expect(seen).toEqual([origin])
    expect(result?.origin).toEqual(origin)
  })
})
