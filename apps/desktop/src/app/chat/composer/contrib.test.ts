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

  it('passes exact native surface identity without breaking legacy handlers', async () => {
    const seen: unknown[] = []

    addMiddleware('legacy', d => ({ ...d, text: `${d.text}:legacy` }), 10)
    addMiddleware(
      'bound',
      (d, invocation) => {
        seen.push(invocation)

        return { ...d, text: `${d.text}:${invocation?.storedSessionId}` }
      },
      20
    )

    const invocation = {
      runtimeSessionId: 'runtime-mail',
      storedSessionId: 'stored-mail',
      target: 'native:stored-mail'
    }

    expect(await runComposerMiddleware({ text: 'x' }, invocation)).toEqual({ text: 'x:legacy:stored-mail' })
    expect(seen).toEqual([invocation])
  })
})
