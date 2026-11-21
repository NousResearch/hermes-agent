import { afterEach, describe, expect, it, vi } from 'vitest'

// Comments must carry a human-readable author, never the old 'desktop'
// literal, whenever the gateway can name the signed-in human. The server
// overrides client claims on gated gateways (`_dashboard_author`); in loopback
// mode the author we send is what gets recorded. See ./api.ts.

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const { addComment, bindApi } = await import('./api')

const noopStorage = { get: <T>(_key: string, fallback: T) => fallback, remove: vi.fn(), set: vi.fn() }

interface Sent {
  body?: Record<string, unknown>
  method?: string
  path: string
}

function bind(restImpl: (path: string, opts?: { body?: Record<string, unknown>; method?: string }) => Promise<unknown>) {
  const sent: Sent[] = []

  const rest = vi.fn(async (path: string, opts?: { body?: Record<string, unknown>; method?: string }) => {
    sent.push({ method: opts?.method, path, body: opts?.body })

    return restImpl(path, opts)
  })

  const dispose = bindApi(rest as never, noopStorage, vi.fn(() => vi.fn()))

  return { dispose, sent }
}

afterEach(() => {
  vi.clearAllMocks()
})

describe('kanban comment author', () => {
  it('sends the gateway-resolved account label when signed in', async () => {
    const { dispose, sent } = bind(async path => (path === '/me' ? { author: 'c.manimaran@gmail.com' } : {}))

    await addComment('t1', 'hello')

    const comment = sent.find(item => item.path.includes('/tasks/t1/comments'))

    expect(comment?.body?.author).toBe('c.manimaran@gmail.com')
    expect(comment?.body?.body).toBe('hello')
    // The hint is fetched once, not per comment.
    await addComment('t1', 'again')
    expect(sent.filter(item => item.path === '/me')).toHaveLength(1)

    dispose()
  })

  it('falls back to the desktop literal when logged out, and retries the hint', async () => {
    const { dispose, sent } = bind(async path => (path === '/me' ? { author: '' } : {}))

    await addComment('t2', 'hello')

    let comment = sent.find(item => item.path.includes('/tasks/t2/comments'))

    expect(comment?.body?.author).toBe('desktop')

    // An empty hint is not cached: the next write asks again (the user may
    // have signed in since).
    await addComment('t2', 'again')

    expect(sent.filter(item => item.path === '/me')).toHaveLength(2)
    comment = sent.filter(item => item.path.includes('/tasks/t2/comments'))[1]

    expect(comment?.body?.author).toBe('desktop')

    dispose()
  })

  it('falls back to the desktop literal when the gateway is unreachable', async () => {
    const { dispose, sent } = bind(async path => {
      if (path === '/me') {
        throw new Error('gateway down')
      }

      return {}
    })

    await addComment('t3', 'hello')

    const comment = sent.find(item => item.path.includes('/tasks/t3/comments'))

    expect(comment?.body?.author).toBe('desktop')

    dispose()
  })
})
