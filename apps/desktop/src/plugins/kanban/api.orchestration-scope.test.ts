import { afterEach, describe, expect, it, vi } from 'vitest'

// The orchestration read/write must carry the caller's explicit board scope: a
// non-empty slug adds `?board=`, an empty slug (global) omits it. See ./api.ts.
// Mirrors the harness in ./api.connection-scope.test.ts.

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  activeGatewayConnectionId: () => null
}))

const { $boardSlug, bindApi, fetchOrchestration, saveOrchestration } = await import('./api')

const noopStorage = { get: <T>(_key: string, fallback: T) => fallback, remove: vi.fn(), set: vi.fn() }

const requests: Array<{ method?: string; path: string }> = []

const bind = () =>
  bindApi(
    (async (path: string, opts?: { method?: string }) => {
      requests.push({ path, method: opts?.method })

      return {}
    }) as never,
    noopStorage,
    vi.fn(() => vi.fn())
  )

afterEach(() => {
  requests.length = 0
  $boardSlug.set('')
})

describe('kanban orchestration request scope', () => {
  it('carries an explicit board slug; omits it for global scope', async () => {
    const dispose = bind()

    await fetchOrchestration('tsa-mgmt')
    await fetchOrchestration('')

    const orchestration = requests.filter(r => r.path.startsWith('/orchestration'))
    expect(orchestration.map(r => r.path)).toEqual(['/orchestration?board=tsa-mgmt', '/orchestration'])

    dispose()
  })

  it('writes to the explicit board scope', async () => {
    const dispose = bind()

    await saveOrchestration('tsa-mgmt', { default_assignee: 'worker' })
    await saveOrchestration('', { default_assignee: 'worker' })

    const orchestration = requests.filter(r => r.path.startsWith('/orchestration'))
    expect(orchestration.map(r => r.path)).toEqual(['/orchestration?board=tsa-mgmt', '/orchestration'])
    expect(orchestration.every(r => r.method === 'PUT')).toBe(true)

    dispose()
  })
})
