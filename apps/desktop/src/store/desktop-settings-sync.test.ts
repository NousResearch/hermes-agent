import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const $gateway = atom<unknown>(null)
const request = vi.fn(async (_profile: string, _method: string, _params?: Record<string, unknown>): Promise<unknown> => undefined)

vi.mock('@/store/gateway', () => ({
  $gateway,
  activeGatewayProfileKey: () => 'work',
  requestGatewayForProfile: request
}))

const { syncJsonSetting } = await import('./desktop-settings-sync')

const isEmptyRecord = (value: Record<string, boolean>) => Object.keys(value).length === 0

// One registration at module scope — same shape as display-toggles.test.ts.
// `mirrors` in the real module is a module-level singleton list consumed on
// every $gateway change, so registering fresh per-`it` would accumulate
// cross-test registrations and double-fire. A single long-lived store,
// reset in beforeEach, mirrors how one real call site behaves in production.
let value: Record<string, boolean> = {}

const store = {
  get: () => value,
  isEmpty: isEmptyRecord,
  onChange: (listener: () => void) => {
    onEdit = listener
    return () => { onEdit = () => undefined }
  },
  set: (next: Record<string, boolean>) => {
    value = next
  }
}
let onEdit: () => void = () => undefined

syncJsonSetting({ configKey: 'desktop.pluginDecisions', ...store })

function configGetCalls() {
  return request.mock.calls.filter(([, method]) => method === 'config.get')
}

function configSetCalls() {
  return request.mock.calls
    .filter(([, method]) => method === 'config.set')
    .map(([profile, , params]) => ({ ...params, profile }))
}

beforeEach(() => {
  value = {}
  request.mockClear()
  request.mockReset()
  request.mockImplementation(async () => undefined)
})

describe('desktop settings sync (JSON mirror)', () => {
  it('mirrors a local edit immediately without waiting for another connection', async () => {
    value = { kanban: true }
    onEdit()
    await Promise.resolve()
    expect(configSetCalls()).toEqual([{ key: 'desktop.pluginDecisions', value: { kanban: true }, profile: 'work' }])
  })

  it('does not let an in-flight pull replace an edit on the same connection', async () => {
    let resolveGet: (result: { value: Record<string, boolean> }) => void = () => undefined
    request.mockImplementation(async (_profile, method) => method === 'config.get'
      ? new Promise(resolve => { resolveGet = resolve })
      : undefined)
    $gateway.set({ generation: 7 })
    await Promise.resolve()
    value = { kanban: false }
    onEdit()
    resolveGet({ value: { kanban: true } })
    await Promise.resolve()
    await Promise.resolve()
    expect(value).toEqual({ kanban: false })
    expect(configSetCalls()).toEqual([{ key: 'desktop.pluginDecisions', value: { kanban: false }, profile: 'work' }])
  })

  it('pushes a non-empty local value on connect (write-through)', async () => {
    value = { kanban: true }

    $gateway.set({ generation: 1 })
    await Promise.resolve()
    await Promise.resolve()

    expect(configSetCalls()).toEqual([{ key: 'desktop.pluginDecisions', value: { kanban: true }, profile: 'work' }])
    expect(configGetCalls()).toEqual([])
  })

  it('imports a non-empty server value into an EMPTY local store (self-heal)', async () => {
    request.mockImplementation(async (_profile, method) =>
      method === 'config.get' ? { value: { kanban: true } } : undefined
    )

    $gateway.set({ generation: 2 })
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()

    expect(value).toEqual({ kanban: true })
    expect(configSetCalls()).toEqual([])
  })

  it('does nothing when both local and server are empty (genuinely fresh install)', async () => {
    request.mockImplementation(async (_profile, method) => (method === 'config.get' ? { value: {} } : undefined))

    $gateway.set({ generation: 3 })
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()

    expect(value).toEqual({})
    expect(configSetCalls()).toEqual([])
  })

  it('never lets a stale in-flight config.get stomp a newer local edit', async () => {
    let resolveFirstGet: (result: { value: Record<string, boolean> }) => void = () => undefined

    request.mockImplementation(async (_profile, method) => {
      if (method !== 'config.get') {
        return undefined
      }

      return new Promise(resolve => {
        resolveFirstGet = resolve
      })
    })

    // First connect starts an in-flight config.get (not yet resolved).
    $gateway.set({ generation: 4 })
    await Promise.resolve()

    // User makes a real edit before that read resolves — a second connect
    // cycle bumps the revision past the in-flight read and pushes instead.
    value = { kanban: false }
    $gateway.set({ generation: 5 })
    await Promise.resolve()

    // The stale read now resolves with an old server value.
    resolveFirstGet({ value: { kanban: true } })
    await Promise.resolve()
    await Promise.resolve()

    // The user's own newer edit must win, not the stale server read.
    expect(value).toEqual({ kanban: false })
  })

  it('never crashes when the gateway is unreachable or too old for the key', async () => {
    request.mockImplementation(async () => {
      throw new Error('method not found')
    })

    $gateway.set({ generation: 6 })
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()

    expect(value).toEqual({})
  })
})
