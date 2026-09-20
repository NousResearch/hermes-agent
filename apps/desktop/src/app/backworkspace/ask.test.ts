import { beforeEach, describe, expect, it, vi } from 'vitest'

const request = vi.fn<(...args: unknown[]) => Promise<unknown>>()
const release = vi.fn()
const listeners = new Map<string, Set<(event: unknown) => void>>()

vi.mock('@/store/gateway', () => ({
  requestGatewayForAgent: (...args: unknown[]) => request(...args),
  retainGatewayForAgent: () => Promise.resolve(release)
}))
vi.mock('@/contrib/events', () => ({
  onGatewayEvent: (type: string, listener: (event: unknown) => void) => {
    const set = listeners.get(type) ?? new Set<(event: unknown) => void>()

    set.add(listener)
    listeners.set(type, set)

    return () => set.delete(listener)
  }
}))

const ROUTE = { connectionId: null, profile: 'default' }
const PAGE_PATH = '/home/u/.hermes/backworkspace/20260920_101010_abcdef.md'

function emit(type: string, event: object) {
  for (const listener of [...(listeners.get(type) ?? [])]) {
    listener(event)
  }
}

function calls(method: string) {
  return request.mock.calls.filter(call => call[2] === method)
}

function submitted() {
  return calls('prompt.submit').map(call => (call[3] as { text: string }).text)
}

// Fresh module state (the cached session per owner) per test.
async function loadAskModule() {
  vi.resetModules()

  return import('./ask')
}

beforeEach(() => {
  request.mockReset()
  release.mockReset()
  listeners.clear()
})

describe('askBackworkspace', () => {
  it('creates the hidden page session once and reuses it for the next question', async () => {
    const ask = await loadAskModule()

    request.mockImplementation((_connection, _profile, method) => {
      if (method === 'session.list') {
        return Promise.resolve({ sessions: [] })
      }

      return Promise.resolve(method === 'session.create' ? { session_id: 'runtime-1' } : {})
    })

    const first = ask.askBackworkspace({ asker: ROUTE, handle: '@hermes', route: ROUTE }, 'what is this?', PAGE_PATH)

    await vi.waitFor(() => expect(submitted()).toHaveLength(1))
    emit('message.complete', { payload: { text: 'an answer' }, session_id: 'runtime-1' })
    await expect(first).resolves.toBe('an answer')

    const second = ask.askBackworkspace({ asker: ROUTE, handle: '@hermes', route: ROUTE }, 'and this?', PAGE_PATH)

    await vi.waitFor(() => expect(submitted()).toHaveLength(2))
    emit('message.complete', { payload: { text: 'a second answer' }, session_id: 'runtime-1' })
    await expect(second).resolves.toBe('a second answer')

    expect(calls('session.create')).toHaveLength(1)
    // Every turn carries the frame: the agent must answer, not write the file.
    expect(submitted()[0]).toContain(PAGE_PATH)
    expect(submitted()[0]).toContain('what is this?')
    expect(submitted()[1]).toContain('Do not write to the page')
    expect(submitted()[1]).toContain('and this?')
    expect(release).toHaveBeenCalledTimes(2)
  })

  it('resolves the session again when the backend lost it, and asks once more', async () => {
    const ask = await loadAskModule()
    let submits = 0

    request.mockImplementation((_connection, _profile, method) => {
      if (method === 'session.list') {
        return Promise.resolve({ sessions: [{ id: 'stored-1' }] })
      }

      if (method === 'session.resume') {
        return Promise.resolve({ session_id: `runtime-${submits + 1}` })
      }

      if (method !== 'prompt.submit') {
        return Promise.resolve({})
      }

      submits += 1

      return submits === 1
        ? Promise.reject(Object.assign(new Error('session not found'), { code: 4001 }))
        : Promise.resolve({})
    })

    const answer = ask.askBackworkspace({ asker: ROUTE, handle: '@hermes', route: ROUTE }, 'still there?', PAGE_PATH)

    await vi.waitFor(() => expect(submits).toBe(2))
    emit('message.complete', { payload: { text: 'still here' }, session_id: 'runtime-2' })

    await expect(answer).resolves.toBe('still here')
    expect(calls('session.create')).toHaveLength(0)
    expect(release).toHaveBeenCalledTimes(1)
  })
})
