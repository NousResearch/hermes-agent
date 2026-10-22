import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'
import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { registerGatewayReconnect } from '@/store/gateway-reconnect'
import { clearNotifications } from '@/store/notifications'

import { runGatewayRestart, runGatewayStart } from './system-actions'

const api = vi.fn()

beforeEach(() => {
  vi.useFakeTimers()
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  setApiRequestConnection('connection-a')
  setApiRequestProfile('profile-a')
  clearNotifications()
})

afterEach(() => {
  vi.useRealTimers()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.resetAllMocks()
})

// Use the actual API helpers and request authority: a helper mock that ignores
// its owner arguments cannot expose a poll moving to another connection.
describe('gateway action request owner', () => {
  it.each([
    ['start', false],
    ['restart', false],
    ['start', true],
    ['restart', true]
  ] as const)(
    'keeps deferred %s on its REST owner after switching away (return: %s)',
    async (action, returnToOrigin) => {
      let accept!: (value: unknown) => void

      const pending = new Promise(resolve => {
        accept = resolve
      })

      api.mockImplementation(async (request: HermesApiRequest) => {
        if (request.path === '/api/status') {return {}}

        if (request.method === 'POST') {return pending}

        return { name: `gateway-${action}`, running: false, exit_code: 0, pid: null, lines: [] }
      })
      const outcome = action === 'start' ? runGatewayStart() : runGatewayRestart()

      // Restart first checks the actual status endpoint before issuing its POST.
      for (let i = 0; i < 8; i += 1) {await Promise.resolve()}
      expect(api.mock.calls.some(([request]) => request.method === 'POST')).toBe(true)
      setApiRequestConnection('connection-b')
      setApiRequestProfile('profile-b')

      if (returnToOrigin) {
        setApiRequestConnection('connection-a')
        setApiRequestProfile('profile-a')
      }

      const handlerB = vi.fn()
      const off = registerGatewayReconnect(handlerB)

      try {
        accept({ ok: true, pid: 4242, name: `gateway-${action}` })
        await vi.advanceTimersByTimeAsync(23_000)
        await expect(outcome).resolves.toBe(true)

        const polls = api.mock.calls
          .map(([request]) => request)
          .filter(request => request.path.startsWith('/api/actions/'))

        expect(polls.length).toBeGreaterThan(0)
        expect(polls).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              connectionId: 'connection-a',
              profile: 'profile-a'
            })
          ])
        )
        expect(polls.every(request => request.connectionId === 'connection-a' && request.profile === 'profile-a')).toBe(
          true
        )
        expect(handlerB).not.toHaveBeenCalled()
      } finally {
        off()
      }
    }
  )

  it('invalidates queued restart followthrough before its handler can touch a newly selected route', async () => {
    let switched = false
    api.mockImplementation(async (request: HermesApiRequest) => {
      if (request.method === 'POST') {return { ok: true, pid: 4242, name: 'gateway-start' }}

      return {
        running: false,
        lines: [],
        get exit_code() {
          // awaitAction resolves, then its caller queues reconnect. Switch in
          // between that scheduling and the controller's handler microtask.
          queueMicrotask(() =>
            queueMicrotask(() => {
              switched = true
              setApiRequestConnection('connection-b')
              setApiRequestProfile('profile-b')
            })
          )

          return 0
        }
      }
    })
    const handler = vi.fn()
    const off = registerGatewayReconnect(handler)

    try {
      const outcome = runGatewayStart()
      await vi.advanceTimersByTimeAsync(23_000)
      await expect(outcome).resolves.toBe(true)
      expect(switched).toBe(true)
      expect(handler).not.toHaveBeenCalled()
    } finally {
      off()
    }
  })

  it('does not silently change the restart POST target while its status preflight is pending', async () => {
    let answer!: (value: unknown) => void

    const pending = new Promise(resolve => {
      answer = resolve
    })

    api.mockImplementation(async (request: HermesApiRequest) => {
      if (request.path === '/api/status') {return pending}

      if (request.method === 'POST') {return { ok: true, pid: 4242, name: 'gateway-restart' }}

      return { running: false, exit_code: 0, lines: [] }
    })
    const outcome = runGatewayRestart()
    setApiRequestConnection('connection-b')
    setApiRequestProfile('profile-b')
    answer({})
    await vi.advanceTimersByTimeAsync(23_000)
    await expect(outcome).resolves.toBe(false)
    expect(api.mock.calls.filter(([request]) => request.method === 'POST')).toHaveLength(0)
  })
})
