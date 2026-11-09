import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { DelegationModelMap, DelegationRouting } from '@/api/delegation'
import type { HermesApiRequest } from '@/global'
import { queryClient } from '@/lib/query-client'

import { routingQueryKey, saveModelRoutes } from './routing-data'

const route = (name: string) => ({ name, description: `${name} route`, provider: 'p', model: `m-${name}` })
const map = (...names: string[]): DelegationModelMap => ({ options: names.map(route), default: names[0] ?? null })

let server: DelegationModelMap
let pending: { body: DelegationModelMap; settle: (fail?: boolean) => void }[]

beforeEach(() => {
  queryClient.clear()
  server = map('a')
  pending = []
  queryClient.setQueryData<DelegationRouting>(routingQueryKey(undefined), {
    configured: 'model_map',
    blocked_reason: null,
    backends: [],
    model_map: server,
    profile_routes: { profiles: [], default: null },
    profiles: []
  })
  // Each PUT waits until the test settles it, so the test picks the order answers come back in.
  window.hermesDesktop = {
    ...window.hermesDesktop,
    api: (request: HermesApiRequest) => {
      if (request.path === '/api/delegation/routing') {
        return Promise.resolve({ ...queryClient.getQueryData(routingQueryKey(undefined)), model_map: server })
      }

      const body = request.body as DelegationModelMap

      return new Promise<unknown>((resolve, reject) =>
        pending.push({
          body,
          settle: fail => {
            if (fail) {
              reject(new Error('500: disk full'))
            } else {
              server = body
              resolve({ ok: true, model_map: body })
            }
          }
        })
      )
    }
  } as typeof window.hermesDesktop
})

afterEach(() => queryClient.clear())

const tick = () => new Promise(resolve => setTimeout(resolve, 0))
const shown = () => queryClient.getQueryData<DelegationRouting>(routingQueryKey(undefined))?.model_map

describe('route saves for one owner', () => {
  it('reach the server in order, and an older answer never undoes a newer edit on screen', async () => {
    const first = saveModelRoutes(undefined, map('a', 'b'), 'failed')
    const second = saveModelRoutes(undefined, map('a', 'b', 'c'), 'failed')

    // The second PUT is not sent until the first is answered: the server applies them in the user's order.
    await tick()
    expect(pending.map(p => p.body)).toEqual([map('a', 'b')])

    pending[0].settle()
    await first
    expect(shown()).toEqual(map('a', 'b', 'c'))

    await tick()
    pending[1].settle()
    await second
    expect([shown(), server]).toEqual([map('a', 'b', 'c'), map('a', 'b', 'c')])
  })

  it('a failed save does not stop the edits queued behind it', async () => {
    const failing = saveModelRoutes(undefined, map('a', 'b'), 'failed')
    const later = saveModelRoutes(undefined, map('a', 'b', 'c'), 'failed')

    await tick()
    pending[0].settle(true)
    await tick()
    pending[1].settle()

    expect(await failing).toBe(false)
    expect(await later).toBe(true)
    expect(shown()).toEqual(map('a', 'b', 'c'))
  })
})
