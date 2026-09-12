import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import {
  getElevenLabsVoices,
  getMemoryProviderConfig,
  getMemoryProviderOAuthStatus,
  restartGateway,
  saveMemoryProviderConfig,
  startMemoryProviderOAuth
} from '@/api/system'
import type { HermesApiRequest } from '@/global'

import { runGatewayRestart } from './system-actions'

const owner = { connectionId: 'fixture-lab', profile: 'research' }

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
})

it('voice and memory settings requests preserve their owner rather than the active source', async () => {
  const api = vi.fn(async (_request: HermesApiRequest) => ({}))
  vi.stubGlobal('hermesDesktop', { api })
  setApiRequestConnection('local')
  setApiRequestProfile('ambient')
  await getElevenLabsVoices(owner)
  await getMemoryProviderConfig('fixture', owner)
  await saveMemoryProviderConfig('fixture', {}, owner)
  await startMemoryProviderOAuth('fixture', owner)
  await getMemoryProviderOAuthStatus('fixture', owner)
  await restartGateway(owner)
  expect(
    api.mock.calls.every(
      ([request]) => request.connectionId === owner.connectionId && request.profile === owner.profile
    )
  ).toBe(true)
})

it('a messaging restart polls the same owner even after the ambient gateway changes', async () => {
  vi.useFakeTimers()

  const api = vi.fn(async (request: HermesApiRequest) =>
    request.method === 'POST' ? { name: 'fixture-action' } : { running: false, exit_code: 0 }
  )

  vi.stubGlobal('hermesDesktop', { api })
  const restarting = runGatewayRestart(owner)
  setApiRequestConnection('local')
  await vi.advanceTimersByTimeAsync(1200)
  expect(await restarting).toBe(true)
  expect(api.mock.calls.map(([request]) => [request.connectionId, request.profile])).toEqual([
    ['fixture-lab', 'research'],
    ['fixture-lab', 'research']
  ])
})

it('does not poll a retired legacy settings owner after a delayed restart response', async () => {
  vi.useFakeTimers()
  let finish!: (value: unknown) => void

  const held = new Promise(resolve => {
    finish = resolve
  })

  const api = vi.fn(async (request: HermesApiRequest) => (request.method === 'POST' ? held : { running: false }))
  vi.stubGlobal('hermesDesktop', { api })
  let current = true
  const restarting = runGatewayRestart(undefined, () => current)
  current = false
  setApiRequestConnection('fixture-other')
  finish({ name: 'fixture-action' })
  await vi.advanceTimersByTimeAsync(1200)
  expect(await restarting).toBe(false)
  expect(api).toHaveBeenCalledTimes(1)
})
