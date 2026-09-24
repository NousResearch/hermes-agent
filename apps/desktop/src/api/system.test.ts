import { afterEach, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'

import {
  getMemoryProviderConfig,
  getMemoryProviderOAuthStatus,
  getMemoryStatus,
  saveMemoryProviderConfig,
  setMemoryProvider,
  startMemoryProviderOAuth
} from './system'

const api = vi.fn(async (_request: HermesApiRequest) => ({}))
window.hermesDesktop = { api } as never

afterEach(() => api.mockClear())

it('every memory request carries the owner it was started for, and a settings save never activates', async () => {
  const owner = { connectionId: 'remote-a', profile: 'alpha' }

  await Promise.all([
    getMemoryStatus(owner),
    setMemoryProvider('one', owner),
    getMemoryProviderConfig('one', owner),
    saveMemoryProviderConfig('one', { workspace: 'w' }, owner),
    startMemoryProviderOAuth('one', owner),
    getMemoryProviderOAuthStatus('one', owner)
  ])

  const requests = api.mock.calls.map(([request]) => request)
  expect(requests.map(r => `${r.method ?? 'GET'} ${r.path}`)).toEqual([
    'GET /api/memory',
    'PUT /api/memory/provider',
    'GET /api/memory/providers/one/config?surface=declared',
    'PUT /api/memory/providers/one/config?surface=declared',
    'POST /api/memory/providers/one/oauth/start?surface=declared',
    'GET /api/memory/providers/one/oauth/status?surface=declared'
  ])

  for (const request of requests) {
    expect(request).toMatchObject({ connectionId: 'remote-a', profile: 'alpha', priority: 'foreground' })
  }

  expect(requests[1].body).toEqual({ provider: 'one' })
  expect(requests[3].body).toEqual({ values: { workspace: 'w' }, activate: false })
})
