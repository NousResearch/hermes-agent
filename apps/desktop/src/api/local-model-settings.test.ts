import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./client', () => ({
  hermesApi: vi.fn(async () => ({ revision: 1 })),
  profileScoped: vi.fn(() => ({ connectionId: 'wrong-default', profile: 'wrong-default' }))
}))

import { hermesApi } from './client'
import { getLocalModelSettings, resetLocalModelSettings, saveLocalModelSettings } from './local-models-settings'

const scope = { connectionId: 'gateway-A', profile: 'writer' }

beforeEach(() => vi.clearAllMocks())

describe('local model settings API scope', () => {
  it('sends a saved zero with explicit owner, revision and artifact identity', async () => {
    const request = {
      expected_fingerprint: 'a'.repeat(64),
      expected_revision: 3,
      overrides: { temperature: 0, min_p: 0 }
    }

    await saveLocalModelSettings('model/variant?', request, scope)

    expect(hermesApi).toHaveBeenCalledWith({
      ...scope,
      body: request,
      method: 'PUT',
      path: '/api/local-models/settings/model%2Fvariant%3F'
    })
  })

  it('uses scoped GET and reset routes for the same encoded model identity', async () => {
    await getLocalModelSettings('a/b', scope)
    await resetLocalModelSettings('a/b', {
      expected_fingerprint: 'b'.repeat(64),
      expected_revision: 5,
      keys: ['seed']
    }, scope)

    expect(vi.mocked(hermesApi).mock.calls.map(([request]) => [request.method ?? 'GET', request.path])).toEqual([
      ['GET', '/api/local-models/settings/a%2Fb'],
      ['POST', '/api/local-models/settings/a%2Fb/reset']
    ])
  })
})
