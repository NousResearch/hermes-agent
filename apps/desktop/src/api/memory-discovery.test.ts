import { afterEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from './client'
import { getMemoryStatus, saveMemoryProviderConfig } from './system'

afterEach(() => {
  vi.unstubAllGlobals()
  $apiRequestScope.set({ connectionId: null, profile: null })
})

it('pins discovery and save-only configuration to the captured owner across A → B → A including an untagged connection', async () => {
  const api = vi.fn(async (_request: { path: string; method?: string }) => ({
    active: 'builtin',
    providers: [],
    builtin_files: { memory: 0, user: 0 }
  }))

  vi.stubGlobal('hermesDesktop', { api })

  const owners = [
    { connectionId: null, profile: 'a' },
    { connectionId: 'remote', profile: 'b' },
    { connectionId: null, profile: 'a' }
  ]

  for (const owner of owners) {
    $apiRequestScope.set({ connectionId: 'unrelated', profile: 'other' })

    const route = {
      ...(owner.connectionId ? { connectionId: owner.connectionId } : {}),
      profile: owner.profile,
      priority: 'foreground'
    }

    await getMemoryStatus(owner)
    expect(api.mock.calls.at(-1)?.[0]).toEqual({ ...route, path: '/api/memory' })
    await saveMemoryProviderConfig('fixture', { workspace: 'updated' }, 'ignored', owner, false)
    expect(api.mock.calls.at(-1)?.[0]).toEqual({
      ...route,
      path: '/api/memory/providers/fixture/config?surface=declared',
      method: 'PUT',
      body: { values: { workspace: 'updated' }, activate: false }
    })
  }

  await saveMemoryProviderConfig('fixture', { workspace: 'legacy' }, 'a')
  expect(api.mock.calls.at(-1)?.[0]).toMatchObject({ body: { values: { workspace: 'legacy' } } })
  expect(api.mock.calls.at(-1)?.[0]).not.toHaveProperty('body.activate')
})
