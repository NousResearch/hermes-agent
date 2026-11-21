import { map } from 'nanostores'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

const api = vi.hoisted(() => vi.fn())
vi.mock('@/api/client', () => ({ hermesApi: api, $apiRequestScope: map({ connectionId: 'a', profile: 'default' }) }))

const snapshot = (instanceId = '00000000-0000-4000-8000-000000000001') => ({
  version: 1,
  instanceId,
  revision: '0',
  rows: []
})

let stop: (() => void) | undefined
beforeEach(() => {
  vi.resetModules()
  api.mockReset()
  localStorage.clear()
})
afterEach(() => {
  stop?.()
  stop = undefined
})

test('acknowledged instance state replaces renderer favorites; a failed write never looks saved', async () => {
  api.mockResolvedValue(snapshot())
  const remote = await import('./instance-model-preferences')
  const favorites = await import('./favorite-models')
  stop = remote.bindInstanceModels('a', 'default')
  await vi.waitFor(() => expect(remote.$instanceModels.get().status).toBe('ready'))
  api.mockRejectedValueOnce(new Error('HTTP 409'))
  favorites.toggleFavoriteModel('p', 'm')
  await vi.waitFor(() => expect(remote.$instanceModels.get().error).toContain('HR-MODEL-002'))
  expect(favorites.$favoriteModels.get()).toEqual([])
  expect(api.mock.calls.find(c => c[0].method === 'POST')?.[0]).toMatchObject({
    connectionId: 'a',
    body: { baseRevision: '0', field: 'favorite' }
  })
})

test('switching instances fences a late read and never forwards a later click to the old instance', async () => {
  let finish!: (value: unknown) => void
  api
    .mockImplementationOnce(
      () =>
        new Promise(resolve => {
          finish = resolve
        })
    )
    .mockResolvedValue(snapshot('00000000-0000-4000-8000-000000000002'))
  const remote = await import('./instance-model-preferences')
  const favorites = await import('./favorite-models')
  remote.bindInstanceModels('a', 'default')
  stop = remote.bindInstanceModels('b', 'default')
  await vi.waitFor(() => expect(remote.$instanceModels.get().snapshot?.instanceId).toContain('0002'))
  finish({
    ...snapshot(),
    rows: [{ provider: 'p', model: 'old', favorite: true, hidden: null, favoriteRevision: '0', hiddenRevision: '0' }]
  })
  await Promise.resolve()
  expect(favorites.$favoriteModels.get()).toEqual([])
  favorites.toggleFavoriteModel('p', 'new')
  await vi.waitFor(() =>
    expect(api.mock.calls.some(c => c[0].method === 'POST' && c[0].connectionId === 'b')).toBe(true)
  )
})

test('an identified missing API retains local behavior, then a supported runtime reinstalls server writers', async () => {
  api.mockRejectedValueOnce(new Error('Error invoking remote method: Error: 404: Not Found'))
  const remote = await import('./instance-model-preferences')
  const favorites = await import('./favorite-models')
  stop = remote.bindInstanceModels('a', 'default')
  await vi.waitFor(() => expect(remote.$instanceModels.get().status).toBe('legacy'))
  favorites.toggleFavoriteModel('p', 'm')
  expect(localStorage.getItem('hermes.desktop.favorite-models')).toContain('p::m')
  api.mockResolvedValue(snapshot())
  remote.refreshInstanceModels()
  await vi.waitFor(() => expect(remote.$instanceModels.get().status).toBe('ready'))
  api.mockResolvedValue({
    ...snapshot(),
    revision: '1',
    rows: [{ provider: 'p', model: 'm', favorite: true, hidden: null, favoriteRevision: '1', hiddenRevision: '0' }]
  })
  favorites.toggleFavoriteModel('p', 'm')
  await vi.waitFor(() => expect(api.mock.calls.some(c => c[0].method === 'POST')).toBe(true))
})

test('confirmed import skips existing cancellations and only imports unambiguous catalog identities', async () => {
  localStorage.setItem('hermes.desktop.favorite-models', JSON.stringify(['p::cancelled', 'p::new', 'p::unknown']))
  api.mockResolvedValue({
    ...snapshot(),
    revision: '1',
    rows: [
      { provider: 'p', model: 'cancelled', favorite: false, hidden: null, favoriteRevision: '1', hiddenRevision: '0' }
    ]
  })
  const remote = await import('./instance-model-preferences')
  stop = remote.bindInstanceModels('a', 'default')
  await vi.waitFor(() => expect(remote.$instanceModels.get().status).toBe('ready'))
  remote.importPreviousPreferences([{ slug: 'p', name: 'P', models: ['cancelled', 'new'] }])
  await vi.waitFor(() => expect(api.mock.calls.filter(c => c[0].method === 'POST')).toHaveLength(1))
  expect(api.mock.calls.find(c => c[0].method === 'POST')?.[0].body).toMatchObject({
    model: 'new',
    baseRevision: '0',
    value: true
  })
  expect(localStorage.getItem('hermes.desktop.favorite-models')).toContain('cancelled')
})
