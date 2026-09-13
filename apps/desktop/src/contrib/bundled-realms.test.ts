import { expect, it, vi } from 'vitest'

// Unrelated disk watcher/Bot Mode are not part of this bundled package test.
vi.mock('./runtime-loader', () => ({ watchRuntimePlugins: vi.fn() }))
vi.mock('../plugins/hermes-bots/plugin.tsx', () => ({ default: { id: 'hermes-bots', defaultEnabled: false, register() {} } }))

it('ships Realms in the real loader, registering nothing until explicitly toggled', async () => {
  const { $pluginRecords, $pluginDecisions, setPluginEnabled } = await import('./plugins-store')
  const { registry } = await import('./registry')
  const { SESSION_AREAS } = await import('./session')
  $pluginDecisions.set({})

  const contributions = () => Object.values(SESSION_AREAS).flatMap(area => registry.getArea(area))
    .filter(row => row.source === 'plugin:hermes-realms')

  const { discoverBundledPlugins } = await import('./plugins')
  discoverBundledPlugins()
  expect($pluginRecords.get()['hermes-realms']).toMatchObject({ kind: 'bundled', status: 'disabled' })
  expect(contributions()).toEqual([])
  await setPluginEnabled('hermes-realms', true)
  expect($pluginRecords.get()['hermes-realms'].status).toBe('loaded')
  expect(contributions().map(row => row.area)).toEqual(expect.arrayContaining([
    SESSION_AREAS.statusStack, SESSION_AREAS.tileBadge, SESSION_AREAS.listBadge
  ]))
  await setPluginEnabled('hermes-realms', false)
  expect(contributions()).toEqual([])
  discoverBundledPlugins()
  expect($pluginRecords.get()['hermes-realms'].status).toBe('disabled')
}, 60_000)
