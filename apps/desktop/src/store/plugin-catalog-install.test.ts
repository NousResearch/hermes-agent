// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PLUGIN_CATALOG_URL, type PluginCatalogLookup } from '@/lib/plugin-catalog'

import { $agentPlugins } from './agent-plugins'
import { $notifications } from './notifications'
import { openCatalogPluginInstall, requestPluginCatalogInstallFromDeepLink } from './plugin-catalog-install'
import { $pluginInstallRequest } from './plugin-install-request'

const lookupFor = (result: PluginCatalogLookup) => vi.fn(async () => result)

describe('requestPluginCatalogInstallFromDeepLink', () => {
  beforeEach(() => {
    $pluginInstallRequest.set(null)
    $notifications.set([])
    $agentPlugins.set([])
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('opens the reviewed/pinned catalog dialog exactly like an in-app pick', async () => {
    const lookup = lookupFor({
      ok: true,
      entry: {
        name: 'aihubmix',
        repo: 'https://github.com/AIhubmix/hermes-provider-aihubmix',
        sha: 'b'.repeat(40),
        subdir: 'aihubmix'
      }
    })

    await requestPluginCatalogInstallFromDeepLink('aihubmix', lookup)

    expect(lookup).toHaveBeenCalledWith('aihubmix')
    expect($pluginInstallRequest.get()).toEqual({
      catalogName: 'aihubmix',
      profile: null,
      repo: 'https://github.com/AIhubmix/hermes-provider-aihubmix#aihubmix',
      sha: 'b'.repeat(40)
    })
    expect($notifications.get()).toEqual([])
  })

  it.each([
    ['unknown', 'not in the Hermes plugin catalog'],
    ['unavailable', 'Could not load the Hermes plugin catalog'],
    ['invalid_name', 'missing or invalid']
  ] as const)('%s → error toast, no dialog, no git fallback', async (error, fragment) => {
    await requestPluginCatalogInstallFromDeepLink('not-a-real-plugin', lookupFor({ ok: false, error }))

    expect($pluginInstallRequest.get()).toBeNull()
    const toasts = $notifications.get()
    expect(toasts).toHaveLength(1)
    expect(toasts[0]).toMatchObject({ kind: 'error', title: 'Plugin install link rejected' })
    expect(toasts[0]?.message).toContain(fragment)
  })

  it('resolves against the live catalog feed by default and rejects unknown names', async () => {
    const feed = JSON.stringify([
      { name: 'weather', repo: 'https://github.com/x/weather', sha: 'a'.repeat(40), subdir: '' }
    ])
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(feed, { status: 200 }))

    await requestPluginCatalogInstallFromDeepLink('weather-evil')

    expect(fetchSpy).toHaveBeenCalledWith(PLUGIN_CATALOG_URL, expect.anything())
    expect($pluginInstallRequest.get()).toBeNull()
    expect($notifications.get()[0]?.message).toContain('\u201Cweather-evil\u201D is not in the Hermes plugin catalog')

    await requestPluginCatalogInstallFromDeepLink('weather')

    expect($pluginInstallRequest.get()).toMatchObject({ catalogName: 'weather', repo: 'https://github.com/x/weather' })
  })
})

describe('openCatalogPluginInstall', () => {
  beforeEach(() => {
    $pluginInstallRequest.set(null)
    $notifications.set([])
  })

  it('short-circuits with a success toast when the entry is installed and current', () => {
    $agentPlugins.set([
      {
        catalog_name: 'weather',
        description: '',
        name: 'weather',
        source: 'git',
        status: 'enabled',
        update_available: false,
        version: '1'
      }
    ])

    openCatalogPluginInstall({ name: 'weather', repo: 'https://github.com/x/weather' }, 'workbot')

    expect($pluginInstallRequest.get()).toBeNull()
    expect($notifications.get()[0]).toMatchObject({ kind: 'success' })
  })
})
