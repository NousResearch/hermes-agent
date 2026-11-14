// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { lookupPluginCatalogEntry, PLUGIN_CATALOG_URL, type PluginCatalogLookup } from '@/lib/plugin-catalog'

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

  it('falls back to the live catalog feed on a backend without the lookup, and rejects unknown names', async () => {
    const feed = JSON.stringify([
      {
        name: 'weather',
        repo: 'https://github.com/x/weather',
        sha: 'a'.repeat(40),
        subdir: '',
        description: 'Shows the  weather\nfor a city. Disclosure — calls a public API.'
      }
    ])

    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(feed, { status: 200 }))

    const lookup = (name: string) =>
      lookupPluginCatalogEntry(name, {
        request: async () => {
          throw new Error('method not found')
        }
      })

    await requestPluginCatalogInstallFromDeepLink('weather-evil', lookup)

    expect(fetchSpy).toHaveBeenCalledWith(PLUGIN_CATALOG_URL, expect.anything())
    expect($pluginInstallRequest.get()).toBeNull()
    expect($notifications.get()[0]?.message).toContain('\u201Cweather-evil\u201D is not in the Hermes plugin catalog')

    await requestPluginCatalogInstallFromDeepLink('weather', lookup)

    // The dialog's one line is the description's first sentence, never its disclosure.
    expect($pluginInstallRequest.get()).toMatchObject({
      catalogName: 'weather',
      repo: 'https://github.com/x/weather',
      summary: 'Shows the weather for a city.'
    })
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

  it("carries the caller's stay-in-place choice to the install dialog", () => {
    $agentPlugins.set([])

    openCatalogPluginInstall({ name: 'weather', repo: 'https://github.com/x/weather' }, 'workbot', {
      stayAfterInstall: true
    })

    expect($pluginInstallRequest.get()).toMatchObject({ catalogName: 'weather', stayAfterInstall: true })
  })

  it('knows a desktop-only entry is installed from the plugin on disk (it has no agent row)', () => {
    $agentPlugins.set([])
    $pluginRecords.set({
      accent: {
        id: 'accent',
        kind: 'disk',
        name: 'Accent Picker',
        packageOrigin: { catalogName: 'accent-picker', repo: 'https://github.com/x/y', sha: 'a'.repeat(40) },
        status: 'loaded'
      }
    })

    openCatalogPluginInstall({ name: 'accent-picker', repo: 'https://github.com/x/y' }, null)

    expect($pluginInstallRequest.get()).toBeNull()
    expect($notifications.get()[0]).toMatchObject({ kind: 'success' })
    $pluginRecords.set({})
  })

  it('opens the dialog for a desktop-only entry the catalog has moved past (an update)', () => {
    $agentPlugins.set([])
    $pluginRecords.set({
      accent: {
        id: 'accent',
        kind: 'disk',
        name: 'Accent Picker',
        packageOrigin: { catalogName: 'accent-picker', repo: 'https://github.com/x/y', sha: 'a'.repeat(40) },
        status: 'loaded'
      }
    })

    openCatalogPluginInstall({ name: 'accent-picker', repo: 'https://github.com/x/y', sha: 'b'.repeat(40) }, null)

    expect($pluginInstallRequest.get()).toMatchObject({ catalogName: 'accent-picker', sha: 'b'.repeat(40) })
    $pluginRecords.set({})
  })

  it('counts a desktop-only entry pinned to an annotated tag as current at the commit that tag names', () => {
    $agentPlugins.set([])
    $pluginRecords.set({
      accent: {
        id: 'accent',
        kind: 'disk',
        name: 'Accent Picker',
        packageOrigin: {
          catalogName: 'accent-picker',
          pin: 't'.repeat(40),
          repo: 'https://github.com/x/y',
          sha: 'c'.repeat(40)
        },
        status: 'loaded'
      }
    })

    openCatalogPluginInstall({ name: 'accent-picker', repo: 'https://github.com/x/y', sha: 't'.repeat(40) }, null)

    expect($pluginInstallRequest.get()).toBeNull()
    expect($notifications.get()[0]).toMatchObject({ kind: 'success' })
    $pluginRecords.set({})
  })

  it("never lets a package's app-wide desktop half stand in for this profile's agent half", () => {
    // Installed in another profile: the desktop half is on disk, this profile has no agent row.
    $agentPlugins.set([])
    $pluginRecords.set({
      pkg: {
        id: 'pkg',
        kind: 'disk',
        name: 'Pkg',
        packageName: 'pkg',
        packageOrigin: { catalogName: 'pkg', repo: 'https://github.com/x/pkg', sha: 'a'.repeat(40) },
        status: 'loaded'
      }
    })

    openCatalogPluginInstall({ name: 'pkg', repo: 'https://github.com/x/pkg', sha: 'a'.repeat(40) }, 'other')

    expect($pluginInstallRequest.get()).toMatchObject({ catalogName: 'pkg', profile: 'other' })
    $pluginRecords.set({})
  })
})
