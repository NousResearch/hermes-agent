// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'

import { $agentPlugins } from './agent-plugins'
import { $notifications } from './notifications'
import { openCatalogPluginInstall } from './plugin-catalog-install'
import { $pluginInstallRequest } from './plugin-install-request'

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
