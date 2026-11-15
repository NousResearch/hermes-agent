import { PALETTE_AREA, type PaletteContribution, ROUTES_AREA } from '@hermes/plugin-sdk'
import { render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { agentHint } from '../plugins/move-to-cloud/location'
import plugin from '../plugins/move-to-cloud/plugin'

import { createPluginContext } from './plugin'
import { ContribRender } from './react/boundary'
import { registry } from './registry'
import {
  pluginSettingsEntries,
  pluginSettingsRouteFrom,
  resolvePluginSettingsTarget,
  SETTINGS_PLUGINS_AREA
} from './settings-pages'

const disposers: Array<() => void> = []

function registerPlugin() {
  plugin.register(createPluginContext(plugin.id, dispose => disposers.push(dispose)))
}

/** What Settings would open for the current hash: the entry its rail and router resolve. */
function settingsTarget() {
  const [path, query = ''] = window.location.hash.slice(1).split('?')

  const entries = pluginSettingsEntries({
    configTitle: 'Agent settings',
    contributions: registry.getArea(SETTINGS_PLUGINS_AREA),
    rows: []
  })

  return { path, target: resolvePluginSettingsTarget(entries, pluginSettingsRouteFrom(new URLSearchParams(query))) }
}

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  window.location.hash = ''
})

describe('Backup & Move lives in Settings', () => {
  it('every ⌘K entry opens its Settings page', () => {
    registerPlugin()

    const entries = registry
      .getArea(PALETTE_AREA)
      .filter(c => c.source === `plugin:${plugin.id}`)
      .map(c => c.data as PaletteContribution)

    expect(entries.map(entry => entry.id).sort()).toEqual(['move-to-cloud.backup', 'move-to-cloud.open'])

    for (const entry of entries) {
      window.location.hash = ''
      void entry.run()

      const { path, target } = settingsTarget()

      expect(path).toBe('/settings')
      expect(target?.entry).toMatchObject({ pluginId: plugin.id, title: 'Backup & Move' })
    }
  })

  it('the portal link’s old route forwards to Settings and keeps the agent it names', () => {
    registerPlugin()

    const [route] = registry.getArea(ROUTES_AREA).filter(c => c.source === `plugin:${plugin.id}`)

    window.location.hash = '#/move-to-cloud?agent=Sam%27s%20agent'
    render(<ContribRender render={route!.render!} />)

    const { path, target } = settingsTarget()

    expect(path).toBe('/settings')
    expect(target?.entry.pluginId).toBe(plugin.id)
    expect(agentHint(window.location.hash)).toBe("Sam's agent")
  })
})
