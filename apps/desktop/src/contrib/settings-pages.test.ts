import { afterEach, describe, expect, it } from 'vitest'

import type { AgentPluginRow } from '@/store/agent-plugins'

import { createPluginContext } from './plugin'
import { registry } from './registry'
import {
  pluginSettingsEntries,
  pluginSettingsHref,
  resolvePluginSettingsTarget,
  SETTINGS_PLUGINS_AREA
} from './settings-pages'
import type { Contribution } from './types'

const render = () => null

const row = (over: Partial<AgentPluginRow>): AgentPluginRow => ({
  description: '',
  name: 'demo',
  source: 'user',
  status: 'enabled',
  version: '1.0.0',
  ...over
})

const field = { description: '', key: 'region', label: 'Region', required: false, type: 'string' as const }

const page = (id: string, title: string, over: Partial<Contribution> = {}): Contribution => ({
  area: SETTINGS_PLUGINS_AREA,
  id,
  render,
  source: `plugin:${id.split(':')[0]}`,
  title,
  ...over
})

describe('ctx.registerSettingsPage', () => {
  const disposers: Array<() => void> = []

  afterEach(() => {
    disposers.splice(0).forEach(dispose => dispose())
  })

  it('lands in the Settings ▸ Plugins area scoped to the registering plugin, and leaves with it', () => {
    const ctx = createPluginContext('weather', dispose => disposers.push(dispose))

    ctx.registerSettingsPage({
      children: [{ id: 'units', render, title: 'Units' }],
      icon: 'cloud',
      id: 'main',
      render,
      title: 'Weather'
    })

    const [registered] = registry.getArea(SETTINGS_PLUGINS_AREA).filter(c => c.source === 'plugin:weather')

    expect(registered).toMatchObject({ id: 'weather:main', source: 'plugin:weather', title: 'Weather' })
    expect(registered?.data).toMatchObject({ children: [{ id: 'units', title: 'Units' }], icon: 'cloud' })

    // The loader's disable/unload runs every tracked disposer.
    disposers.splice(0).forEach(dispose => dispose())
    expect(registry.getArea(SETTINGS_PLUGINS_AREA).some(c => c.source === 'plugin:weather')).toBe(false)
  })
})

describe('pluginSettingsEntries', () => {
  it('lists registered pages by order then title, dropping malformed ones', () => {
    const entries = pluginSettingsEntries({
      configTitle: 'Agent settings',
      contributions: [
        page('zeta:main', 'Zeta'),
        page('alpha:main', 'Alpha'),
        page('first:main', 'Last alphabetically', { order: -1 }),
        page('broken:main', '', {}),
        page('norender:main', 'No render', { render: undefined })
      ],
      rows: []
    })

    expect(entries.map(entry => entry.title)).toEqual(['Last alphabetically', 'Alpha', 'Zeta'])
  })

  it('keeps valid sub-pages in declared order and drops malformed or duplicate ones', () => {
    const [entry] = pluginSettingsEntries({
      configTitle: 'Agent settings',
      contributions: [
        page('weather:main', 'Weather', {
          data: {
            children: [
              { id: 'units', render, title: 'Units' },
              { id: 'units', render, title: 'Duplicate' },
              { id: 'alerts', render, title: 'Alerts' },
              { id: 'bad', title: 'No render' },
              null
            ]
          }
        })
      ],
      rows: []
    })

    expect(entry?.children.map(child => child.title)).toEqual(['Units', 'Alerts'])
  })

  it('gives every agent plugin with a config_schema its own automatic page', () => {
    const entries = pluginSettingsEntries({
      configTitle: 'Agent settings',
      contributions: [],
      rows: [
        row({ key: 'no-schema', name: 'no-schema' }),
        row({ key: 'notes', name: 'notes', settings_schema: [field] }),
        row({ name: 'keyless', settings_schema: [field] })
      ]
    })

    expect(entries).toHaveLength(1)
    expect(entries[0]).toMatchObject({ agentKey: 'notes', children: [], key: 'agent:notes', title: 'notes' })
  })

  it('folds a unified package’s schema form into its desktop page as a sub-page', () => {
    const entries = pluginSettingsEntries({
      configTitle: 'Agent settings',
      contributions: [page('pixel-overlay:main', 'Pixel Overlay')],
      packageOf: pluginId => (pluginId === 'pixel-overlay' ? 'pixel_overlay' : null),
      rows: [row({ key: 'pixel_overlay', name: 'pixel_overlay', settings_schema: [field] })]
    })

    expect(entries).toHaveLength(1)
    expect(entries[0]?.children).toEqual([{ agentKey: 'pixel_overlay', id: 'config', title: 'Agent settings' }])
  })
})

describe('resolvePluginSettingsTarget', () => {
  const entries = pluginSettingsEntries({
    configTitle: 'Agent settings',
    contributions: [page('weather:main', 'Weather', { data: { children: [{ id: 'units', render, title: 'Units' }] } })],
    packageOf: pluginId => (pluginId === 'weather' ? 'weather_pkg' : null),
    rows: [
      row({ key: 'weather_pkg', name: 'weather_pkg', settings_schema: [field] }),
      row({ key: 'notes', name: 'notes', settings_schema: [field] })
    ]
  })

  it('finds an entry by its key or by the plugin id, and a sub-page under it', () => {
    expect(resolvePluginSettingsTarget(entries, 'weather:main', null)?.entry.title).toBe('Weather')
    expect(resolvePluginSettingsTarget(entries, 'weather', 'units')?.child?.title).toBe('Units')
    expect(resolvePluginSettingsTarget(entries, 'weather', 'nope')?.child).toBeUndefined()
  })

  it('routes an agent:<key> deep link to the folded sub-page or the standalone page', () => {
    const folded = resolvePluginSettingsTarget(entries, 'agent:weather_pkg', null)

    expect(folded?.entry.key).toBe('weather:main')
    expect(folded?.child?.agentKey).toBe('weather_pkg')
    expect(resolvePluginSettingsTarget(entries, 'agent:notes', null)?.entry.agentKey).toBe('notes')
  })

  it('returns null for an unknown plugin (overview fallback)', () => {
    expect(resolvePluginSettingsTarget(entries, 'gone', null)).toBeNull()
    expect(resolvePluginSettingsTarget(entries, null, null)).toBeNull()
  })
})

describe('pluginSettingsHref', () => {
  it('builds the Settings ▸ Plugins deep link', () => {
    expect(pluginSettingsHref()).toBe('/settings?tab=plugins')
    expect(pluginSettingsHref('agent:image_gen/fal')).toBe('/settings?tab=plugins&plugin=agent%3Aimage_gen%2Ffal')
    expect(pluginSettingsHref('weather', 'units')).toBe('/settings?tab=plugins&plugin=weather&ppage=units')
  })
})
