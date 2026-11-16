import { afterEach, describe, expect, it } from 'vitest'

import type { AgentPluginRow } from '@/store/agent-plugins'

import { createPluginContext } from './plugin'
import { registry } from './registry'
import {
  pluginSettingsEntries,
  pluginSettingsHref,
  pluginSettingsNavId,
  pluginSettingsRouteFrom,
  pluginSettingsRouteHref,
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
    expect(entries[0]).toMatchObject({
      agentKey: 'notes',
      children: [],
      key: 'notes',
      kind: 'agent',
      title: 'notes',
      uid: 'agent:notes'
    })
  })

  it('folds a unified package’s schema form into its desktop page as a sub-page', () => {
    const entries = pluginSettingsEntries({
      configTitle: 'Agent settings',
      contributions: [page('pixel-overlay:main', 'Pixel Overlay')],
      packageOf: pluginId => (pluginId === 'pixel-overlay' ? 'pixel_overlay' : null),
      rows: [row({ key: 'pixel_overlay', name: 'pixel_overlay', settings_schema: [field] })]
    })

    expect(entries).toHaveLength(1)
    expect(entries[0]?.children).toEqual([{ agentKey: 'pixel_overlay', id: 'pixel_overlay', title: 'Agent settings' }])
  })

  // Review of #133182: a desktop plugin may pick ANY id, so `agent` + page
  // `notes` once produced the same `agent:notes` key as agent plugin `notes`.
  it('keeps desktop pages and automatic agent pages in disjoint identity namespaces', () => {
    const entries = pluginSettingsEntries({
      configTitle: 'Agent settings',
      contributions: [page('agent:notes', 'Desktop notes')],
      rows: [row({ key: 'notes', name: 'notes', settings_schema: [field] })]
    })

    expect(entries.map(entry => entry.uid).sort()).toEqual(['agent:notes', 'desktop:agent:notes'])
    expect(new Set(entries.map(entry => pluginSettingsNavId(entry))).size).toBe(2)
    expect(resolvePluginSettingsTarget(entries, { agent: 'notes' })?.entry.kind).toBe('agent')
    expect(resolvePluginSettingsTarget(entries, { plugin: 'agent:notes' })?.entry.kind).toBe('desktop')
    expect(resolvePluginSettingsTarget(entries, { plugin: 'agent' })?.entry.kind).toBe('desktop')
  })

  // Review of #133182: `config` was silently reserved for the folded schema form.
  it('never consumes a plugin-chosen sub-page id for the folded schema form', () => {
    const entries = pluginSettingsEntries({
      configTitle: 'Agent settings',
      contributions: [
        page('pixel-overlay:main', 'Pixel Overlay', {
          data: { children: [{ id: 'config', render, title: 'Configuration child' }] }
        })
      ],
      packageOf: pluginId => (pluginId === 'pixel-overlay' ? 'config' : null),
      rows: [row({ key: 'config', name: 'config', settings_schema: [field] })]
    })

    expect(entries[0]?.children.map(child => child.title)).toEqual(['Configuration child', 'Agent settings'])
    expect(resolvePluginSettingsTarget(entries, { page: 'config', plugin: 'pixel-overlay' })?.child?.title).toBe(
      'Configuration child'
    )
    expect(resolvePluginSettingsTarget(entries, { agent: 'config' })?.child?.title).toBe('Agent settings')

    const [child, schema] = entries[0]!.children

    expect(pluginSettingsNavId(entries[0]!, child)).not.toBe(pluginSettingsNavId(entries[0]!, schema))
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

  it('finds a desktop entry by its key or by the plugin id, and a sub-page under it', () => {
    expect(resolvePluginSettingsTarget(entries, { plugin: 'weather:main' })?.entry.title).toBe('Weather')
    expect(resolvePluginSettingsTarget(entries, { page: 'units', plugin: 'weather' })?.child?.title).toBe('Units')
    expect(resolvePluginSettingsTarget(entries, { page: 'nope', plugin: 'weather' })?.child).toBeUndefined()
  })

  it('routes an agent key to the folded sub-page or the standalone page', () => {
    const folded = resolvePluginSettingsTarget(entries, { agent: 'weather_pkg' })

    expect(folded?.entry.key).toBe('weather:main')
    expect(folded?.child?.agentKey).toBe('weather_pkg')
    expect(resolvePluginSettingsTarget(entries, { agent: 'notes' })?.entry.agentKey).toBe('notes')
    // An agent key is never matched as a desktop plugin id, nor vice versa.
    expect(resolvePluginSettingsTarget(entries, { plugin: 'notes' })).toBeNull()
    expect(resolvePluginSettingsTarget(entries, { agent: 'weather' })).toBeNull()
  })

  it('returns null for an unknown plugin (overview fallback)', () => {
    expect(resolvePluginSettingsTarget(entries, { plugin: 'gone' })).toBeNull()
    expect(resolvePluginSettingsTarget(entries, null)).toBeNull()
  })
})

describe('pluginSettingsHref', () => {
  it('builds the Settings ▸ Plugins deep link', () => {
    expect(pluginSettingsHref()).toBe('/settings?tab=plugins')
    expect(pluginSettingsHref('weather')).toBe('/settings?tab=plugins&plugin=weather')
    expect(pluginSettingsHref('weather', 'units')).toBe('/settings?tab=plugins&plugin=weather&ppage=units')
  })

  it('round-trips agent routes and the profile hand-off', () => {
    const href = pluginSettingsRouteHref({ agent: 'image_gen/fal' }, 'review-b')

    expect(href).toBe('/settings?tab=plugins&agent=image_gen%2Ffal&profile=review-b')

    const params = new URLSearchParams(href.split('?')[1])

    expect(pluginSettingsRouteFrom(params)).toEqual({ agent: 'image_gen/fal' })
    expect(pluginSettingsRouteFrom(new URLSearchParams('plugin=weather&ppage=units'))).toEqual({
      page: 'units',
      plugin: 'weather'
    })
    expect(pluginSettingsRouteFrom(new URLSearchParams('tab=plugins'))).toBeNull()
  })
})
