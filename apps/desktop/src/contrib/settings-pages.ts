/**
 * Settings ▸ Plugins — one home for every plugin's preferences, WoW-AddOns
 * style: each plugin gets an entry in the Settings rail, optionally with its
 * own sub-pages beneath it.
 *
 * Two feeds produce entries:
 *  - Desktop plugins call `ctx.registerSettingsPage(...)` (a contribution in
 *    `SETTINGS_PLUGINS_AREA`, so provenance + disable/unload cleanup come from
 *    the registry like every other contribution).
 *  - Agent plugins whose `plugin.yaml` declares a `config_schema` get an
 *    automatic page rendering the schema form — no Desktop code required. A
 *    unified package that does both shows ONE entry; the schema form becomes
 *    its "Agent settings" sub-page.
 *
 * The two feeds live in disjoint identity namespaces. A desktop plugin picks
 * any id it likes, so no string an agent key can form is safe to share with
 * `<pluginId>:<pageId>`: entries carry their `kind`, routes carry it as the
 * query param (`?plugin=` vs `?agent=`), and the automatic schema form is
 * addressed by its agent key, never by a sub-page id a plugin could also pick.
 *
 * Pure: no React, no stores. The Settings view feeds it the live registry
 * area and the agent plugin list.
 */

import type { ReactNode } from 'react'

import type { AgentPluginRow } from '@/store/agent-plugins'

import type { Contribution } from './types'

/** Contribution area for plugin settings pages. Same id as the section area
 *  proposed in #91935, so a plugin written against that draft registers here. */
export const SETTINGS_PLUGINS_AREA = 'settings.plugins'

/** One-shot hand-off: the profile a deep link (the Capabilities gear) wants
 *  Settings to edit. Settings adopts it into its scope and drops the param. */
export const PLUGIN_SETTINGS_PROFILE_PARAM = 'profile'

export interface PluginSettingsSubpage {
  /** Unique within the page; becomes the `ppage` URL param. */
  id: string
  title: string
  render: () => ReactNode
}

/** What `ctx.registerSettingsPage` accepts. */
export interface PluginSettingsPage {
  /** Unique within the plugin (namespaced to `<pluginId>:<id>` on register). */
  id: string
  /** Rail label and breadcrumb. */
  title: string
  /** Codicon name for the rail (e.g. `'cloud'`); a plug when omitted. */
  icon?: string
  /** Ascending; ties sort alphabetically by title. */
  order?: number
  /** The page's landing content. */
  render: () => ReactNode
  /** Sub-pages listed beneath the entry when it is selected. */
  children?: PluginSettingsSubpage[]
}

export interface PluginSettingsNode {
  /** Plugin-chosen sub-page id (`ppage`). Schema-form nodes are addressed by
   *  `agentKey` instead, so they never take an id a plugin could choose. */
  id: string
  title: string
  /** Plugin-rendered content. Absent on schema-form nodes. */
  render?: () => ReactNode
  /** Agent plugin key whose `config_schema` form this node renders. */
  agentKey?: string
}

/** Which feed produced an entry; part of its identity. */
export type PluginSettingsEntryKind = 'agent' | 'desktop'

export interface PluginSettingsEntry extends PluginSettingsNode {
  kind: PluginSettingsEntryKind
  /** Route value within `kind`: the contribution id (`<pluginId>:<pageId>`)
   *  for desktop pages, the agent plugin key for automatic pages. */
  key: string
  /** Collision-free identity across both kinds: `<kind>:<key>`. */
  uid: string
  icon?: string
  order: number
  /** Registering desktop plugin id, when contributed by one. */
  pluginId?: string
  children: PluginSettingsNode[]
}

export interface PluginSettingsTarget {
  entry: PluginSettingsEntry
  child?: PluginSettingsNode
}

/** Where Settings ▸ Plugins should land: a desktop entry (by contribution id
 *  or plugin id, optionally a sub-page) or an agent plugin's schema form
 *  (standalone or folded under its package's desktop page). */
export type PluginSettingsRoute = { agent: string } | { page?: string; plugin: string }

/** The route that reaches `child` (or the entry's landing page). */
export function pluginSettingsRouteOf(entry: PluginSettingsEntry, child?: PluginSettingsNode): PluginSettingsRoute {
  if (child?.agentKey) {
    return { agent: child.agentKey }
  }

  if (entry.kind === 'agent') {
    return { agent: entry.key }
  }

  return child ? { page: child.id, plugin: entry.key } : { plugin: entry.key }
}

/** Read the route from Settings' query string (`null` = the overview). */
export function pluginSettingsRouteFrom(params: URLSearchParams): null | PluginSettingsRoute {
  const agent = params.get('agent')

  if (agent) {
    return { agent }
  }

  const plugin = params.get('plugin')

  return plugin ? { page: params.get('ppage') ?? undefined, plugin } : null
}

/** Rail identity of a node: disjoint across kinds, and a folded schema form
 *  keeps the identity its standalone page would have had. */
export function pluginSettingsNavId(entry: PluginSettingsEntry, child?: PluginSettingsNode): string {
  if (child?.agentKey) {
    return `plugins:agent:${child.agentKey}`
  }

  return child ? `plugins:${entry.uid}:${child.id}` : `plugins:${entry.uid}`
}

/** Shape a registration into a registry contribution (the plugin context
 *  namespaces the id and stamps the source). */
export function settingsPageContribution(page: PluginSettingsPage): Omit<Contribution, 'source'> {
  return {
    area: SETTINGS_PLUGINS_AREA,
    data: { children: page.children ?? [], icon: page.icon },
    id: page.id,
    order: page.order,
    render: page.render,
    title: page.title
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const nonEmpty = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0

/** Runtime plugins are untrusted JS: validate the sub-page list rather than
 *  trusting the TypeScript shape. Malformed and duplicate ids are dropped. */
function subpagesOf(data: unknown): PluginSettingsNode[] {
  const raw = isRecord(data) && Array.isArray(data.children) ? data.children : []
  const seen = new Set<string>()
  const nodes: PluginSettingsNode[] = []

  for (const child of raw) {
    if (!isRecord(child) || !nonEmpty(child.id) || !nonEmpty(child.title) || typeof child.render !== 'function') {
      continue
    }

    if (seen.has(child.id)) {
      continue
    }

    seen.add(child.id)
    nodes.push({ id: child.id, render: child.render as () => ReactNode, title: child.title })
  }

  return nodes
}

const pluginIdOf = (source: string | undefined) =>
  source?.startsWith('plugin:') ? source.slice('plugin:'.length) : undefined

export interface PluginSettingsEntriesInput {
  /** Live `SETTINGS_PLUGINS_AREA` contributions. */
  contributions: readonly Contribution[]
  /** Agent plugin rows (already filtered to what the user can manage). */
  rows: readonly AgentPluginRow[]
  /** Label for a schema form folded under a desktop page. */
  configTitle: string
  /** Desktop plugin id → its package folder name (the agent row's `name`),
   *  for unified packages. */
  packageOf?: (pluginId: string) => null | string | undefined
}

export function pluginSettingsEntries({
  configTitle,
  contributions,
  packageOf,
  rows
}: PluginSettingsEntriesInput): PluginSettingsEntry[] {
  const entries: PluginSettingsEntry[] = []

  for (const contribution of contributions) {
    if (!nonEmpty(contribution.title) || typeof contribution.render !== 'function') {
      continue
    }

    const icon = isRecord(contribution.data) && nonEmpty(contribution.data.icon) ? contribution.data.icon : undefined

    entries.push({
      children: subpagesOf(contribution.data),
      icon,
      id: contribution.id,
      key: contribution.id,
      kind: 'desktop',
      order: typeof contribution.order === 'number' ? contribution.order : 0,
      pluginId: pluginIdOf(contribution.source),
      render: contribution.render,
      title: contribution.title,
      uid: `desktop:${contribution.id}`
    })
  }

  for (const row of rows) {
    if (!row.key || !row.settings_schema?.length) {
      continue
    }

    const owner = entries.find(entry => {
      if (!entry.pluginId || entry.agentKey) {
        return false
      }

      const pkg = packageOf?.(entry.pluginId)

      return row.name === pkg || row.name === entry.pluginId || row.key === entry.pluginId
    })

    if (owner) {
      if (!owner.children.some(child => child.agentKey)) {
        owner.children.push({ agentKey: row.key, id: row.key, title: configTitle })
      }

      continue
    }

    entries.push({
      agentKey: row.key,
      children: [],
      id: row.key,
      key: row.key,
      kind: 'agent',
      order: 0,
      title: row.name || row.key,
      uid: `agent:${row.key}`
    })
  }

  return entries.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title))
}

/** Resolve a route to an entry (+ sub-page). `plugin` matches desktop
 *  entries only (by contribution id, then plugin id) and `page` only their
 *  plugin-chosen sub-pages; `agent` matches the agent key's schema form,
 *  standalone or folded under a desktop page. Null = show the overview. */
export function resolvePluginSettingsTarget(
  entries: readonly PluginSettingsEntry[],
  route: null | PluginSettingsRoute
): null | PluginSettingsTarget {
  if (!route) {
    return null
  }

  if ('agent' in route) {
    const standalone = entries.find(entry => entry.kind === 'agent' && entry.key === route.agent)

    if (standalone) {
      return { entry: standalone }
    }

    for (const entry of entries) {
      const child = entry.kind === 'desktop' ? entry.children.find(node => node.agentKey === route.agent) : undefined

      if (child) {
        return { child, entry }
      }
    }

    return null
  }

  const desktop = entries.filter(entry => entry.kind === 'desktop')

  const entry =
    desktop.find(candidate => candidate.key === route.plugin) ??
    desktop.find(candidate => candidate.pluginId === route.plugin)

  if (!entry) {
    return null
  }

  return {
    child: route.page ? entry.children.find(child => !child.agentKey && child.id === route.page) : undefined,
    entry
  }
}

/** Settings ▸ Plugins href for a route, optionally handing Settings the
 *  profile to edit (`PLUGIN_SETTINGS_PROFILE_PARAM`). */
export function pluginSettingsRouteHref(route: null | PluginSettingsRoute, profile?: null | string): string {
  const params = new URLSearchParams({ tab: 'plugins' })

  if (route && 'agent' in route) {
    params.set('agent', route.agent)
  } else if (route) {
    params.set('plugin', route.plugin)

    if (route.page) {
      params.set('ppage', route.page)
    }
  }

  if (profile) {
    params.set(PLUGIN_SETTINGS_PROFILE_PARAM, profile)
  }

  return `/settings?${params}`
}

/** Hash-route path to Settings ▸ Plugins (▸ entry (▸ sub-page)). */
export function pluginSettingsHref(plugin?: string, page?: string): string {
  return pluginSettingsRouteHref(plugin ? { page, plugin } : null)
}
