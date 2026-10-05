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
 * Pure: no React, no stores. The Settings view feeds it the live registry
 * area and the agent plugin list.
 */

import type { ReactNode } from 'react'

import type { AgentPluginRow } from '@/store/agent-plugins'

import type { Contribution } from './types'

/** Contribution area for plugin settings pages. Same id as the section area
 *  proposed in #91935, so a plugin written against that draft registers here. */
export const SETTINGS_PLUGINS_AREA = 'settings.plugins'

/** Entry-key prefix for automatic config_schema pages (`agent:<key>`). */
export const AGENT_SETTINGS_PREFIX = 'agent:'

/** Sub-page id the schema form takes when folded under a desktop page. */
export const AGENT_SETTINGS_SUBPAGE = 'config'

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
  id: string
  title: string
  /** Plugin-rendered content. Absent on schema-form nodes. */
  render?: () => ReactNode
  /** Agent plugin key whose `config_schema` form this node renders. */
  agentKey?: string
}

export interface PluginSettingsEntry extends PluginSettingsNode {
  /** URL key: the contribution id (`<pluginId>:<pageId>`) or `agent:<key>`. */
  key: string
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

    if (seen.has(child.id) || child.id === AGENT_SETTINGS_SUBPAGE) {
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
      order: typeof contribution.order === 'number' ? contribution.order : 0,
      pluginId: pluginIdOf(contribution.source),
      render: contribution.render,
      title: contribution.title
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
        owner.children.push({ agentKey: row.key, id: AGENT_SETTINGS_SUBPAGE, title: configTitle })
      }

      continue
    }

    entries.push({
      agentKey: row.key,
      children: [],
      id: `${AGENT_SETTINGS_PREFIX}${row.key}`,
      key: `${AGENT_SETTINGS_PREFIX}${row.key}`,
      order: 0,
      title: row.name || row.key
    })
  }

  return entries.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title))
}

/** Resolve `?plugin=&ppage=` to an entry (+ sub-page). `plugin` may be the
 *  entry key, the registering plugin id, or `agent:<key>` (the Capabilities
 *  gear) — which also finds a schema form folded under a desktop page.
 *  Null = show the overview. */
export function resolvePluginSettingsTarget(
  entries: readonly PluginSettingsEntry[],
  plugin: null | string,
  page: null | string
): null | PluginSettingsTarget {
  if (!plugin) {
    return null
  }

  const pick = (entry: PluginSettingsEntry): PluginSettingsTarget => ({
    child: page ? entry.children.find(child => child.id === page) : undefined,
    entry
  })

  const direct = entries.find(entry => entry.key === plugin) ?? entries.find(entry => entry.pluginId === plugin)

  if (direct) {
    return pick(direct)
  }

  if (plugin.startsWith(AGENT_SETTINGS_PREFIX)) {
    const agentKey = plugin.slice(AGENT_SETTINGS_PREFIX.length)

    for (const entry of entries) {
      const child = entry.children.find(node => node.agentKey === agentKey)

      if (child) {
        return { child, entry }
      }
    }
  }

  return null
}

/** Hash-route path to Settings ▸ Plugins (▸ entry (▸ sub-page)). */
export function pluginSettingsHref(plugin?: string, page?: string): string {
  const params = new URLSearchParams({ tab: 'plugins' })

  if (plugin) {
    params.set('plugin', plugin)
  }

  if (plugin && page) {
    params.set('ppage', page)
  }

  return `/settings?${params}`
}
