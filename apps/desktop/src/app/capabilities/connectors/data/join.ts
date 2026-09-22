import type {
  ConnectorAccountRow,
  ConnectorCatalogRow,
  ConnectorRow as ConnectorListRow,
  ConnectorPolicyGetResult,
  ConnectorPolicyLayer,
  McpRuntimeStatus,
  McpServerRuntimeRow,
  McpServerSummary
} from '@hermes/shared'

import type { McpCatalogEntry } from '@/hermes'
import { connectorTitle } from '@/lib/connector-tools'
import type { McpServers } from '@/lib/mcp-servers'

import { canAuthenticate } from '../../mcp/mcp-status'
import { toolRows } from '../derive-tools'
import type {
  BundledEntryInput,
  ConnectorAuthType,
  HostedConnectorInput,
  LocalServerInput,
  LocalServerStatus,
  ToolInput,
  ToolRowModel
} from '../types'

export interface ConnectorPolicyTagRules {
  disable: readonly string[]
  enable: readonly string[]
}

export interface ConnectorPolicyRules {
  allowed: ReadonlySet<string> | null
  denied: ReadonlySet<string>
  revision: string
  tags: ConnectorPolicyTagRules
  tools: Readonly<Record<string, readonly string[]>>
}

export interface ConnectorPolicyView {
  member: ConnectorPolicyRules | null
  others: readonly ConnectorPolicyRules[]
}

export const EMPTY_POLICY: ConnectorPolicyView = { member: null, others: [] }

const NO_TAGS: ConnectorPolicyTagRules = { disable: [], enable: [] }

function rulesOf(layer: ConnectorPolicyLayer): ConnectorPolicyRules {
  const body = layer.body
  const base = { revision: layer.revision, tags: NO_TAGS, tools: {} }

  if (body.mode === 'unrestricted') {
    return { ...base, allowed: null, denied: new Set() }
  }

  if (body.mode === 'deny-all') {
    return { ...base, allowed: new Set(), denied: new Set() }
  }

  const tags: ConnectorPolicyTagRules = {
    disable: body.tags?.disable ?? [],
    enable: body.tags?.enable ?? []
  }

  return body.mode === 'allow'
    ? { allowed: new Set(body.connectors), denied: new Set(), revision: layer.revision, tags, tools: body.tools }
    : {
        allowed: null,
        denied: new Set(body.disabled_connectors),
        revision: layer.revision,
        tags,
        tools: body.tools
      }
}

export function readPolicy(result: ConnectorPolicyGetResult): ConnectorPolicyView {
  const layers = result.layers
  const member = layers.find(layer => layer.kind === 'member')

  return {
    member: member ? rulesOf(member) : null,
    others: layers.filter(layer => layer.kind !== 'member').map(rulesOf)
  }
}

const layerAllows = (rules: ConnectorPolicyRules, slug: string): boolean =>
  (rules.allowed === null || rules.allowed.has(slug)) && !rules.denied.has(slug)

export const memberEnables = (policy: ConnectorPolicyView, slug: string): boolean =>
  policy.member === null || layerAllows(policy.member, slug)

export const orgLocks = (policy: ConnectorPolicyView, slug: string): boolean =>
  policy.others.some(rules => !layerAllows(rules, slug))

export const memberDisabledTools = (policy: ConnectorPolicyView, slug: string): readonly string[] =>
  policy.member?.tools[slug] ?? []

export const memberRevision = (policy: ConnectorPolicyView): string | undefined => policy.member?.revision

export function orgLockedTools(policy: ConnectorPolicyView, slug: string, tools: readonly ToolInput[]): Set<string> {
  const locked = new Set<string>()

  for (const rules of policy.others) {
    for (const tool of rules.tools[slug] ?? []) {
      locked.add(tool)
    }

    for (const tool of tools) {
      const hints = new Set(tool.hints)
      const disabledByTag = rules.tags.disable.some(tag => hints.has(tag))
      const missingEveryEnabledTag = rules.tags.enable.length > 0 && !rules.tags.enable.some(tag => hints.has(tag))

      if (disabledByTag || missingEveryEnabledTag) {
        locked.add(tool.slug)
      }
    }
  }

  return locked
}

export function connectorToolRows(
  policy: ConnectorPolicyView,
  slug: string,
  tools: readonly ToolInput[]
): ToolRowModel[] {
  return toolRows(tools, new Set(memberDisabledTools(policy, slug)), orgLockedTools(policy, slug, tools))
}

export interface HostedJoinInput {
  accounts: readonly ConnectorAccountRow[]
  catalog: readonly ConnectorCatalogRow[]
  list: readonly ConnectorListRow[]
  policy: ConnectorPolicyView
  rulesReadable?: boolean
}

const text = (value: null | string | undefined): string | undefined =>
  value !== null && value !== undefined && value.trim() !== '' ? value : undefined

export function pickAccount(accounts: readonly ConnectorAccountRow[], slug: string): ConnectorAccountRow | undefined {
  const mine = accounts.filter(account => account.connector === slug)
  const rank = (account: ConnectorAccountRow) => (account.active ? 0 : 1)

  return [...mine].sort((a, b) => rank(a) - rank(b) || b.created_at.localeCompare(a.created_at))[0]
}

export function connectorTitles(input: Pick<HostedJoinInput, 'catalog' | 'list'>) {
  const titles: Record<string, string> = {}

  for (const slug of hostedSlugs(input)) {
    titles[slug] = text(input.catalog.find(row => row.slug === slug)?.name) ?? connectorTitle(slug)
  }

  return titles
}

function hostedSlugs({ catalog, list }: Pick<HostedJoinInput, 'catalog' | 'list'>): string[] {
  const slugs = new Set(catalog.map(row => row.slug))

  for (const row of list) {
    if (row.connector) {
      slugs.add(row.connector)
    }
  }

  return [...slugs]
}

export function joinHostedConnectors({
  accounts,
  catalog,
  list,
  policy,
  rulesReadable = true
}: HostedJoinInput): HostedConnectorInput[] {
  return hostedSlugs({ catalog, list }).map(slug => {
    const entry = catalog.find(row => row.slug === slug)
    const row = list.find(candidate => candidate.connector === slug)
    const account = pickAccount(accounts, slug)
    const off = rulesReadable ? memberDisabledTools(policy, slug) : (row?.gateway_disabled_tools ?? [])

    return {
      accountLabel: account?.label,
      connected: account !== undefined || row?.connected === true,
      connectedAt: account?.created_at,
      connectionStatus: account?.status ?? row?.connection_status ?? undefined,
      description: entry?.description,
      disabledTools: off,
      enabled: memberEnables(policy, slug) && row?.enabled !== false,
      inCatalog: entry !== undefined,
      orgLocked: orgLocks(policy, slug),
      slug,
      statusReason: text(account?.status_reason) ?? text(row?.status_reason),
      toolsOff: off.length > 0 ? off.length : undefined
    }
  })
}

export interface LocalJoinInput {
  catalog: readonly McpCatalogEntry[]
  servers: McpServers
  status: Readonly<Record<string, LocalServerStatus>>
  toolCounts?: Readonly<Record<string, { on: number; total: number }>>
  usage?: Readonly<Record<string, number>>
}

interface ServerEntry {
  args?: string[]
  command?: string
  url?: string
}

const targetOf = (entry: McpServers[string]): string => {
  // SAFETY: mcp.json is hand-written, so each field is read back as absent unless it is the string it claims.
  const { args, command, url } = entry as ServerEntry

  if (text(url) !== undefined) {
    return url ?? ''
  }

  const parts = Array.isArray(args) ? args : []

  return [text(command) ?? '', ...parts].join(' ').trim()
}

export function joinLocalServers({
  catalog,
  servers,
  status,
  toolCounts,
  usage
}: LocalJoinInput): LocalServerInput[] {
  return Object.entries(servers).map(([name, entry]) => {
    const bundled = catalog.find(candidate => candidate.name === name)
    const counts = toolCounts?.[name]
    const calls = usage?.[name]
    const raw = status[name] ?? 'unknown'

    return {
      canAuthenticate: canAuthenticate(entry, raw),
      connectorSlug: text(bundled?.connector_slug),
      description: text(bundled?.description),
      enabled: entry.enabled !== false,
      inCatalog: bundled !== undefined,
      name,
      status: raw,
      target: targetOf(entry),
      toolsOn: counts?.on,
      toolsTotal: counts?.total,
      unused: calls === undefined ? undefined : calls === 0
    }
  })
}

function authTypeOf(declared: string): ConnectorAuthType {
  if (declared === 'api_key') {
    return 'apiKey'
  }

  return declared === 'oauth' ? 'oauth' : 'none'
}

export function joinBundledEntries(catalog: readonly McpCatalogEntry[]): BundledEntryInput[] {
  return catalog.map(entry => ({
    authType: authTypeOf(entry.auth_type),
    connectorSlug: text(entry.connector_slug),
    description: text(entry.description),
    name: entry.name,
    needsEnv: entry.required_env.some(field => field.required)
  }))
}

const RUNTIME_STATUS = {
  configured: 'unknown',
  connected: 'ok',
  connecting: 'probing',
  disabled: 'off',
  failed: 'error',
  lazy: 'unknown'
} satisfies Record<McpRuntimeStatus, LocalServerStatus>

export interface PluginServerJoinInput {
  runtime: readonly McpServerRuntimeRow[]
  servers: readonly McpServerSummary[]
}

export function pluginServerRows({ runtime, servers }: PluginServerJoinInput): LocalServerInput[] {
  const live = new Map(runtime.map(row => [row.name, row]))

  return servers
    .filter(row => row.source === 'plugin')
    .map(row => {
      const state = live.get(row.name)
      const split = row.name.indexOf('__')

      return {
        enabled: row.enabled,
        name: row.name,
        plugin: row.plugin ?? '',
        status: state ? RUNTIME_STATUS[state.status] : 'unknown',
        target: text(row.url) ?? [row.command ?? '', ...row.args].join(' ').trim(),
        title: connectorTitle(split === -1 ? row.name : row.name.slice(split + 2)),
        toolsTotal: state?.tools
      }
    })
}
