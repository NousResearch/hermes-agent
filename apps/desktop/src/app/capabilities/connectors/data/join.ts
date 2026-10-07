import type { McpRuntimeStatus, McpServerRuntimeRow, McpServerSummary } from '@rabbit/shared'

import type { McpCatalogEntry } from '@/rabbit'
import { connectorTitle } from '@/lib/connector-tools'
import { type McpServers, serverEnabled } from '@/lib/mcp-servers'

import { canAuthenticate } from '../../mcp/mcp-status'
import type { BundledEntryInput, ConnectorAuthType, LocalServerInput, LocalServerStatus } from '../types'

const text = (value: null | string | undefined): string | undefined =>
  value !== null && value !== undefined && value.trim() !== '' ? value : undefined

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

export function joinLocalServers({ catalog, servers, status, toolCounts, usage }: LocalJoinInput): LocalServerInput[] {
  return Object.entries(servers).map(([name, entry]) => {
    const bundled = catalog.find(candidate => candidate.name === name)
    const counts = toolCounts?.[name]
    const calls = usage?.[name]
    const raw = status[name] ?? 'unknown'

    return {
      canAuthenticate: canAuthenticate(entry, raw),
      connectorSlug: text(bundled?.connector_slug),
      description: text(bundled?.description),
      enabled: serverEnabled(entry),
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
