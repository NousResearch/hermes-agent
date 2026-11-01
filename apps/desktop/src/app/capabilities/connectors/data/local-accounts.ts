import { capabilityScoped, type ProfileScope } from '@/api/client'
import type { McpCatalogEntry } from '@/hermes'
import type { McpServers } from '@/lib/mcp-servers'
import { requestGatewayForAgent } from '@/store/gateway'

import { aliasProblem } from './alias'

/** Only a known OAuth endpoint is a safe preset; never clone headers or client credentials. */
export function localAccountCatalog(catalog: readonly McpCatalogEntry[], servers: McpServers, name: string) {
  const server = servers[name]

  return server?.auth === 'oauth'
    ? catalog.find(entry => entry.auth_type === 'oauth' && entry.url && entry.url === server.url)
    : undefined
}

export function localAccountName(preset: string, alias: string): string {
  return `${preset}-${alias}`
}

const tokenIdentity = (name: string) =>
  (
    name
      .replace(/[^\p{L}\p{N}_-]/gu, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 128) || 'default'
  ).toLowerCase()

const toolIdentity = (name: string) => name.replace(/[^A-Za-z0-9_]/g, '_').toLowerCase()

export function localAccountProblem(preset: string, alias: string, names: readonly string[]) {
  const name = localAccountName(preset, alias)

  if (aliasProblem(alias, []) || !/^[a-z0-9][a-z0-9-]{0,95}$/.test(name)) {
    return 'invalid'
  }

  return names.some(
    existing => tokenIdentity(existing) === tokenIdentity(name) || toolIdentity(existing) === toolIdentity(name)
  )
    ? 'taken'
    : null
}

export async function createLocalAccount(
  name: string,
  preset: string,
  tools: McpServers[string]['tools'],
  profile: ProfileScope
): Promise<void> {
  const { connectionId = null, profile: targetProfile = 'default' } = capabilityScoped(profile)

  const result = await requestGatewayForAgent<{ ok: boolean; error_message?: string }>(
    connectionId,
    targetProfile,
    'mcp.servers.add',
    { name, preset, config: tools === undefined ? {} : { tools } },
    60_000
  )

  if (!result.ok) {
    throw new Error(result.error_message || 'Could not add the MCP account.')
  }
}
