import { capabilityScoped, hermesApi, type ProfileScope } from './client'

export interface McpCandidate {
  id: string
  name: string
  source: string
  transport: 'stdio' | 'http'
  summary: string
  connectable: boolean
  reason?: string
}

export interface McpDiscoveryResult {
  candidates: McpCandidate[]
  warnings: string[]
}

/** Discovery reads configuration and probes loopback HTTP. It never launches stdio servers. */
export function discoverMcpServers(profile: ProfileScope): Promise<McpDiscoveryResult> {
  return hermesApi<McpDiscoveryResult>({
    ...capabilityScoped(profile),
    path: '/api/mcp/discovery',
    method: 'POST',
    timeoutMs: 60_000
  })
}

/** The backend resolves the candidate again; renderer-supplied commands are never accepted. */
export function connectDiscoveredMcp(
  candidateId: string,
  profile: ProfileScope
): Promise<{ ok: boolean; name: string }> {
  return hermesApi<{ ok: boolean; name: string }>({
    ...capabilityScoped(profile),
    path: '/api/mcp/discovery/connect',
    method: 'POST',
    body: { candidate_id: candidateId },
    timeoutMs: 60_000
  })
}
