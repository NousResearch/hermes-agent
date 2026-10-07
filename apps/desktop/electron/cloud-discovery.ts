import { readJsonErrorBody, readStatusCode } from './api-transport'

// The NAS (status, error code) pairs that mean the remembered team itself is
// gone for this user, as opposed to the credential or the request being bad.
const STALE_TEAM_RESPONSES: Record<number, string> = { 403: 'org_access_denied', 404: 'org_not_found' }

// NAS reports "active" | "degraded" | "down" | "unknown". Only the first three
// say anything about the agent's gateway; "unknown", a missing field, or any
// other token becomes null so no consumer has to string-match a sentinel.
const CLOUD_GATEWAY_STATES = ['active', 'degraded', 'down'] as const

export type CloudGatewayState = (typeof CLOUD_GATEWAY_STATES)[number]

export function cloudGatewayState(value: unknown): CloudGatewayState | null {
  const state = typeof value === 'string' ? value.trim().toLowerCase() : ''

  return (CLOUD_GATEWAY_STATES as readonly string[]).includes(state) ? (state as CloudGatewayState) : null
}

// Project NAS's agent rows to the trimmed DTO the renderer consumes.
export function trimCloudAgents(body: any) {
  const agents: any[] = Array.isArray(body?.agents) ? body.agents : []

  return agents
    .filter(a => a && typeof a === 'object' && typeof a.id === 'string')
    .map(a => ({
      id: a.id as string,
      name: typeof a.name === 'string' ? a.name : (a.id as string),
      status: typeof a.status === 'string' ? a.status : 'unknown',
      dashboardUrl: typeof a.dashboardUrl === 'string' ? a.dashboardUrl : null,
      dashboardGatewayState: cloudGatewayState(a.dashboardGatewayState)
    }))
}

// A remembered team is a discovery preference, not an authorization grant.
// If NAS says it no longer exists or is inaccessible, let NAS resolve current
// memberships (including its 409 team picker). Never retry generic denials.
export async function discoverWithTeamFallback<T>(fetchAgents: (org?: string) => Promise<T>, org?: string): Promise<T> {
  try {
    return await fetchAgents(org)
  } catch (error) {
    const staleTeamCode = STALE_TEAM_RESPONSES[readStatusCode(error)]

    if (org && staleTeamCode && readJsonErrorBody(error)?.error === staleTeamCode) {
      return fetchAgents()
    }

    throw error
  }
}
