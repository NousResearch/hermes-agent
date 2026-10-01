import { readStatusCode } from './api-transport'
import {
  CLOUD_NOT_SIGNED_IN_MESSAGE,
  CLOUD_SESSION_EXPIRED_MESSAGE,
  cloudLoginRequiredError
} from './cloud-auth-errors'
import type { NativeAccessTokenOptions } from './native-access-token'
import { portalTokenOrgId } from './portal-oauth'

export interface CloudAgentSummary {
  id: string
  name: string
  status: string
  dashboardUrl: null | string
  dashboardGatewayState: string
}

export interface CloudOrgSummary {
  id: string
  slug: null | string
  name: string
  isPersonal: boolean
  role: string
}

export interface CloudDiscoveryDeps {
  portalBaseUrl: string
  /** The desktop portal access token (portal-session getPortalAccessToken). */
  getAccessToken: (options?: NativeAccessTokenOptions) => Promise<null | string>
  /** Cookieless JSON fetch; throws an httpStatusError on >= 400. */
  fetchJson: (url: string, token: null, options: { method: string; bearer: string; timeoutMs?: number }) => Promise<any>
}

/** Project NAS's agent rows to the trimmed DTO the renderer consumes. */
export function trimCloudAgents(body: any): CloudAgentSummary[] {
  const agents = Array.isArray(body?.agents) ? body.agents : []

  return agents
    .filter((a: any) => a && typeof a === 'object' && typeof a.id === 'string')
    .map((a: any) => ({
      id: a.id,
      name: typeof a.name === 'string' ? a.name : a.id,
      status: typeof a.status === 'string' ? a.status : 'unknown',
      dashboardUrl: typeof a.dashboardUrl === 'string' ? a.dashboardUrl : null,
      dashboardGatewayState: typeof a.dashboardGatewayState === 'string' ? a.dashboardGatewayState : 'unknown'
    }))
}

/** Project a NAS org ({ id, slug, name, isPersonal, role }) or null. */
export function trimCloudOrg(org: any): CloudOrgSummary | null {
  if (!org || typeof org !== 'object' || typeof org.id !== 'string') {
    return null
  }

  return {
    id: org.id,
    slug: typeof org.slug === 'string' ? org.slug : null,
    name: typeof org.name === 'string' ? org.name : org.id,
    isPersonal: Boolean(org.isPersonal),
    role: typeof org.role === 'string' ? org.role : 'MEMBER'
  }
}

/**
 * §4 discovery: GET {portal}/api/agents with the desktop bearer. The bearer
 * pins the org (its `org_id` claim), so there is no `?org=` and no 409 team
 * picker — switching team means signing in again and choosing it in the
 * browser. A 401 earns ONE forced refresh of the rejected token and one
 * retry; a second 401, or a refresh that signs the user out, is the
 * needsCloudLogin "expired" error the renderer already understands.
 */
export async function discoverCloudAgentsWithBearer(
  deps: CloudDiscoveryDeps
): Promise<{ agents: CloudAgentSummary[]; org: CloudOrgSummary | null }> {
  const url = `${deps.portalBaseUrl.replace(/\/+$/, '')}/api/agents`
  const fetchAgents = (bearer: string) => deps.fetchJson(url, null, { method: 'GET', bearer, timeoutMs: 15_000 })

  let bearer = await deps.getAccessToken()

  if (!bearer) {
    throw cloudLoginRequiredError(CLOUD_NOT_SIGNED_IN_MESSAGE)
  }

  let body: any

  try {
    body = await fetchAgents(bearer)
  } catch (error) {
    if (readStatusCode(error) !== 401) {
      throw error
    }

    const rotated = await deps.getAccessToken({ forceRefresh: true, rejectedAccessToken: bearer })

    if (!rotated || rotated === bearer) {
      throw cloudLoginRequiredError(CLOUD_SESSION_EXPIRED_MESSAGE, error)
    }

    bearer = rotated

    try {
      body = await fetchAgents(bearer)
    } catch (retryError) {
      if (readStatusCode(retryError) === 401) {
        throw cloudLoginRequiredError(CLOUD_SESSION_EXPIRED_MESSAGE, retryError)
      }

      throw retryError
    }
  }

  // Prefer the org NAS echoes; otherwise name the org the token is pinned to
  // so the renderer records the team the agents actually belong to.
  const tokenOrgId = portalTokenOrgId(bearer)

  return {
    agents: trimCloudAgents(body),
    org: trimCloudOrg(body?.org) ?? (tokenOrgId ? trimCloudOrg({ id: tokenOrgId }) : null)
  }
}
