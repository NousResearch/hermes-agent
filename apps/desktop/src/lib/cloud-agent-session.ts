import type { DesktopCloudAgentSignInResult } from '@/global'

/**
 * Re-establish a Hermes Cloud agent's gateway session from the shared portal
 * session: drop any lapsed gateway cookies, make sure the portal session is
 * live (interactive login window when it has lapsed), then run the silent
 * per-agent cascade so a fresh gateway session cookie lands for `url`.
 *
 * This is the recovery behind the "Open Settings → Gateway and sign in again"
 * copy: a saved cloud connection on a local-primary device has no other
 * sign-in surface when its gateway session lapses. The boot overlay used this
 * sequence inline; Settings reuses the exact same ladder so there is ONE
 * recovery path, not two that drift.
 */

export interface CloudAgentSessionBridge {
  cloud?: {
    status: () => Promise<{ signedIn: boolean }>
    login: () => Promise<{ ok: boolean; signedIn: boolean }>
    agentSignIn: (dashboardUrl: string) => Promise<DesktopCloudAgentSignInResult>
  }
  oauthLogoutConnectionConfig?: (url: string) => Promise<unknown>
}

/**
 * One silent cascade attempt against `url`. Assumes the portal session is
 * live (call `ensureCloudPortalSession` first); returns whether the gateway
 * session cookie landed.
 */
async function cascadeCloudAgentSession(desktop: CloudAgentSessionBridge, url: string): Promise<boolean> {
  const result = await desktop.cloud!.agentSignIn(url)

  return result.connected === true
}

/** True when the portal session is live, signing in (a login window) when it is not. */
async function ensureCloudPortalSession(desktop: CloudAgentSessionBridge): Promise<boolean> {
  const status = await desktop.cloud!.status()

  if (!status.signedIn) {
    const login = await desktop.cloud!.login()

    if (!login.signedIn) {
      return false
    }
  }

  return true
}

/**
 * Full recovery ladder for one cloud agent's gateway session. Returns
 * `'connected'` when the silent cascade landed a fresh gateway session,
 * `'portal-incomplete'` when the interactive portal login never completed,
 * or throws the underlying error (network, portal outage, cascade failure)
 * for the caller to surface the way it already surfaces failures.
 */
export async function reestablishCloudAgentSession(
  desktop: CloudAgentSessionBridge,
  url: string
): Promise<'connected' | 'portal-incomplete'> {
  // Drop this gateway's lapsed cookies first — a dead AT/RT pair must not be
  // mistaken for a live session by the cascade or any liveness probe.
  await desktop.oauthLogoutConnectionConfig?.(url)

  if (!(await ensureCloudPortalSession(desktop))) {
    return 'portal-incomplete'
  }

  return (await cascadeCloudAgentSession(desktop, url)) ? 'connected' : 'portal-incomplete'
}
