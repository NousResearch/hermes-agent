import type { DesktopCloudAgentSignInResult } from '@/global'

/**
 * Re-establish a Hermes Cloud agent's gateway session from the shared desktop
 * portal session: drop this gateway's lapsed bearer, make sure the portal
 * session is live (a system-browser sign-in when it has lapsed), then run the
 * silent per-agent token exchange so a fresh agent bearer is stored for `url`.
 * The exchange audience comes only from portal discovery (main process); the
 * renderer never supplies an agent id here.
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
    // `cancelled` = the user denied in the browser or pressed Cancel sign-in.
    login: () => Promise<{ ok: boolean; signedIn: boolean; cancelled?: boolean }>
    agentSignIn: (dashboardUrl: string) => Promise<DesktopCloudAgentSignInResult>
  }
  oauthLogoutConnectionConfig?: (url: string) => Promise<unknown>
}

export interface CloudAgentSessionOptions {
  /**
   * Called with `true` when a browser sign-in starts and `false` when it ends,
   * so the caller can offer "Copy link" / "Cancel sign-in" while it is pending.
   */
  onBrowserSignIn?: (pending: boolean) => void
}

/**
 * `'connected'`: a fresh agent bearer was stored. `'portal-incomplete'`: the
 * browser sign-in (or the exchange) did not complete. `'cancelled'`: the user
 * backed out of the browser sign-in — their choice, not a failure.
 */
export type CloudAgentSessionOutcome = 'cancelled' | 'connected' | 'portal-incomplete'

/**
 * One silent exchange attempt against `url`. Assumes the portal session is
 * live (call `ensureCloudPortalSession` first); returns whether an agent
 * bearer was stored.
 */
async function exchangeCloudAgentSession(desktop: CloudAgentSessionBridge, url: string): Promise<boolean> {
  const result = await desktop.cloud!.agentSignIn(url)

  return result.connected === true
}

/** The portal session state, signing in (in the system browser) when it is not live. */
async function ensureCloudPortalSession(
  desktop: CloudAgentSessionBridge,
  options: CloudAgentSessionOptions
): Promise<CloudAgentSessionOutcome> {
  const status = await desktop.cloud!.status()

  if (status.signedIn) {
    return 'connected'
  }

  options.onBrowserSignIn?.(true)

  try {
    const login = await desktop.cloud!.login()

    if (login.cancelled) {
      return 'cancelled'
    }

    return login.signedIn ? 'connected' : 'portal-incomplete'
  } finally {
    options.onBrowserSignIn?.(false)
  }
}

/**
 * Full recovery ladder for one cloud agent's gateway session. Returns the
 * outcome, or throws the underlying error (network, portal outage, exchange
 * failure) for the caller to surface the way it already surfaces failures.
 */
export async function reestablishCloudAgentSession(
  desktop: CloudAgentSessionBridge,
  url: string,
  options: CloudAgentSessionOptions = {}
): Promise<CloudAgentSessionOutcome> {
  // Drop this gateway's lapsed bearer first — a rejected token set must not
  // be mistaken for a live session by the exchange or any liveness probe.
  await desktop.oauthLogoutConnectionConfig?.(url)

  const portal = await ensureCloudPortalSession(desktop, options)

  if (portal !== 'connected') {
    return portal
  }

  return (await exchangeCloudAgentSession(desktop, url)) ? 'connected' : 'portal-incomplete'
}
