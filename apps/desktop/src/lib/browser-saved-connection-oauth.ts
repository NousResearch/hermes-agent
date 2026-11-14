import { atom } from 'nanostores'

import type { DesktopOauthLoginOptions, DesktopOauthLoginResult, HermesApiRequest } from '@/global'
import type { BrowserSavedConnections } from '@/lib/browser-saved-connections'

/**
 * Signing in to an oauth-mode saved gateway from the Webapp. The serving gateway runs the native
 * PKCE flow itself (`hermes_cli/gateway_connections_oauth.py`) and keeps the tokens; this tab only
 * opens the other gateway's sign-in page. Its redirect comes back to the serving machine's
 * loopback, which a browser on another machine cannot reach: that browser lands on its own
 * loopback, and the address it shows is pasted here (`$webappOauthPending`) to finish.
 */

type Api = <T>(request: HermesApiRequest) => Promise<T>

export interface WebappOauthPending {
  connectionId: string
  /** Where the other gateway sends the browser once signed in: the serving machine's loopback. */
  redirectUri: string
}

/** The sign-in this tab is waiting on, for the field that takes a pasted redirect address. */
export const $webappOauthPending = atom<null | WebappOauthPending>(null)

const POLL_MS = 1_500
const route = (connectionId: string) => `/api/gateway-connections/${encodeURIComponent(connectionId)}/oauth`

const sameUrl = (a?: string, b?: string) => Boolean(a && b) && a!.replace(/\/+$/, '') === b!.replace(/\/+$/, '')

/** Finish the pending sign-in from the address the browser landed on. */
export async function completeWebappOauth(api: Api, connectionId: string, url: string): Promise<void> {
  await api({ body: { url }, method: 'POST', path: `${route(connectionId)}/complete` })
}

async function savedConnectionFor(
  saved: BrowserSavedConnections,
  remoteUrl: string,
  identity?: DesktopOauthLoginOptions
): Promise<string> {
  const rows = await saved.gateways()
  const known = rows.find(row =>
    identity?.connectionId ? row.id === identity.connectionId : sameUrl(row.url, remoteUrl)
  )

  if (known && known.authMode === 'oauth' && sameUrl(known.url, remoteUrl)) {
    return known.id
  }

  // The serving gateway signs in for a saved connection, so an unsaved draft (or one switched to
  // sign-in) is saved as oauth first; its Save then edits the same id, as on native Desktop.
  const label = identity?.label?.trim() || known?.label

  if (!label) {
    throw new Error('Name this connection, then sign in.')
  }

  const result = await saved.connections.save({
    ...(known ? { id: known.id } : {}),
    authMode: 'oauth',
    kind: identity?.kind === 'cloud' ? 'cloud' : 'remote',
    label,
    url: remoteUrl
  })

  return result.connection.id
}

async function waitForSignIn(api: Api, connectionId: string, expiresAt: number): Promise<boolean> {
  while (Date.now() / 1000 < expiresAt) {
    const status = await api<{ pending?: boolean; signedIn?: boolean }>({ path: route(connectionId) })

    if (status?.signedIn) {
      return true
    }

    // A completed or failed flight is no longer pending; a pasted address that failed says why itself.
    if (status && status.pending === false) {
      return false
    }

    await new Promise(resolve => window.setTimeout(resolve, POLL_MS))
  }

  return false
}

export async function signInSavedConnection(
  api: Api,
  saved: BrowserSavedConnections,
  remoteUrl: string,
  identity?: DesktopOauthLoginOptions
): Promise<DesktopOauthLoginResult> {
  const connectionId = await savedConnectionFor(saved, remoteUrl, identity)

  const started = await api<{ authorizeUrl: string; expiresAt: number; redirectUri: string }>({
    body: {},
    method: 'POST',
    path: `${route(connectionId)}/start`
  })

  window.open(started.authorizeUrl, '_blank', 'noopener,noreferrer')
  $webappOauthPending.set({ connectionId, redirectUri: started.redirectUri })

  try {
    const connected = await waitForSignIn(api, connectionId, started.expiresAt)

    return {
      baseUrl: remoteUrl,
      connected,
      connectionId,
      ...(connected ? {} : { error: 'The sign-in did not finish' }),
      ok: connected
    }
  } finally {
    if ($webappOauthPending.get()?.connectionId === connectionId) {
      $webappOauthPending.set(null)
    }
  }
}

export async function signOutSavedConnection(
  api: Api,
  saved: BrowserSavedConnections,
  remoteUrl: string
): Promise<{ connected: false; ok: boolean; error?: string }> {
  const row = (await saved.gateways()).find(entry => sameUrl(entry.url, remoteUrl) && entry.authMode === 'oauth')

  if (!row) {
    return { connected: false, error: 'No saved connection signs in at that address.', ok: false }
  }

  await api({ method: 'DELETE', path: route(row.id) })

  return { connected: false, ok: true }
}
