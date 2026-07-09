/**
 * Zero-touch provisioning client — the desktop side of the admin portal's
 * GET /api/portal/desktop/provision.
 *
 * After a successful OTP login (and on boot with a restored session) the
 * main process calls the endpoint THROUGH the persist:ix-agency-portal
 * session's fetch (the httpOnly OTP cookie lives only in that jar) and
 * fills in whatever the employee has not configured yet: the admin-mcp
 * gateway token, the LiteLLM key, the Cognito S2S secret, and the
 * WireGuard usa-vpn.conf (keychain import — the existing VPN auto-connect
 * then brings the tunnel up on its own).
 *
 * Manual values always win: a slot is only filled when it is empty, so
 * re-provisioning is idempotent and never clobbers a hand-entered override.
 *
 * Pure logic with an injected fetch — see ix-auto-provision.test.ts.
 */
import { DEFAULT_IX_AGENCY_SETTINGS, type IxAgencySettings } from './ix-agency'
import { looksLikeWireGuardConf } from './ix-status'

export interface IxDesktopProvisionPayload {
  gatewayToken: string
  litellm: { url: string; key: string; source: string; note?: string }
  cognito: { clientId: string; clientSecret: string; note?: string }
  wireguard: { conf: string; source: string; note?: string }
}

/** GET the provisioning payload with the signed-in portal session's cookies. */
export async function fetchIxDesktopProvision(
  portalUrl: string,
  fetchImpl: typeof fetch = fetch
): Promise<IxDesktopProvisionPayload> {
  const res = await fetchImpl(new URL('/api/portal/desktop/provision', portalUrl).toString(), {
    method: 'GET',
    credentials: 'include',
    signal: AbortSignal.timeout(30_000)
  })

  if (res.status === 401 || res.status === 403) {
    throw new Error('Portal session is not active — sign in first.')
  }

  if (!res.ok) {
    throw new Error(`Provisioning endpoint returned HTTP ${res.status}.`)
  }

  const body = (await res.json()) as Partial<IxDesktopProvisionPayload> & { ok?: boolean }

  if (!body?.ok) {
    throw new Error('The portal did not return a provisioning payload.')
  }

  return {
    gatewayToken: String(body.gatewayToken ?? ''),
    litellm: {
      url: String(body.litellm?.url ?? ''),
      key: String(body.litellm?.key ?? ''),
      source: String(body.litellm?.source ?? ''),
      ...(body.litellm?.note ? { note: String(body.litellm.note) } : {})
    },
    cognito: {
      clientId: String(body.cognito?.clientId ?? ''),
      clientSecret: String(body.cognito?.clientSecret ?? ''),
      ...(body.cognito?.note ? { note: String(body.cognito.note) } : {})
    },
    wireguard: {
      conf: String(body.wireguard?.conf ?? ''),
      source: String(body.wireguard?.source ?? ''),
      ...(body.wireguard?.note ? { note: String(body.wireguard.note) } : {})
    }
  }
}

export interface IxProvisionApplyResult {
  next: IxAgencySettings
  /** Settings keys that were auto-filled this pass (empty = nothing to do). */
  filled: string[]
}

/**
 * Fill ONLY the empty slots from the portal payload. A slot the user already
 * configured (manually or by a previous provisioning pass) is left alone, so
 * the operation is idempotent and manual overrides survive.
 */
export function applyIxProvisionToSettings(
  current: IxAgencySettings,
  payload: IxDesktopProvisionPayload
): IxProvisionApplyResult {
  const next = { ...current }
  const filled: string[] = []

  if (!current.gatewayToken && payload.gatewayToken) {
    next.gatewayToken = payload.gatewayToken
    filled.push('gatewayToken')
  }

  if (!current.litellmKey && payload.litellm.key) {
    next.litellmKey = payload.litellm.key
    filled.push('litellmKey')

    // Follow the portal's LiteLLM base URL only while ours is the stock
    // default — a custom URL is a manual override and stays.
    if (payload.litellm.url && current.litellmUrl === DEFAULT_IX_AGENCY_SETTINGS.litellmUrl) {
      next.litellmUrl = payload.litellm.url
    }
  }

  if (!current.cognitoClientSecret && payload.cognito.clientSecret) {
    next.cognitoClientSecret = payload.cognito.clientSecret
    filled.push('cognitoClientSecret')

    // Keep the client id paired with the secret it belongs to (only while
    // ours is still the stock default).
    if (payload.cognito.clientId && current.cognitoClientId === DEFAULT_IX_AGENCY_SETTINGS.cognitoClientId) {
      next.cognitoClientId = payload.cognito.clientId
    }
  }

  // VPN: import into the keychain slot only when NO profile exists at all —
  // neither an imported conf nor a manually-configured .conf path.
  if (
    !current.vpnConfSecret &&
    !current.vpnConfPath &&
    payload.wireguard.conf &&
    looksLikeWireGuardConf(payload.wireguard.conf)
  ) {
    next.vpnConfSecret = payload.wireguard.conf
    filled.push('vpnConfSecret')
  }

  return { next, filled }
}
