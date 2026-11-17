import { JsonRpcGatewayError } from '@hermes/shared'

import type { GatewayRequest } from './agent-plugins'

type AppBlockerReason = 'missing_app' | 'version_too_old' | 'unsupported_os' | 'unsupported_gpu'
export type AppPresenceReason = AppBlockerReason | 'available' | 'app_not_running' | 'no_requirements' | 'uncertain'

export interface AppGuidance {
  reason: AppPresenceReason
  app_title: string
  required_version?: string | null
  found_version?: string | null
  download_url?: string | null
}

const BLOCKERS: ReadonlySet<AppPresenceReason> = new Set<AppBlockerReason>([
  'missing_app',
  'version_too_old',
  'unsupported_os',
  'unsupported_gpu'
])

/** The two reasons a vendor download fixes; the backend only sends a link for these. */
const LINKED: ReadonlySet<AppPresenceReason> = new Set<AppBlockerReason>(['missing_app', 'version_too_old'])

const REASONS = new Set<AppPresenceReason>([
  'missing_app',
  'version_too_old',
  'unsupported_os',
  'unsupported_gpu',
  'available',
  'app_not_running',
  'no_requirements',
  'uncertain'
])

function parseAppGuidance(value: unknown): AppGuidance | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const data = value as Record<string, unknown>

  if (
    typeof data.app_title !== 'string' ||
    !data.app_title.trim() ||
    typeof data.reason !== 'string' ||
    !REASONS.has(data.reason as AppPresenceReason)
  ) {
    return null
  }

  return {
    reason: data.reason as AppPresenceReason,
    app_title: data.app_title,
    required_version: typeof data.required_version === 'string' ? data.required_version : null,
    found_version: typeof data.found_version === 'string' ? data.found_version : null,
    download_url: typeof data.download_url === 'string' ? data.download_url : null
  }
}

/** A presence read that takes longer answers "uncertain": Install stays possible and the backend's
 *  install gate decides, instead of a stalled read keeping Install disabled until the RPC default. */
export const PRESENCE_TIMEOUT_MS = 30_000

/** Only catalog picks use backend app discovery. Null means an older backend has no preflight. */
export async function presenceForCatalogPlugin(
  request: GatewayRequest,
  name: string,
  profile: string
): Promise<AppGuidance | null> {
  try {
    const result = await request<{ presence?: unknown }>(
      'plugins.manage',
      {
        action: 'presence',
        names: [name],
        profile
      },
      PRESENCE_TIMEOUT_MS
    )

    if (!Array.isArray(result?.presence)) {
      return null
    }

    const row = result.presence.find(candidate => candidate?.name === name)

    return parseAppGuidance(row) ?? { reason: 'uncertain', app_title: name }
  } catch (error) {
    if (
      error instanceof JsonRpcGatewayError &&
      (error.code === -32601 || error.code === 4019 || error.code === -32602)
    ) {
      return null
    }

    // A failed read cannot assert a missing application; offer an honest uncertain state.
    return { reason: 'uncertain', app_title: name }
  }
}

/** An install race can re-introduce a typed refusal after a successful preflight. */
export function appGuidanceFromError(error: unknown): AppGuidance | null {
  return error instanceof JsonRpcGatewayError && error.code === 5026 ? parseAppGuidance(error.data) : null
}

/** Only backend-vetted catalog metadata reaches this field, not a plugin's docs_url or manifest URL. */
export function officialAppDownloadUrl(guidance: AppGuidance): string | null {
  const value = guidance.download_url

  if (
    !LINKED.has(guidance.reason) ||
    !value ||
    Array.from(value).some(char => /\s/.test(char) || char.charCodeAt(0) < 32) ||
    // The backend refuses any explicit port; the URL parser would silently drop a default ":443".
    /^https:\/\/[^/?#@]*:/i.test(value)
  ) {
    return null
  }

  try {
    const url = new URL(value)

    if (
      url.protocol !== 'https:' ||
      url.username ||
      url.password ||
      url.port ||
      url.hash ||
      !/^[a-z0-9-]+(?:\.[a-z0-9-]+)+$/i.test(url.hostname) ||
      // WHATWG already rewrote numeric hosts ("127.1") to a dotted quad; special-use names are never a vendor.
      /^\d+(?:\.\d+){3}$/.test(url.hostname) ||
      /(?:^|\.)(?:localhost|local|internal|home\.arpa)$/i.test(url.hostname)
    ) {
      return null
    }

    return value
  } catch {
    return null
  }
}

export function isAppBlocked(guidance: AppGuidance | null): boolean {
  return (
    guidance !== null &&
    (BLOCKERS.has(guidance.reason) ||
      (guidance.reason === 'uncertain' && Boolean(guidance.required_version) && !guidance.found_version))
  )
}
