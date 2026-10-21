/**
 * Backend identity for the unified session list (RT-42).
 *
 * Session totals are counted once per BACKEND, keyed on the gateway `install_id` that
 * `/api/status` reports (`hermes_cli/install_identity.py`). A sidebar refresh never waits on an
 * identity probe: an unanswered probe reads as "unknown", which only makes the totals take
 * max() instead of a sum.
 */

import { connectionInstallIds } from './connection-caches'

const IDENTITY_PROBE_BUDGET_MS = 2_000

// Per-connection identity is probed on the ~5s Bot Mode roster poll, which only hits /api/profiles,
// so the status probe is TTL-cached per connection to avoid doubling roster traffic; the Test button
// refreshes it eagerly. A missing id simply bypasses the same-backend roster collapse and makes
// session totals take max() instead of a sum — fully backward compatible.
export const INSTALL_ID_TTL_MS = 5 * 60_000
export const INSTALL_ID_NEGATIVE_TTL_MS = 60_000

/** Settle an identity probe within the sidebar's budget; a slow or failed probe is unknown. */
export async function boundedInstallId(
  probe: Promise<string | undefined>,
  budgetMs = IDENTITY_PROBE_BUDGET_MS
): Promise<string | undefined> {
  let timer: ReturnType<typeof setTimeout> | undefined

  try {
    return await Promise.race([
      probe,
      new Promise<undefined>(resolve => {
        timer = setTimeout(() => resolve(undefined), budgetMs)
      })
    ])
  } catch {
    return undefined
  } finally {
    clearTimeout(timer)
  }
}

/** The non-empty `install_id` a `/api/status` body reports, else undefined. */
export function installIdFromStatus(body: unknown): string | undefined {
  const raw = body && typeof body === 'object' ? (body as { install_id?: unknown }).install_id : undefined

  return typeof raw === 'string' && raw.trim() ? raw.trim() : undefined
}

export interface PrimaryInstallIdDeps {
  /** The primary backend's descriptor, without dialing or spawning anything. */
  resolve: () => Promise<unknown>
  /** GET /api/status against a descriptor. */
  status: (descriptor: unknown) => Promise<unknown>
  ttlMs: number
  negativeTtlMs: number
  now?: () => number
  budgetMs?: number
}

/**
 * Resolve the install_id of the backend the primary session aggregate reads (local or global
 * remote). Cached per resolved base URL, so switching the primary never reuses the previous
 * backend's identity; a miss is cached for the shorter negative TTL.
 */
export function createPrimaryInstallIdResolver(deps: PrimaryInstallIdDeps): () => Promise<string | undefined> {
  const now = deps.now ?? Date.now
  let cached: { id?: string; key: string; ts: number } | null = null

  return async () => {
    try {
      const descriptor = await deps.resolve()
      const key = String((descriptor as { baseUrl?: unknown } | null)?.baseUrl || '')

      if (!key) {
        return undefined
      }

      if (cached && cached.key === key && now() - cached.ts < (cached.id ? deps.ttlMs : deps.negativeTtlMs)) {
        return cached.id
      }

      const id = await boundedInstallId(
        Promise.resolve(deps.status(descriptor)).then(installIdFromStatus),
        deps.budgetMs
      )

      cached = { id, key, ts: now() }

      return id
    } catch {
      return undefined
    }
  }
}

/** Record the install_id a connection's `/api/status` body reports (or its absence). */
export function rememberConnectionInstallId(connectionId: string, statusBody: unknown): string | undefined {
  const id = installIdFromStatus(statusBody)
  connectionInstallIds.set(connectionId, { id, ts: Date.now() })

  return id
}

/** A connection's install_id, TTL-cached; `fetchStatus` runs only on a cache miss. */
export async function probeCachedConnectionInstallId(
  connectionId: string,
  fetchStatus: () => Promise<unknown>
): Promise<string | undefined> {
  const cached = connectionInstallIds.get(connectionId)

  if (cached && Date.now() - cached.ts < (cached.id ? INSTALL_ID_TTL_MS : INSTALL_ID_NEGATIVE_TTL_MS)) {
    return cached.id
  }

  try {
    return rememberConnectionInstallId(connectionId, await fetchStatus())
  } catch {
    // Keep any previously-known id (identity is stable; a transient fetch failure must not flap
    // the roster collapse), but do not cache a MISS over it.
    if (cached?.id) {
      return cached.id
    }

    connectionInstallIds.set(connectionId, { id: undefined, ts: Date.now() })

    return undefined
  }
}
