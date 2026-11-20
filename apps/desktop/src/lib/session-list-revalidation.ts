/**
 * Validator store for the desktop's conditional session-list reads (#C3
 * desktop half, t_5c4b52bf) — the IPC-bridge mirror of web's
 * `api-conditional.ts`.
 *
 * A reconnect refreshes the session list even when nothing moved. The server
 * stamps a content ETag on GET /api/sessions; the `hermes:api` bridge carries
 * it back in a branded envelope (electron/api-conditional.ts). This module
 * keeps validator + derived value beside each (descriptor, path) in a JS-side
 * Map — deliberately not any HTTP cache — and re-serves the cached value on a
 * 304: zero JSON on the wire, zero re-parse, and the SAME object reference,
 * so downstream stores keep their no-op referential identity.
 *
 * The descriptor is the backend identity the request rode
 * (connection, profile) — one validator must never be sent to another
 * backend, so the key declares its scope. A main process that predates the
 * seam answers the plain body (no brand): the read is used as-is and NO
 * validator is stored — today's behavior, exactly (the capability gate).
 */

import { isConditionalEnvelope } from '../../electron/api-conditional'

export interface RevalidationRead<T> {
  /** True when the server confirmed 304 and the cached value was served. */
  notModified: boolean
  /** The cached value (only when `notModified`). */
  value?: T
  /** The fresh parsed body (only when modified / unconditional). */
  raw?: unknown
  /** The validator the fresh body stamped (null: server stamps none). */
  etag?: string | null
}

export interface SessionListRevalidation {
  /** Revalidate one list read under `key`; 304 serves the stored value. */
  read<T>(key: string, fetch: (ifNoneMatch?: string) => Promise<unknown>): Promise<RevalidationRead<T>>
  /** Store the derived value under the fresh body's validator. */
  remember(key: string, etag: string, value: unknown): void
  /** Drop every stored validator (tests / connection re-point). */
  reset(): void
}

/** Bounded because a long-lived window can cycle (connection, profile, query). */
const MAX_ENTRIES = 32

export function createSessionListRevalidation(): SessionListRevalidation {
  const entries = new Map<string, { etag: string; value: unknown }>()

  const touch = (key: string) => {
    const entry = entries.get(key)

    if (entry) {
      entries.delete(key)
      entries.set(key, entry)
    }
  }

  return {
    async read<T>(key: string, fetch: (ifNoneMatch?: string) => Promise<unknown>): Promise<RevalidationRead<T>> {
      const cached = entries.get(key)
      const first = await fetch(cached?.etag)

      if (!isConditionalEnvelope(first)) {
        // Older main process (or a non-list read): plain body, no validator.
        return { notModified: false, raw: first, etag: null }
      }

      if (first.status === 304) {
        if (cached) {
          // The server confirmed the stored rows still stand: nothing
          // transferred, nothing parsed, same reference out.
          touch(key)

          return { notModified: true, value: cached.value as T }
        }

        // A 304 we did not ask for: untrustworthy — re-read once
        // unconditionally rather than serve nothing.
        const retry = await fetch(undefined)

        return isConditionalEnvelope(retry)
          ? { notModified: false, raw: retry.status === 304 ? null : retry.data, etag: retry.etag }
          : { notModified: false, raw: retry, etag: null }
      }

      return { notModified: false, raw: first.data, etag: first.etag }
    },

    remember(key: string, etag: string, value: unknown) {
      entries.delete(key)
      entries.set(key, { etag, value })

      while (entries.size > MAX_ENTRIES) {
        const oldest = entries.keys().next().value

        if (oldest === undefined) {
          break
        }

        entries.delete(oldest)
      }
    },

    reset() {
      entries.clear()
    }
  }
}
