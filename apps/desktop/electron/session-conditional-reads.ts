/**
 * Main-process revalidation state for the session-list reads the splice path
 * fans out (C3 desktop half, t_5c4b52bf).
 *
 * `fetchSessionRowsInPages` turns ONE logical list read into several paged
 * HTTP calls (the registry branch and the per-profile splice), so a validator
 * per logical read would miss: the unit of revalidation is the PAGE. This
 * cache keeps, per (descriptor, page path), the ETag + last parsed page, and
 * per (descriptor, base path, query) the derived join — so when every page of
 * a fan-out revalidates as 304 the join is served back unchanged (same rows
 * array reference: zero re-derivation, zero body parse) instead of being
 * recomputed from rows nothing changed.
 *
 * A server that stamps no ETag (older runtime, or a route without the
 * conditional response) yields `etag: null` on every read: nothing is ever
 * stored, nothing is ever sent, and behavior is exactly the unconditional
 * path. Bounded like every long-lived main-process map (LRU, oldest first).
 */

/** What a conditional page read resolves to (see api-json-fetch). */
export interface ConditionalPageResult {
  data: unknown
  status: number
  etag: string | null
}

/** The internal fetchJson conditional envelope, or a plain parsed body from
 *  an unconditional/legacy read. Plain bodies are usable but carry no
 *  validator: they are never stored. */
export function normalizeConditionalResult(value: unknown): ConditionalPageResult {
  if (
    value &&
    typeof value === 'object' &&
    'data' in value &&
    'status' in value &&
    'etag' in value
  ) {
    const v = value as Record<string, unknown>
    const status = v.status

    const etag: null | string | undefined =
      typeof v.etag === 'string' ? v.etag : v.etag === null ? null : undefined

    if (typeof status === 'number' && etag !== undefined) {
      return { data: v.data, status, etag }
    }
  }

  return { data: value, status: 200, etag: null }
}

export interface PageRead {
  data: unknown
  /** True when the server confirmed 304 and the cached page was served. */
  notModified: boolean
  /** The validator this read left stored (null: server stamps none). */
  etag: string | null
}

export interface ConditionalReadCache {
  /** Revalidate one page read; 304 serves the stored page without parsing. */
  readPage(
    descriptorKey: string,
    path: string,
    fetch: (ifNoneMatch?: string) => Promise<unknown>
  ): Promise<PageRead>
  derivedGet<T = unknown>(descriptorKey: string, basePath: string, query: string): T | undefined
  derivedSet<T = unknown>(descriptorKey: string, basePath: string, query: string, value: T): void
  reset(): void
}

interface PageEntry {
  etag: string
  data: unknown
}

const DEFAULT_MAX_PAGE_ENTRIES = 256
const DEFAULT_MAX_DERIVED_ENTRIES = 64

export function createConditionalReadCache(options: { maxPageEntries?: number; maxDerivedEntries?: number } = {}): ConditionalReadCache {
  const maxPageEntries = options.maxPageEntries ?? DEFAULT_MAX_PAGE_ENTRIES
  const maxDerivedEntries = options.maxDerivedEntries ?? DEFAULT_MAX_DERIVED_ENTRIES
  const pages = new Map<string, PageEntry>()
  const derived = new Map<string, unknown>()

  const pageKey = (descriptorKey: string, path: string) => `${descriptorKey}\u0000${path}`
  const derivedKey = (descriptorKey: string, basePath: string, query: string) => `${descriptorKey}\u0000${basePath}\u0000${query}`

  const touchPage = (key: string) => {
    const entry = pages.get(key)

    if (entry) {
      pages.delete(key)
      pages.set(key, entry)
    }
  }

  const rememberPage = (key: string, entry: PageEntry) => {
    pages.delete(key)
    pages.set(key, entry)

    while (pages.size > maxPageEntries) {
      const oldest = pages.keys().next().value

      if (oldest === undefined) {
        break
      }

      pages.delete(oldest)
    }
  }

  return {
    async readPage(descriptorKey, path, fetch) {
      const key = pageKey(descriptorKey, path)
      const cached = pages.get(key)
      let result = normalizeConditionalResult(await fetch(cached?.etag))

      if (result.status === 304) {
        if (cached) {
          // The server confirmed the stored page still stands. Serve it: no
          // body was transferred, nothing was parsed.
          touchPage(key)

          return { data: cached.data, notModified: true, etag: cached.etag }
        }

        // A 304 we did not ask for (no stored validator): untrustworthy here —
        // re-read once unconditionally rather than serve an empty page.
        result = normalizeConditionalResult(await fetch(undefined))
      }

      if (result.status === 304) {
        return { data: null, notModified: true, etag: result.etag }
      }

      if (result.etag) {
        rememberPage(key, { etag: result.etag, data: result.data })

        return { data: result.data, notModified: false, etag: result.etag }
      }

      // No validator stamped: drop any stale one and serve fresh.
      pages.delete(key)

      return { data: result.data, notModified: false, etag: null }
    },

    derivedGet: <T = unknown>(descriptorKey: string, basePath: string, query: string) =>
      derived.get(derivedKey(descriptorKey, basePath, query)) as T | undefined,

    derivedSet(descriptorKey, basePath, query, value) {
      const key = derivedKey(descriptorKey, basePath, query)

      derived.delete(key)
      derived.set(key, value)

      while (derived.size > maxDerivedEntries) {
        const oldest = derived.keys().next().value

        if (oldest === undefined) {
          break
        }

        derived.delete(oldest)
      }
    },

    reset() {
      pages.clear()
      derived.clear()
    }
  }
}
