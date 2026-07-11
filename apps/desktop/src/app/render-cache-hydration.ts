/**
 * Render-cache hydration + write-through (Phase 2 of the startup-latency work).
 *
 * Boot path: read the cache (one IPC hop), seed the session-list/status stores
 * so the sidebar paints last-known-good rows instead of skeletons, then let the
 * normal live boot reconcile wholesale (the server copy wins, I1). The cached
 * paint is read-only by construction: the composer is gateway-gated (disabled
 * until the WS handshake), and transcript deltas are keyed by RUNTIME session
 * id, which doesn't exist until resume — so no live delta can be applied to a
 * hydrated store before the first wholesale snapshot (I5's ordering invariant,
 * proven in render-cache-hydration.test.ts).
 *
 * Write path: after every live session-list/status settle, push the fresh copy
 * to the main-process writer (debounced there). The transcript push is wired
 * where the wholesale snapshot lands (use-session-state-cache flush).
 *
 * Everything here is fail-open: cache errors degrade to today's boot (I3).
 */

import type { SessionInfo } from '@/types/hermes'

export interface CachedSessionList {
  sessions: SessionInfo[]
  total: number
}

/** The renderCache preload surface (optional — older mains lack it). */
function renderCacheApi() {
  return (window as any).hermesDesktop?.renderCache ?? null
}

export interface HydrationResult {
  /** True when cached sessions were painted. */
  painted: boolean
  /** True when a cached transcript was painted for the remembered session. */
  transcriptPainted: boolean
  gatewayUrl: string | null
  cachedSessionIds: string[]
}

/**
 * Read the render cache and seed the stores via the provided setters. Only
 * hydrates when the live list hasn't loaded yet (sessions store still empty) —
 * a second call (e.g. re-mount) must never clobber live data with stale cache.
 */
export async function hydrateFromRenderCache(deps: {
  getSessions: () => SessionInfo[]
  setSessions: (rows: SessionInfo[]) => void
  setSessionsTotal: (total: number) => void
  rememberedSessionId?: string | null
  /** Optional transcript paint: only used when the messages store is empty. */
  getMessages?: () => unknown[]
  setMessages?: (rows: unknown[]) => void
}): Promise<HydrationResult> {
  const none: HydrationResult = { painted: false, transcriptPainted: false, gatewayUrl: null, cachedSessionIds: [] }
  try {
    const api = renderCacheApi()
    if (!api) {
      return none
    }
    const result = await api.read(null, deps.rememberedSessionId ?? null)
    if (!result?.enabled) {
      return { ...none, gatewayUrl: result?.gatewayUrl ?? null }
    }

    // Transcript paint (chat pane): only into an EMPTY messages store, and only
    // for the session the app is about to restore. The first live wholesale
    // flush replaces these rows entirely (I5 — see use-session-state-cache's
    // thread-switch branch, which publishes pending.state.messages unmerged).
    let transcriptPainted = false
    const cachedTranscript = result.transcript as { rows?: unknown[] } | null
    if (
      deps.setMessages &&
      deps.getMessages &&
      deps.rememberedSessionId &&
      Array.isArray(cachedTranscript?.rows) &&
      cachedTranscript!.rows.length > 0 &&
      deps.getMessages().length === 0
    ) {
      deps.setMessages(cachedTranscript!.rows)
      transcriptPainted = true
    }

    const cached = result.sessions as CachedSessionList | null
    const rows = Array.isArray(cached?.sessions) ? cached!.sessions : null
    if (!rows || rows.length === 0) {
      return { ...none, transcriptPainted, gatewayUrl: result.gatewayUrl }
    }
    // Never clobber live data: hydrate only into an EMPTY list.
    if (deps.getSessions().length > 0) {
      return { ...none, transcriptPainted, gatewayUrl: result.gatewayUrl }
    }
    deps.setSessions(rows)
    deps.setSessionsTotal(Number.isFinite(cached!.total) ? cached!.total : rows.length)
    return {
      painted: true,
      transcriptPainted,
      gatewayUrl: result.gatewayUrl,
      cachedSessionIds: rows.map(r => r.id)
    }
  } catch {
    return none
  }
}

/**
 * Rows that differ between the cached paint and the first live list —
 * the Phase-0 RC4 divergence signal. Counts adds + removals + field drift on
 * the identity fields the sidebar renders (title/pinned/archived).
 */
export function computeListDivergence(cached: SessionInfo[], live: SessionInfo[]): number {
  const liveById = new Map(live.map(s => [s.id, s]))
  let divergent = 0
  for (const c of cached) {
    const l = liveById.get(c.id)
    if (!l) {
      divergent += 1 // cached row no longer live
      continue
    }
    if ((c.title ?? '') !== (l.title ?? '') || !!c.pinned !== !!l.pinned || !!c.archived !== !!l.archived) {
      divergent += 1
    }
    liveById.delete(c.id)
  }
  divergent += liveById.size // live rows the cache didn't know about
  return divergent
}

/**
 * Post-reconcile hook: report divergence, push the fresh live list into the
 * cache, and run the boot sweep (I4b) exactly once per launch.
 */
export function reconcileRenderCache(opts: {
  gatewayUrl: string | null
  cachedSessionIds: string[]
  cachedSessions: SessionInfo[] | null
  liveSessions: SessionInfo[]
  liveTotal: number
  sweptRef: { current: boolean }
}): void {
  try {
    const api = renderCacheApi()
    if (!api || !opts.gatewayUrl) {
      return
    }
    if (opts.cachedSessions) {
      api.reportDivergence(computeListDivergence(opts.cachedSessions, opts.liveSessions))
    }
    api.putSessions(opts.gatewayUrl, { sessions: opts.liveSessions, total: opts.liveTotal })
    if (!opts.sweptRef.current) {
      opts.sweptRef.current = true
      api.sweep(
        opts.gatewayUrl,
        opts.liveSessions.map(s => s.id)
      )
    }
  } catch {
    // fail-open
  }
}

/** Push a live status snapshot into the cache (fire-and-forget). */
export function pushStatusToRenderCache(gatewayUrl: string | null, status: unknown): void {
  try {
    const api = renderCacheApi()
    if (api && gatewayUrl) {
      api.putStatus(gatewayUrl, status)
    }
  } catch {
    // fail-open
  }
}

/** Push the active session's transcript tail (fire-and-forget). */
export function pushTranscriptToRenderCache(
  gatewayUrl: string | null,
  storedSessionId: string | null,
  rows: unknown[]
): void {
  try {
    const api = renderCacheApi()
    if (api && gatewayUrl && storedSessionId && Array.isArray(rows) && rows.length > 0) {
      api.putTranscript(gatewayUrl, storedSessionId, rows)
    }
  } catch {
    // fail-open
  }
}

/** Forward a session delete to the cache culler (the I4b delete wire). */
export function cullRenderCacheSession(gatewayUrl: string | null, storedSessionId: string | null): void {
  try {
    const api = renderCacheApi()
    if (api && gatewayUrl && storedSessionId) {
      api.cullSession(gatewayUrl, storedSessionId)
    }
  } catch {
    // fail-open
  }
}
