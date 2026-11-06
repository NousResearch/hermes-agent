/**
 * Per-profile worker liveness for the fleet rail.
 *
 * A bot can be mid-card with no chat traffic: kanban/tool worker sessions are
 * deny-listed out of every conversation listing, so no gateway socket streams
 * anything the rail could pulse on. The gateway's own session store still
 * heartbeats `last_activity_at` on those rows, and
 * `/api/profiles/sessions?sources=kanban,tool,oneshot` returns them per
 * profile. This module turns that page into the rail's "someone is working"
 * truth without a new backend surface.
 *
 * Freshness is derived from `last_activity_at`, never from the fetch time: a
 * roster poll that stops updating cannot keep a dot green past the heartbeat
 * window, and the renderer marks the claim hollow when the poll itself is old.
 */

/** Same window the Bot Mode roster uses for its worker pulse. The activity
 *  heartbeat refreshes `last_activity_at` every ~60s while a worker runs. */
export const FLEET_WORKER_ACTIVE_WINDOW_S = 150

/** Two missed 60s roster refreshes: past this the renderer paints the dot
 *  hollow instead of confidently green. */
export const FLEET_WORK_STALE_MS = 120_000

export const FLEET_WORKER_SOURCES = ['kanban', 'tool', 'oneshot'] as const

/** One bounded page, freshest first, scoped to worker sources only. The
 *  endpoint fans out across every profile's state.db, so 50 rows is enough to
 *  find the newest worker session per profile without paying a full scan. */
export const FLEET_WORKER_SESSIONS_PATH =
  '/api/profiles/sessions?limit=50&offset=0&min_messages=0&order=recent&archived=exclude' +
  `&sources=${FLEET_WORKER_SOURCES.join(',')}`

export interface FleetWorkerSession {
  sessionId: string
  source: string
  title?: string
  /** Epoch seconds, straight from the session row's `last_activity_at`. */
  lastActive: number
}

/** Newest still-open worker session per profile. Rows whose heartbeat is older
 *  than the liveness window are dropped rather than carried as "recently". */
export function fleetWorkerSessionsFromBody(body: unknown, nowSeconds: number): Record<string, FleetWorkerSession> {
  const rows = (body as { sessions?: unknown } | null)?.sessions

  if (!Array.isArray(rows)) {
    return {}
  }

  const newest = new Map<string, FleetWorkerSession>()

  for (const raw of rows) {
    const row = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : null

    if (!row) {
      continue
    }

    const profile = typeof row.profile === 'string' ? row.profile.trim() : ''
    const sessionId = typeof row.id === 'string' ? row.id.trim() : ''
    const lastActive = Number(row.last_active)

    if (!profile || !sessionId || !Number.isFinite(lastActive)) {
      continue
    }

    // A finished worker row can keep a fresh last_active; only an open session
    // is still working.
    if (row.ended_at) {
      continue
    }

    if (nowSeconds - lastActive >= FLEET_WORKER_ACTIVE_WINDOW_S) {
      continue
    }

    const current = newest.get(profile)

    if (current && current.lastActive >= lastActive) {
      continue
    }

    newest.set(profile, {
      sessionId,
      source: typeof row.source === 'string' ? row.source : '',
      ...(typeof row.title === 'string' && row.title.trim() ? { title: row.title.trim() } : {}),
      lastActive
    })
  }

  return Object.fromEntries([...newest.entries()].sort(([a], [b]) => a.localeCompare(b)))
}
