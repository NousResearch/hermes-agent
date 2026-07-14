// Persistent registry of the backend child PIDs the desktop app has spawned,
// so a LATER launch can reap orphans left behind by a NON-graceful quit of a
// PRIOR instance — force-quit (⌘Q without teardown), a crash, or a
// wake-from-sleep that severed the parent. On a clean quit the before-quit
// SIGTERM already reaps the children; the gap is only the ungraceful path,
// where the old backend keeps running and the next launch spawns ANOTHER one.
// Stacking backends is what produced #64304: three `hermes serve` / `hermes
// dashboard` processes writing the same session → duplicate sidebar entries,
// message loss, and role confusion.
//
// The desktop's single-instance lock (app.requestSingleInstanceLock) guarantees
// only one Hermes One runs at a time, so any PID recorded here that is still
// alive at the next startup is necessarily an orphan from a dead instance —
// safe to kill. We record ONLY the PIDs we spawn and reap ONLY those, so a
// user's own `hermes dashboard` in a terminal is never touched.
//
// These helpers are pure (no fs / child_process / Electron) so they unit-test
// in isolation; main.ts owns the impure read/write/verify/kill wiring.

export interface BackendRegistryEntry {
  pid: number
  command: string
  startedAt: number
}

/**
 * Parse on-disk registry text into a clean, de-duplicated entry list.
 *
 * Tolerant by design — a half-written file (killed mid-write) or hand-edited
 * garbage must never throw or block startup. Returns [] on any parse failure
 * and silently drops entries lacking a valid positive-integer `pid`. Accepts
 * either the wrapped `{backends:[...]}` shape we write or a bare array.
 */
export function parseRegistry(text: unknown): BackendRegistryEntry[] {
  let raw: unknown

  try {
    raw = JSON.parse(String(text ?? ''))
  } catch {
    return []
  }

  const list: unknown[] = Array.isArray(raw)
    ? raw
    : raw && Array.isArray((raw as { backends?: unknown[] }).backends)
      ? ((raw as { backends: unknown[] }).backends)
      : []

  const seen = new Set<number>()
  const out: BackendRegistryEntry[] = []

  for (const item of list) {
    if (!item || typeof item !== 'object') {continue}
    const rec = item as { pid?: unknown; command?: unknown; startedAt?: unknown }
    const pid = Number(rec.pid)

    if (!Number.isInteger(pid) || pid <= 0 || seen.has(pid)) {continue}
    seen.add(pid)
    out.push({
      pid,
      command: typeof rec.command === 'string' ? rec.command : '',
      startedAt: Number.isFinite(Number(rec.startedAt)) ? Number(rec.startedAt) : 0
    })
  }

  return out
}

/** Serialize entries to the canonical on-disk shape. */
export function stringifyRegistry(entries: BackendRegistryEntry[]): string {
  return JSON.stringify({ backends: Array.isArray(entries) ? entries : [] }, null, 2)
}

/**
 * Add or replace an entry by pid. Returns a NEW array (never mutates input).
 * A non-positive / non-integer pid is a no-op.
 */
export function upsertEntry(
  entries: BackendRegistryEntry[],
  entry: { pid: number; command?: string; startedAt?: number }
): BackendRegistryEntry[] {
  const base = Array.isArray(entries) ? entries : []
  const pid = Number(entry && entry.pid)

  if (!Number.isInteger(pid) || pid <= 0) {return base.slice()}
  const next = base.filter(e => e.pid !== pid)
  next.push({
    pid,
    command: entry && typeof entry.command === 'string' ? entry.command : '',
    startedAt: entry && Number.isFinite(Number(entry.startedAt)) ? Number(entry.startedAt) : 0
  })

  return next
}

/** Drop entries whose pid appears in `pids`. Returns a NEW array. */
export function removePids(entries: BackendRegistryEntry[], pids: number[]): BackendRegistryEntry[] {
  const base = Array.isArray(entries) ? entries : []
  const drop = new Set((Array.isArray(pids) ? pids : []).map(Number))

  return base.filter(e => !drop.has(e.pid))
}

/**
 * PIDs eligible for reaping at startup: every recorded pid except our own
 * process (defensive — we never record ourselves, but a recycled pid could
 * in principle collide). De-duplicated, order-preserving.
 */
export function reapablePids(entries: BackendRegistryEntry[], selfPid: number): number[] {
  const base = Array.isArray(entries) ? entries : []
  const self = Number(selfPid)
  const out: number[] = []
  const seen = new Set<number>()

  for (const e of base) {
    const pid = Number(e && e.pid)

    if (!Number.isInteger(pid) || pid <= 0) {continue}

    if (pid === self) {continue}

    if (seen.has(pid)) {continue}
    seen.add(pid)
    out.push(pid)
  }

  return out
}

/**
 * True when a process command line looks like a long-lived Hermes backend
 * (`hermes serve` / `hermes dashboard`, incl. the `python -m hermes_cli.main`
 * forms). Used to re-verify a recorded PID before killing it, so a recycled
 * PID now belonging to an unrelated process is never reaped. Mirrors the
 * command shapes in Python's `_find_stale_dashboard_pids`.
 */
export function backendCommandMatches(command: unknown): boolean {
  const s = String(command ?? '').toLowerCase()

  if (!s.includes('hermes')) {return false}

  return /(^|[\s/\\._])(serve|dashboard)(\s|$)/.test(s)
}
