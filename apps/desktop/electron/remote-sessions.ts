// Remote-profile session helpers for the desktop main process.
//
// Why this module exists: the sidebar's session list is the first thing the
// user sees, and it must never be held hostage by remote I/O. A profile that
// points at a remote backend (connection.json `profiles[name]`) participates
// in the unified session list, but a dead tunnel or sleeping host used to
// stall every sidebar refresh for the full local-boot readiness deadline
// (45s) — on boot AND on every subsequent refresh, because a failed pool
// entry is dropped and re-probed from scratch each time.
//
// The contract enforced here:
//   - Reachability probes against a REMOTE backend fail fast: short deadline,
//     short per-attempt timeout, and an immediate reject on "nothing is
//     listening" errors (ECONNREFUSED & co). The long deadline only makes
//     sense for freshly spawned LOCAL children, where the port stays refused
//     until uvicorn binds mid-boot.
//   - A remote that failed its probe enters a cooldown window. While in
//     cooldown the session splice skips it instantly instead of re-probing.
//   - The splice itself runs on a budget: a remote that cannot produce rows
//     within its budget contributes nothing to THIS refresh. Its fetch keeps
//     running in the background (warming the pool), so the next refresh picks
//     the rows up without paying the cold-start cost again.

// Reachability probe for an already-running remote backend. A healthy remote
// answers /api/status in tens of milliseconds; 3s absorbs a tailnet hiccup
// without making a human wait on a dead one.
const REMOTE_PROBE_DEADLINE_MS = 3_000
const REMOTE_PROBE_ATTEMPT_TIMEOUT_MS = 1_500
// How long a failed remote is skipped by the session splice before the next
// real probe. Long enough to stop per-refresh stalls, short enough that a
// revived tunnel reappears without user action.
const REMOTE_DOWN_COOLDOWN_MS = 30_000
// Sidebar wait ceilings for the remote splice. "Cold" = the profile's backend
// connection is not established yet (probe + fetch must both fit); "warm" =
// an established connection where only the list fetch remains. These are
// intentionally conservative for interactive sidebar refreshes: a slow WAN
// remote may miss one refresh cycle, but the in-flight fetch keeps warming the
// pool so a later refresh can pick it up without blocking the local list.
const COLD_REMOTE_SPLICE_BUDGET_MS = 2_000
const WARM_REMOTE_SPLICE_BUDGET_MS = 5_000

// Local readiness loop defaults (historic waitForHermes behavior).
const LOCAL_BOOT_DEADLINE_MS = 45_000
const READINESS_RETRY_INTERVAL_MS = 500

// Errors that mean "no server is listening at this address" rather than "the
// server exists but is still coming up". For a remote probe these are final:
// retrying cannot succeed until something rebinds the port, and that never
// happens within a 3s window. (EHOSTDOWN is the macOS cousin of EHOSTUNREACH.)
const NOTHING_LISTENING_CODES = new Set([
  'ECONNREFUSED',
  'EHOSTDOWN',
  'EHOSTUNREACH',
  'ENETDOWN',
  'ENETUNREACH',
  'ENOTFOUND'
])

const BUDGET_EXCEEDED = Symbol('remote-splice-budget-exceeded')

function isNothingListeningError(error) {
  return Boolean(error && NOTHING_LISTENING_CODES.has(error.code))
}

/**
 * Poll `probe()` until it resolves or the deadline passes. This is the core
 * of waitForHermes, extracted so the deadline/fail-fast behavior is unit
 * testable without a real HTTP server.
 *
 * Options:
 *   deadlineMs   — total wall-clock budget (default 45s: local child boot).
 *   intervalMs   — sleep between attempts (default 500ms).
 *   failFast     — reject on the FIRST "nothing is listening" error instead
 *                  of retrying. Correct for remotes (already supposed to be
 *                  up); wrong for local children (port is refused until the
 *                  server binds mid-boot).
 *   now / sleep  — injectable clocks for tests.
 */
async function waitForBackendReady(probe, options: any = {}) {
  const deadlineMs = Number.isFinite(options.deadlineMs) && options.deadlineMs > 0
    ? options.deadlineMs
    : LOCAL_BOOT_DEADLINE_MS
  const intervalMs = Number.isFinite(options.intervalMs) && options.intervalMs >= 0
    ? options.intervalMs
    : READINESS_RETRY_INTERVAL_MS
  const failFast = Boolean(options.failFast)
  const now = options.now || Date.now
  const sleep = options.sleep || (ms => new Promise(resolve => setTimeout(resolve, ms)))

  const deadline = now() + deadlineMs
  let lastError = null

  while (now() < deadline) {
    try {
      await probe()
      return
    } catch (error) {
      lastError = error
      if (failFast && isNothingListeningError(error)) {
        throw new Error(`Hermes backend is not listening: ${error.message || error.code}`)
      }
      await sleep(intervalMs)
    }
  }

  throw new Error(`Hermes backend did not become ready: ${lastError?.message || 'timeout'}`)
}

/**
 * Cooldown registry for unreachable remote backends, keyed by profile name.
 * Purely in-memory: a restart of the app retries every remote once.
 */
function createRemoteAvailability(options: any = {}) {
  const cooldownMs = Number.isFinite(options.cooldownMs) && options.cooldownMs > 0
    ? options.cooldownMs
    : REMOTE_DOWN_COOLDOWN_MS
  const now = options.now || Date.now
  const log = options.log || (() => {})
  const downUntil = new Map()

  const keyOf = profile => String(profile ?? '').trim()

  return {
    markDown(profile, reason = null) {
      const key = keyOf(profile)
      if (!key) return
      const until = now() + cooldownMs
      downUntil.set(key, { until, reason: reason || 'unreachable' })
      log(
        `[remote-sessions] remote profile "${key}" marked unreachable for ${Math.round(cooldownMs / 1000)}s` +
          (reason ? ` (${reason})` : '')
      )
    },
    markUp(profile) {
      const key = keyOf(profile)
      if (downUntil.delete(key)) {
        log(`[remote-sessions] remote profile "${key}" is reachable again`)
      }
    },
    inCooldown(profile) {
      const key = keyOf(profile)
      const entry = downUntil.get(key)
      if (!entry) return false
      if (now() >= entry.until) {
        downUntil.delete(key)
        return false
      }
      return true
    },
    clear() {
      downUntil.clear()
    }
  }
}

/** Resolve to BUDGET_EXCEEDED if `promise` doesn't settle within budgetMs. */
async function raceWithBudget(promise, budgetMs, setTimeoutFn = setTimeout, clearTimeoutFn = clearTimeout) {
  let timer = null
  try {
    return await Promise.race([
      promise,
      new Promise(resolve => {
        timer = setTimeoutFn(() => resolve(BUDGET_EXCEEDED), budgetMs)
      })
    ])
  } finally {
    if (timer !== null) clearTimeoutFn(timer)
  }
}

const rowsOf = data => (Array.isArray(data?.sessions) ? data.sessions : [])

/**
 * Merge each remote profile's session rows into the primary's local aggregate,
 * re-sort by recency, and re-window to the requested page — without ever
 * letting a dead or slow remote stall the result.
 *
 * Per remote profile, in parallel:
 *   - in cooldown            → skip instantly (and drop its stale local total).
 *   - fetch settles in budget → splice its rows, record its real total.
 *   - fetch fails            → contributes nothing; the probe layer
 *                              (spawnPoolBackend) owns marking it down.
 *   - budget exceeded        → contributes nothing THIS refresh; the fetch
 *                              keeps warming the pool for the next one.
 *
 * `base` is consumed as-is: callers pass the primary's response (already
 * containing every LOCAL profile's rows plus possibly-stale rows for remote
 * profiles, which are swapped out here).
 */
async function spliceRemoteSessions({
  base,
  remoteProfiles,
  limit,
  offset,
  order,
  fetchRemote,
  availability,
  isWarm = () => false,
  coldBudgetMs = COLD_REMOTE_SPLICE_BUDGET_MS,
  warmBudgetMs = WARM_REMOTE_SPLICE_BUDGET_MS,
  log = () => {},
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout
}: any) {
  const safeLimit = Math.max(1, Number(limit) || 20)
  const safeOffset = Math.max(0, Number(offset) || 0)
  const recencyField = order === 'started_at' ? 'started_at' : 'last_active'

  const remoteSet = new Set(remoteProfiles)
  const merged = rowsOf(base).filter(s => !remoteSet.has(s?.profile))
  const profileTotals = { ...(base?.profile_totals || {}) }
  let total = (Number(base?.total) || 0) - remoteProfiles.reduce((n, p) => n + (profileTotals[p] || 0), 0)

  await Promise.all(
    remoteProfiles.map(async name => {
      if (availability?.inCooldown(name)) {
        log(`[remote-sessions] splice skipping "${name}" (in unreachable cooldown)`)
        delete profileTotals[name]
        return
      }

      const budgetMs = isWarm(name) ? warmBudgetMs : coldBudgetMs
      const pending = fetchRemote(name)
      let list = null
      try {
        const settled = await raceWithBudget(pending, budgetMs, setTimeoutFn, clearTimeoutFn)
        if (settled === BUDGET_EXCEEDED) {
          // Leave the fetch running: it is warming the profile's backend
          // connection, so the NEXT refresh gets the rows instantly. Swallow
          // its eventual outcome so a late rejection can't crash the process.
          pending.catch(() => {})
          log(`[remote-sessions] splice timed out waiting ${budgetMs}ms for "${name}"; deferring to next refresh`)
          delete profileTotals[name]
          return
        }
        list = settled
      } catch (error) {
        log(`[remote-sessions] splice fetch for "${name}" failed: ${error?.message || error}`)
        delete profileTotals[name]
        return
      }

      if (!list) {
        delete profileTotals[name]
        return
      }

      availability?.markUp(name)
      const rows = rowsOf(list)
      merged.push(...rows)
      profileTotals[name] = Number(list.total) || rows.length
      total += profileTotals[name]
    })
  )

  const recency = s => s?.[recencyField] ?? s?.started_at ?? 0
  merged.sort((a, b) => recency(b) - recency(a))

  return {
    ...base,
    sessions: merged.slice(safeOffset, safeOffset + safeLimit),
    total,
    profile_totals: profileTotals
  }
}

export {
  BUDGET_EXCEEDED,
  COLD_REMOTE_SPLICE_BUDGET_MS,
  LOCAL_BOOT_DEADLINE_MS,
  READINESS_RETRY_INTERVAL_MS,
  REMOTE_DOWN_COOLDOWN_MS,
  REMOTE_PROBE_ATTEMPT_TIMEOUT_MS,
  REMOTE_PROBE_DEADLINE_MS,
  WARM_REMOTE_SPLICE_BUDGET_MS,
  createRemoteAvailability,
  isNothingListeningError,
  raceWithBudget,
  spliceRemoteSessions,
  waitForBackendReady
}
