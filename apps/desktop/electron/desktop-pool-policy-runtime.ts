import type * as nodeFs from 'node:fs'
import path from 'node:path'

import { clampPoolLimits, parsePoolLimits, POOL_LIMITS_DEFAULTS } from './pool-limits'
import {
  BackgroundSlotRetryBackoff,
  isBackgroundSlotRetryDeferred,
  isBackgroundSlotWaitTimeout,
  LocalBackendSpawnCoordinator,
  type LocalBackendSpawnPriority
} from './pool-spawn-coordinator'
import { poolTouchKeys } from './pool-touch-scope'

interface PoolPolicyRetirer {
  assertCanOpen: (key: string, priority: LocalBackendSpawnPriority) => void
  evictTo: (keep: number, freshMs: number) => Promise<string[]>
  retireIdle: (key: string, idleMs: number) => Promise<boolean>
}

export interface DesktopPoolPolicyRuntimeDeps {
  app: { getPath: (name: string) => string }
  backendPool: Map<string, any>
  fs: typeof nodeFs
  getPoolRetirer: () => PoolPolicyRetirer
  getPoolIdleReaper: () => ReturnType<typeof setInterval> | null
  setPoolIdleReaper: (timer: ReturnType<typeof setInterval> | null) => void
  rememberLog: (message: string) => void
  stopPoolBackend: (key: string) => Promise<void>
}

export function createDesktopPoolPolicyRuntime(deps: DesktopPoolPolicyRuntimeDeps) {
  const { app, backendPool, fs, getPoolIdleReaper, setPoolIdleReaper, rememberLog, stopPoolBackend } = deps

  // The real retirer is created later with the pool backend runtime. Resolve it
  // only when a deferred policy function runs; the single retirer remains owner.
  const poolRetirer = {
    assertCanOpen: (key: string, priority: LocalBackendSpawnPriority) => deps.getPoolRetirer().assertCanOpen(key, priority),
    evictTo: (keep: number, freshMs: number) => deps.getPoolRetirer().evictTo(keep, freshMs),
    retireIdle: (key: string, idleMs: number) => deps.getPoolRetirer().retireIdle(key, idleMs)
  }

  // Keep the pool light: cap concurrent profile backends (LRU eviction) and reap
  // idle ones. A user idles at exactly the primary backend; pool backends only
  // exist while a non-primary profile is actively being chatted through.
  // Pool sizing is a device preference (Settings → Advanced → pool rows), not a
  // launch constant: mutable at runtime, persisted in userData, applied live.
  // The legacy HERMES_DESKTOP_POOL_* env vars remain the initial-value fallback
  // for scripted/headless setups; after launch the stored preference wins.
  const POOL_LIMITS_PATH = path.join(app.getPath('userData'), 'pool-limits.json')

  function readPersistedPoolLimits() {
    try {
      const limits = parsePoolLimits(fs.readFileSync(POOL_LIMITS_PATH, 'utf8'))
      rememberLog(
        `[pool-limits] loaded from ${POOL_LIMITS_PATH}: maxBackends=${limits.maxBackends}, idleMs=${limits.idleMs}`
      )

      return limits
    } catch {
      // No persisted file yet — fall back to the legacy env vars so scripted
      // setups keep working. Log which source won: a silently-ignored env var
      // here costs a scripted-setup user a debugging session.
      const fromEnv = clampPoolLimits({
        maxBackends: Number(process.env.HERMES_DESKTOP_POOL_MAX) || undefined,
        idleMs: Number(process.env.HERMES_DESKTOP_POOL_IDLE_MS) || undefined
      })

      if (fromEnv.maxBackends !== POOL_LIMITS_DEFAULTS.maxBackends || fromEnv.idleMs !== POOL_LIMITS_DEFAULTS.idleMs) {
        rememberLog(
          `[pool-limits] no saved file; using env-var overrides: maxBackends=${fromEnv.maxBackends}, idleMs=${fromEnv.idleMs}`
        )
      } else {
        rememberLog('[pool-limits] no saved file and no env overrides; using defaults')
      }

      return fromEnv
    }
  }

  function persistPoolLimits(limits) {
    try {
      fs.mkdirSync(path.dirname(POOL_LIMITS_PATH), { recursive: true })
      // Atomic write: write to a temp file in the same directory, then rename.
      // A crash mid-write would otherwise leave truncated JSON and silently
      // lose the user's saved sizing.
      const tmpPath = `${POOL_LIMITS_PATH}.tmp`
      fs.writeFileSync(tmpPath, JSON.stringify(limits, null, 2), 'utf8')
      fs.renameSync(tmpPath, POOL_LIMITS_PATH)
    } catch (error) {
      rememberLog(`[pool-limits] write failed: ${error.message}`)
    }
  }

  let poolLimits = readPersistedPoolLimits()
  // Hard cap on local backends that are starting OR running (the LRU eviction
  // above is soft — it spares keepalive-fresh entries). Follows the live
  // preference: setPoolLimits() pushes a new max into the coordinator.
  const localBackendSpawnCoordinator = new LocalBackendSpawnCoordinator(poolLimits.maxBackends)
  const backgroundSlotRetryBackoff = new BackgroundSlotRetryBackoff()
  // How long a spawn may wait for a free local slot. Must stay under the
  // renderer's BACKEND_BOOT_WAIT_TIMEOUT_MS (45s, src/lib/with-timeout.ts) so
  // the queued ticket fails before the renderer does and the user sees why.
  const POOL_SLOT_WAIT_MS = 30_000

  function spawnPriorityFrom(value: unknown): LocalBackendSpawnPriority {
    return value === 'foreground' ? 'foreground' : 'background'
  }

  // Foreground intent for a dial whose pool entry does not exist yet: a user
  // click that joins an in-flight backendDialClaims claim never re-enters
  // ensureBackend(), and the claim owner may still be awaiting poolStopper /
  // registry resolution before backendPool.set(). The local spawn takes the mark
  // right before its slot request; the IPC handler that set it clears it once
  // the claim settles, so a dial that never reaches a slot request (primary
  // route, remote scope, a guard rejection) cannot leave it for a later
  // hydration spawn of the same key to pick up.
  const pendingForegroundSpawns = new Set<string>()

  function takeForegroundSpawn(...poolKeys: string[]): boolean {
    let marked = false

    for (const poolKey of poolKeys) {
      marked = pendingForegroundSpawns.delete(poolKey) || marked
    }

    return marked
  }

  // Upgrade a pooled entry (running, spawning, or queued for a slot) to
  // foreground so a queued slot wait can take the reserved foreground slot.
  function promotePoolEntry(entry: any): void {
    entry.spawnPriority = 'foreground'
    entry.localBackendSpawnRequest?.promote?.('foreground')
  }

  // Land a spawn failure in desktop.log. Background slot waits back off per
  // profile under a saturated pool, so a roster refresh cannot create a retry
  // storm while a user-triggered foreground open still gets its reserved slot.
  function logPoolSpawnFailure(label: string, error: unknown): void {
    if (isBackgroundSlotRetryDeferred(error)) {
      return
    }

    if (isBackgroundSlotWaitTimeout(error)) {
      rememberLog(`Profile backend ${label} slot wait timed out (background); retry is backing off`)
    } else {
      rememberLog(
        `Hermes backend for profile ${label} failed to start: ${error instanceof Error ? error.message : String(error)}`
      )
    }
  }

  // Apply foreground intent to the dial claim for `scopeKey`: an entry already
  // in the pool is promoted directly, otherwise the intent is marked for the
  // spawn the claim owner is about to start. Returns the cleanup that clears a
  // mark the dial never consumed.
  function applySpawnPriority(scopeKey: string, spawnPriority: LocalBackendSpawnPriority): () => void {
    // The renderer's socket-close event may beat its parking IPC. Main owns
    // this fence too, so that race cannot resurrect the retired generation.
    for (const key of poolTouchKeys(scopeKey)) {
      poolRetirer.assertCanOpen(key, spawnPriority)
    }

    if (spawnPriority !== 'foreground') {
      return () => undefined
    }

    const existing = backendPool.get(scopeKey)

    if (existing) {
      promotePoolEntry(existing)
    } else {
      pendingForegroundSpawns.add(scopeKey)
    }

    return () => void pendingForegroundSpawns.delete(scopeKey)
  }

  function poolMaxBackends() {
    return poolLimits.maxBackends
  }

  function poolIdleMs() {
    return poolLimits.idleMs
  }

  /**
   * Apply new limits live: persist, then converge the running pool — evict
   * LRU backends down to the new max, and let the (already running) idle
   * reaper handle a shortened idle window on its next tick. Returns the
   * limits actually in force (post-clamp).
   */
  function setPoolLimits(raw) {
    poolLimits = clampPoolLimits(raw)
    persistPoolLimits(poolLimits)
    localBackendSpawnCoordinator.setLimit(poolLimits.maxBackends)
    void evictLruPoolBackends(poolMaxBackends())
    startPoolIdleReaper()

    return { ...poolLimits }
  }

  // A backend touched within this window has a live renderer socket (the keepalive
  // pings every 60s for every open profile). LRU eviction must spare these — a
  // concurrent multi-profile session keeps several backends "fresh" at once, and
  // killing one to honor the soft cap would abort a running agent.
  //
  // The window is intentionally MUCH wider than the 60s ping cadence:
  //   * 1 missed ping    = +60s of apparent silence
  //   * WSL2 IPC stall  = the renderer's `hermes:backend:touch` roundtrips
  //                       through 9p; a single brief 9p hiccup can stretch a
  //                       ping to ~30s of observed silence (#95189: gateways
  //                       exited every ~2 min on WSL2 because the previous
  //                       90s window left no headroom — one delayed ping
  //                       pushed a live backend past the threshold and the
  //                       cap-driven eviction killed the active profile's
  //                       backend mid-session, re-minting runtime ids and
  //                       re-allocating pooled gateway secondaries ~700×/day).
  //   * 3× ping + 60s headroom = ~4 min, comfortable margin for two missed
  //     pings + WSL2 IPC stall. The hard ceiling for the cap-eligible set is
  //     pool idle window above (default 10 min) — this constant only governs the
  //     "is this backend plausibly still alive" question for LRU eviction,
  //     not when the idle reaper definitively tears a backend down.
  const POOL_KEEPALIVE_FRESH_MS = Math.max(
    120_000,
    Number(process.env.HERMES_DESKTOP_POOL_KEEPALIVE_FRESH_MS) || 4 * 60_000
  )

  // Mark a pool profile as recently used so the idle reaper spares it. The
  // renderer calls this when it opens a profile's chat WS and periodically while
  // streaming, since the main process can't see the direct renderer↔backend WS.
  // It also reports whether a prompt turn currently leases the backend: a
  // foreground dial that must retire a resident skips leased ones early. That
  // flag is an optimisation, never the proof — the backend probe is (see
  // pool-retire.ts). Shape from #104871 by @bounce12340.
  function touchPoolBackend(profile, options: { activeTurn?: boolean } = {}) {
    for (const key of poolTouchKeys(profile)) {
      const entry = backendPool.get(key)

      if (entry) {
        entry.lastActiveAt = Date.now()

        if (typeof options.activeTurn === 'boolean') {
          entry.activeTurn = options.activeTurn
        }

        return
      }
    }
  }

  // Evict least-recently-used SPAWNED pool backends until at most `keep` remain —
  // but only ever evict backends without a live renderer socket (stale beyond the
  // keepalive window). When every backend is actively kept alive we let the pool
  // exceed the soft cap rather than kill a running session. Process-less
  // descriptor entries (remote/cloud registry sources, per-profile remote
  // overrides — `entry.process === null`) are excluded from the cap entirely:
  // they hold no local process, so counting them used to let a roster refresh
  // across N registered remote connections LRU-evict a REAL local backend that
  // was merely idle past the keepalive window. Descriptors are still reclaimed
  // by the idle reaper.
  async function evictLruPoolBackends(keep) {
    return poolRetirer.evictTo(Math.max(0, keep), POOL_KEEPALIVE_FRESH_MS)
  }

  function startPoolIdleReaper() {
    let poolIdleReaper = getPoolIdleReaper()

    if (poolIdleReaper) {
      return
    }

    poolIdleReaper = setInterval(() => {
      const now = Date.now()

      for (const [profile, entry] of [...backendPool.entries()]) {
        if (now - (entry.lastActiveAt || 0) > poolIdleMs()) {
          // Remote descriptors hold no child/slot. Local children require the
          // same admission authority as foreground and LRU reclamation.
          const retiring = entry.process ? poolRetirer.retireIdle(profile, poolIdleMs()) : stopPoolBackend(profile)

          void retiring.catch(error => rememberLog(`Pool idle retirement failed: ${String(error)}`))
        }
      }

      if (backendPool.size === 0 && poolIdleReaper) {
        clearInterval(poolIdleReaper)
        poolIdleReaper = null
        setPoolIdleReaper(null)
      }
    }, 60_000)
    setPoolIdleReaper(poolIdleReaper)

    if (typeof poolIdleReaper.unref === 'function') {
      poolIdleReaper.unref()
    }
  }

  return {
    localBackendSpawnCoordinator,
    backgroundSlotRetryBackoff,
    POOL_SLOT_WAIT_MS,
    spawnPriorityFrom,
    takeForegroundSpawn,
    promotePoolEntry,
    logPoolSpawnFailure,
    applySpawnPriority,
    poolMaxBackends,
    poolIdleMs,
    setPoolLimits,
    touchPoolBackend,
    evictLruPoolBackends,
    startPoolIdleReaper,
    getPoolLimits: () => poolLimits
  }
}
