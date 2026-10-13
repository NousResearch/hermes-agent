// Attach to the host's running Hermes backend (multiplex-only, Desktop half).
//
// `backend-discovery.ts` owns the pure decision; this module performs the IO
// ladder around it: read the machine-root ledger, validate a candidate at the
// boundary that actually matters (HTTP readiness → served session token →
// WebSocket auth), and hold a host-level gate so two apps starting at once
// produce one backend instead of two.
//
// Every dependency is injected so the ladder runs in a test without Electron.

import {
  classifyHostSpawnGate,
  type HostBackendRecord,
  parseSpawnLedger,
  recordBaseUrl,
  SPAWN_LEDGER_FILENAME,
  spawnOrAttach
} from './backend-discovery'

/** A gate held longer than this belongs to a spawner that never finished. */
export const HOST_SPAWN_GATE_STALE_MS = 60_000

/**
 * How long a ready backend whose identity is unreadable is re-read before the
 * token attach. A slow /api/health is ours far more often than not, so it gets
 * a few retries, not the whole spawn-gate budget: a token-valid backend must
 * not hold startup for a minute.
 */
export const HOST_IDENTITY_RETRY_MS = 3_000

export interface AttachedBackend {
  baseUrl: string
  pid: number
  port: number
  token: string
  wsUrl: string
}

export interface HostBackendAttachDeps {
  /** Read the ledger file; return null when it is missing/unreadable. */
  readLedger: (path: string) => string | null
  /** Resolve the token the backend actually serves at `GET /`. */
  resolveServedToken: (baseUrl: string) => Promise<string | null>
  /**
   * Session token the backend published for this record when `GET /` withholds
   * it. Absent readers keep the dashboard-HTML-only handshake.
   */
  publishedTokenFor?: (record: HostBackendRecord) => string | null
  /** Reject unless the backend answers its readiness probe. */
  waitForReady: (baseUrl: string, token: string) => Promise<unknown>
  /** Reject unless `/api/ws` accepts the token — the leg the renderer uses. */
  probeWebSocket: (wsUrl: string) => Promise<{ ok: boolean; reason?: string }>
  /**
   * PID liveness probe (real: `isPidAliveWindows`). Records whose backend is
   * already gone are skipped before any network I/O (#123586). Absent, every
   * record is probed as before.
   */
  isPidAlive?: (pid: number) => boolean
  /**
   * Commit this Desktop would spawn its own backend from (the checkout HEAD).
   * A backend that booted from other code — a CLI/launchd `hermes serve` that
   * outlived an update — answers 503 "Restart required" forever, and Restart
   * would re-attach it, so it is skipped. Absent or null (packaged/non-git
   * install): the token-only handshake, as before.
   */
  expectedCodeIdentity?: () => Promise<string | null>
  /**
   * Boot commit a backend reports; defaults to {@link fetchBackendCodeIdentity}.
   * Null = the backend answered without one; a rejection = it could not be read.
   */
  backendCodeIdentity?: (baseUrl: string) => Promise<string | null>
  log: (message: string) => void
}

/**
 * The commit a backend booted from: `commit` on the public `GET /api/health`,
 * resolved once at import (`get_version_info` is cached), so an update moving
 * the checkout underneath the process does not change it. Null only when the
 * backend answered and predates the field; a timeout, network error, non-2xx
 * or unparsable body rejects, because a slow backend is not a different one.
 */
export async function fetchBackendCodeIdentity(baseUrl: string, timeoutMs = 3000): Promise<string | null> {
  const response = await fetch(`${baseUrl}/api/health`, { signal: AbortSignal.timeout(timeoutMs) })

  if (!response.ok) {
    throw new Error(`/api/health answered ${response.status}`)
  }

  const body: unknown = await response.json()

  return body && typeof body === 'object' && 'commit' in body ? nonemptyToken(String(body.commit ?? '')) : null
}

/** A ready backend whose identity could not be read yet: neither attach nor spawn beside it. */
const UNCONFIRMED = 'unconfirmed'

/** The commit this Desktop would spawn its backend from: its checkout's HEAD, or null off git. */
export async function checkoutHeadIdentity(
  root: string,
  isGitCheckout: (root: string) => boolean,
  git: (args: string[], options: { cwd: string; timeoutMs: number }) => Promise<{ code: number | null; stdout: string }>
): Promise<string | null> {
  if (!isGitCheckout(root)) {
    return null
  }

  const head = await git(['rev-parse', 'HEAD'], { cwd: root, timeoutMs: 5000 })

  return head.code === 0 ? head.stdout.trim() || null : null
}

export function spawnLedgerPath(hermesHomeRoot: string, join: (...parts: string[]) => string): string {
  return join(hermesHomeRoot, SPAWN_LEDGER_FILENAME)
}

function wsUrlFor(baseUrl: string, token: string): string {
  return `${baseUrl.replace(/^http/, 'ws')}/api/ws?token=${encodeURIComponent(token)}`
}

function nonemptyToken(value: string | null | undefined): string | null {
  const token = String(value ?? '').trim()

  return token || null
}

/**
 * Validate one candidate all the way to a usable connection, or return null.
 *
 * A failed rung is not an error: it means this record is not the backend we can
 * use, and the caller falls through to the next rung (another record, then
 * spawning). Only a *validated* candidate is ever returned.
 */
async function validate(
  record: HostBackendRecord,
  deps: HostBackendAttachDeps,
  expectedCommit: string | null,
  unreadableIdentity: 'unconfirmed' | 'token-attach'
): Promise<AttachedBackend | null | typeof UNCONFIRMED> {
  const baseUrl = recordBaseUrl(record)
  const servedToken = nonemptyToken(await deps.resolveServedToken(baseUrl).catch(() => null))
  let publishedToken: string | null = null

  if (!servedToken && deps.publishedTokenFor) {
    try {
      publishedToken = nonemptyToken(deps.publishedTokenFor(record))
    } catch {
      publishedToken = null
    }
  }

  const token = servedToken || publishedToken

  if (!token) {
    deps.log(`[attach] ${baseUrl} (pid ${record.pid}) did not publish a session token; not attaching`)

    return null
  }

  try {
    await deps.waitForReady(baseUrl, token)
  } catch (error) {
    deps.log(`[attach] ${baseUrl} (pid ${record.pid}) is not ready: ${(error as Error).message}`)

    return null
  }

  if (expectedCommit) {
    // Unknown backend identity is a mismatch: a backend predating the field is
    // older code by construction.
    const resolve = deps.backendCodeIdentity ?? fetchBackendCodeIdentity
    let theirs: string | null

    try {
      theirs = nonemptyToken(await resolve(baseUrl))
    } catch (error) {
      deps.log(
        `[attach] ${baseUrl} (pid ${record.pid}) is ready but its code identity is unreadable: ${(error as Error).message}`
      )

      if (unreadableIdentity === 'unconfirmed') {
        return UNCONFIRMED
      }

      // Retries exhausted: an unreadable identity is still not a mismatch, so
      // keep the token-validated attach rather than spawn a second backend.
      theirs = expectedCommit
    }

    if (theirs?.toLowerCase() !== expectedCommit.toLowerCase()) {
      deps.log(
        `[attach] ${baseUrl} (pid ${record.pid}) runs ${theirs ? `code ${theirs}` : 'code of unknown version'}, ` +
          `this Desktop expects ${expectedCommit}; not attaching`
      )

      return null
    }
  }

  const wsUrl = wsUrlFor(baseUrl, token)
  const probe = await deps.probeWebSocket(wsUrl).catch(error => ({ ok: false, reason: error.message }))

  if (!probe.ok) {
    deps.log(`[attach] ${baseUrl} (pid ${record.pid}) rejected the session token on /api/ws: ${probe.reason}`)

    return null
  }

  return { baseUrl, pid: record.pid, port: record.port, token, wsUrl }
}

/**
 * Discover and attach to the host's running backend.
 *
 * Returns null when the host has none (or the escape hatch is set), which is
 * the caller's signal to spawn exactly one.
 */
export async function attachToHostBackend(
  options: { isolated: boolean; ledgerPath: string },
  deps: HostBackendAttachDeps
): Promise<AttachedBackend | null> {
  const found = await findHostBackend(options, deps, await expectedCommitFor(deps))

  return found === UNCONFIRMED ? null : found
}

/** Resolved once per attach call: `git rev-parse HEAD` must not run per retry round. */
async function expectedCommitFor(deps: HostBackendAttachDeps): Promise<string | null> {
  return nonemptyToken(await deps.expectedCodeIdentity?.().catch(() => null))
}

async function findHostBackend(
  { isolated, ledgerPath }: { isolated: boolean; ledgerPath: string },
  deps: HostBackendAttachDeps,
  expectedCommit: string | null,
  unreadableIdentity: 'unconfirmed' | 'token-attach' = 'unconfirmed'
): Promise<AttachedBackend | null | typeof UNCONFIRMED> {
  const records = parseSpawnLedger(deps.readLedger(ledgerPath))
  const decision = spawnOrAttach({ isolated, records, isPidAlive: deps.isPidAlive })

  if (decision.action === 'spawn') {
    if (decision.reason === 'isolated') {
      deps.log('[attach] HERMES_DESKTOP_ISOLATED_BACKEND is set; spawning a dedicated backend')
    }

    return null
  }

  // Newest first, then the rest: a stale record must not cost us a live one.
  // Dead PIDs are skipped here too, so the fallback rung never dials a port
  // whose owner is already gone (#123586).
  const ordered = [
    decision.record,
    ...records.filter(
      candidate => candidate !== decision.record && (!deps.isPidAlive || deps.isPidAlive(candidate.pid))
    )
  ]

  let unconfirmed = false

  for (const record of ordered) {
    const attached = await validate(record, deps, expectedCommit, unreadableIdentity)

    if (attached === UNCONFIRMED) {
      unconfirmed = true
    } else if (attached) {
      deps.log(
        `[attach] attached to the running Hermes backend on ${attached.baseUrl} ` +
          `(pid ${attached.pid}, registered by profile "${record.profile || 'default'}"); spawning nothing`
      )

      return attached
    }
  }

  return unconfirmed ? UNCONFIRMED : null
}

export interface HostSpawnGateDeps {
  now: () => number
  /** Read the gate record; null when absent, unreadable, or its owner is gone. */
  read: () => { ownerAlive: boolean; startedAt: number } | null
  /** Atomically claim the gate; null means another process won the race. */
  take: () => (() => void) | null
  sleep: (ms: number) => Promise<void>
}

export interface SpawnReservation {
  release: () => void
}

/**
 * Attach to the host backend, or come back holding the host spawn gate.
 *
 * Two apps launching at once both find an empty ledger; without a gate they
 * each spawn a backend and the host ends up with two. The loser waits for the
 * winner's backend to register and attaches to it instead. The wait is bounded
 * and a gate whose owner died is taken over, so a crashed spawner cannot wedge
 * every later launch — worst case we spawn, which is today's behaviour.
 *
 * The caller MUST release the reservation once its spawn is ready or has
 * failed; the ledger entry only appears after the new backend binds.
 */
export async function attachOrReserveSpawn(
  options: { isolated: boolean; ledgerPath: string },
  deps: HostBackendAttachDeps,
  gate: HostSpawnGateDeps,
  { pollMs = 500, waitBudgetMs = HOST_SPAWN_GATE_STALE_MS }: { pollMs?: number; waitBudgetMs?: number } = {}
): Promise<{ attached: AttachedBackend } | { reservation: SpawnReservation }> {
  const expectedCommit = await expectedCommitFor(deps)
  let found = await findHostBackend(options, deps, expectedCommit)

  if (found && found !== UNCONFIRMED) {
    return { attached: found }
  }

  if (options.isolated) {
    return { reservation: { release: () => {} } }
  }

  const deadline = gate.now() + waitBudgetMs
  // An unreadable identity gets its own short budget, counted from first sight.
  let identityDeadline = found === UNCONFIRMED ? gate.now() + HOST_IDENTITY_RETRY_MS : Infinity

  while (gate.now() < (found === UNCONFIRMED ? Math.min(deadline, identityDeadline) : deadline)) {
    const gateState = gate.read()

    // A ready backend we could not identify may well be ours: re-read it
    // instead of spawning a second one beside it.
    if (
      found !== UNCONFIRMED &&
      classifyHostSpawnGate(gateState, {
        now: gate.now(),
        staleAfterMs: HOST_SPAWN_GATE_STALE_MS
      }) === 'take'
    ) {
      const release = gate.take()

      if (release) {
        return { reservation: { release } }
      }
    }

    deps.log(
      found === UNCONFIRMED
        ? '[attach] retrying the running backend whose code identity was unreadable before spawning another'
        : '[attach] another app is starting the host backend; waiting for it instead of spawning a second one'
    )
    await gate.sleep(pollMs)

    found = await findHostBackend(options, deps, expectedCommit)

    if (found && found !== UNCONFIRMED) {
      return { attached: found }
    }

    if (found === UNCONFIRMED) {
      identityDeadline = Math.min(identityDeadline, gate.now() + HOST_IDENTITY_RETRY_MS)
    }
  }

  // Only a definitive non-matching commit is a mismatch: a ready backend whose
  // identity stayed unreadable for its whole retry budget keeps the token attach.
  if (found === UNCONFIRMED) {
    found = await findHostBackend(options, deps, expectedCommit, 'token-attach')

    if (found && found !== UNCONFIRMED) {
      return { attached: found }
    }
  }

  // Preserve the bounded startup fallback when a stale/unreadable gate cannot
  // be claimed. The no-op reservation owns no file and therefore removes none.
  return { reservation: { release: gate.take() ?? (() => {}) } }
}
