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

export interface AttachedBackend {
  baseUrl: string
  pid: number
  port: number
  token: string
  wsUrl: string
}

export interface HostBackendRendezvous {
  record: HostBackendRecord
  token: string
}

export interface HostBackendRendezvousState {
  candidate: HostBackendRendezvous | null
  /** A safe loopback port extracted even when the private record is invalid. */
  port: number | null
}

export interface HostBackendAttachDeps {
  /** Read the ledger file; return null when it is missing/unreadable. */
  readLedger: (path: string) => string | null
  /** Read the private same-user host record and token, when available. */
  readRendezvous?: () => HostBackendRendezvousState
  /** Resolve the token the backend actually serves at `GET /`. */
  resolveServedToken: (baseUrl: string) => Promise<string | null>
  /** Reject unless the backend answers its readiness probe. */
  waitForReady: (baseUrl: string, token: string) => Promise<unknown>
  /** Prove a private rendezvous token belongs to the recorded serve process. */
  probeHostIdentity?: (
    baseUrl: string,
    token: string,
    record: HostBackendRecord
  ) => Promise<{ ok: boolean; reason?: string }>
  /** Reject unless `/api/ws` accepts the token — the leg the renderer uses. */
  probeWebSocket: (wsUrl: string) => Promise<{ ok: boolean; reason?: string }>
  log: (message: string) => void
}

export function spawnLedgerPath(hermesHomeRoot: string, join: (...parts: string[]) => string): string {
  return join(hermesHomeRoot, SPAWN_LEDGER_FILENAME)
}

function wsUrlFor(baseUrl: string, token: string): string {
  return `${baseUrl.replace(/^http/, 'ws')}/api/ws?token=${encodeURIComponent(token)}`
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
  rendezvousToken?: string
): Promise<AttachedBackend | null> {
  const baseUrl = recordBaseUrl(record)

  const servedToken = await deps.resolveServedToken(baseUrl).catch(() => null)

  const tokenCandidates = [rendezvousToken, servedToken].filter(
    (token, index, tokens): token is string => Boolean(token) && tokens.indexOf(token) === index
  )

  if (tokenCandidates.length === 0) {
    deps.log(`[attach] ${baseUrl} (pid ${record.pid}) did not publish a session token; not attaching`)

    return null
  }

  for (const token of tokenCandidates) {
    if (rendezvousToken) {
      const identityProbe = deps.probeHostIdentity
        ? await deps.probeHostIdentity(baseUrl, token, record).catch(error => ({
            ok: false,
            reason: error instanceof Error ? error.message : String(error)
          }))
        : null

      if (!identityProbe?.ok) {
        deps.log(
          `[attach] ${baseUrl} (pid ${record.pid}) did not prove its host identity: ${
            identityProbe?.reason || 'identity probe unavailable'
          }`
        )

        continue
      }
    }

    try {
      await deps.waitForReady(baseUrl, token)
    } catch (error) {
      deps.log(
        `[attach] ${baseUrl} (pid ${record.pid}) is not ready: ${error instanceof Error ? error.message : String(error)}`
      )

      continue
    }

    const wsUrl = wsUrlFor(baseUrl, token)

    const probe = await deps.probeWebSocket(wsUrl).catch(error => ({
      ok: false,
      reason: error instanceof Error ? error.message : String(error)
    }))

    if (!probe.ok) {
      deps.log(`[attach] ${baseUrl} (pid ${record.pid}) rejected the session token on /api/ws: ${probe.reason}`)

      continue
    }

    return { baseUrl, pid: record.pid, port: record.port, token, wsUrl }
  }

  return null
}

/**
 * Discover and attach to the host's running backend.
 *
 * Returns null when the host has none (or the escape hatch is set), which is
 * the caller's signal to spawn exactly one.
 */
export async function attachToHostBackend(
  { isolated, ledgerPath }: { isolated: boolean; ledgerPath: string },
  deps: HostBackendAttachDeps
): Promise<AttachedBackend | null> {
  const records = parseSpawnLedger(deps.readLedger(ledgerPath))
  const rendezvousState = deps.readRendezvous?.() || { candidate: null, port: null }
  const rendezvous = rendezvousState.candidate
  const discoveryRecords = rendezvous ? [rendezvous.record, ...records] : records
  const decision = spawnOrAttach({ isolated, records: discoveryRecords })

  if (decision.action === 'spawn') {
    if (decision.reason === 'isolated') {
      deps.log('[attach] HERMES_DESKTOP_ISOLATED_BACKEND is set; spawning a dedicated backend')
    }

    return null
  }

  // Newest first, then the rest: a stale record must not cost us a live one.
  // The private rendezvous record is authoritative when present, but keep the
  // ledger candidates as a fallback for older backends and mixed-version hosts.
  const ordered = [
    ...(rendezvous ? [{ record: rendezvous.record, rendezvousToken: rendezvous.token }] : []),
    ...records.map(record => ({ record, rendezvousToken: undefined }))
  ].sort((left, right) => {
    if (left.rendezvousToken && !right.rendezvousToken) {
      return -1
    }

    if (!left.rendezvousToken && right.rendezvousToken) {
      return 1
    }

    return right.record.registeredAt - left.record.registeredAt
  })

  const rendezvousPort = rendezvous?.record.port ?? rendezvousState.port

  for (const candidate of ordered) {
    // Once a private record names an endpoint, do not bypass its identity proof
    // with the less authoritative ledger token for that same endpoint. A
    // different ledger endpoint is still a valid mixed-version fallback.
    if (!candidate.rendezvousToken && rendezvousPort === candidate.record.port) {
      continue
    }

    const attached = await validate(candidate.record, deps, candidate.rendezvousToken)

    if (attached) {
      deps.log(
        `[attach] attached to the running Hermes backend on ${attached.baseUrl} ` +
          `(pid ${attached.pid}, registered by profile "${candidate.record.profile || 'default'}"); spawning nothing`
      )

      return attached
    }
  }

  return null
}

export interface HostSpawnGateDeps {
  now: () => number
  /** Read the gate record; null when absent, unreadable, or its owner is gone. */
  read: () => { ownerAlive: boolean; startedAt: number } | null
  /** Claim the gate for this process; returns the release. */
  take: () => () => void
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
  const attached = await attachToHostBackend(options, deps)

  if (attached) {
    return { attached }
  }

  if (!options.isolated) {
    const deadline = gate.now() + waitBudgetMs

    while (
      gate.now() < deadline &&
      classifyHostSpawnGate(gate.read(), { now: gate.now(), staleAfterMs: HOST_SPAWN_GATE_STALE_MS }) === 'wait'
    ) {
      deps.log('[attach] another app is starting the host backend; waiting for it instead of spawning a second one')
      await gate.sleep(pollMs)

      const late = await attachToHostBackend(options, deps)

      if (late) {
        return { attached: late }
      }
    }
  }

  return { reservation: { release: gate.take() } }
}
