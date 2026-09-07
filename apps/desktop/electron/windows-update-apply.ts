import fs from 'node:fs'

import {
  acquireUpdateMarker,
  markerPath,
  readLiveUpdateMarker,
  releaseUpdateMarkerIfOwnedBy,
  transferUpdateMarkerIfOwnedBy,
  type UpdateMarkerClaim,
  type UpdateMarkerState
} from './update-marker'
import {
  authorizeUpdateMutation,
  runAuthorizedUpdateMutation,
  type UpdateMutationPermit,
  type UpdatePreflightOutcome
} from './update-preflight'
import type { VenvBlockerScanResult } from './venv-blocker-scan'
import { queryProcessCreatedAt } from './windows-process-identity'
import {
  runUpdaterHandoffTransaction,
  type UpdaterHandoffObservation,
  type UpdaterHandoffResult
} from './windows-update-orchestration'

/**
 * Handoff ack sidecar (#B4) — `<marker>.ack`, three lines:
 *
 *     <nonce>\n<claiming pid>\n<claiming pid's kernel creation epoch>\n
 *
 * A sidecar rather than extra marker lines: the two-line marker body is a
 * cross-language wire format (Rust updater, windows.ps1, update-marker.ts,
 * hermes_mcp_update_gate.py, update_lock.py) and widening it would need all
 * five readers to move at once.
 */
export const UPDATE_HANDOFF_ACK_SUFFIX = '.ack'
const DEFAULT_CLAIM_TIMEOUT_MS = 10_000
const DEFAULT_CLAIM_POLL_MS = 100
// The claimant stamps whole seconds; kernel creation time carries fractions.
const CLAIM_IDENTITY_TOLERANCE_SECONDS = 1.5

export interface UpdaterHandoffAck {
  nonce: string
  pid: number
  createdAt: number
}

export function updateHandoffAckPath(hermesHome: string): string {
  return markerPath(hermesHome) + UPDATE_HANDOFF_ACK_SUFFIX
}

/** Parse the ack sidecar. Anything unexpected is "no ack", never a partial one. */
export function readUpdateHandoffAck(
  hermesHome: string,
  readFile: (file: string) => string = file => fs.readFileSync(file, 'utf8')
): UpdaterHandoffAck | null {
  let raw: string

  try {
    raw = readFile(updateHandoffAckPath(hermesHome))
  } catch {
    return null
  }

  const lines = String(raw).split(/\r?\n/)

  if (lines.at(-1) === '') { lines.pop() }

  if (lines.length !== 3) { return null }

  const [nonce, pidLine, createdAtLine] = lines.map(line => line.trim())
  const pid = Number(pidLine)
  const createdAt = Number(createdAtLine)

  if (!/^[0-9a-f]{16,128}$/.test(nonce)) { return null }

  if (!Number.isSafeInteger(pid) || pid <= 0 || !Number.isFinite(createdAt) || createdAt <= 0) { return null }

  return { nonce, pid, createdAt }
}

export interface AcknowledgedUpdaterClaimDeps {
  hermesHome: string
  /** The nonce handed to this updater, and to nobody else. */
  nonce: string
  /** Claimants that cannot be the updater (this Desktop, the cmd wrapper). */
  excludedPids: number[]
  /** Marker claims stamped before the spawn are not ours. */
  startedAfter: number
  timeoutMs?: number
  pollMs?: number
  now?: () => number
  sleep?: (ms: number) => Promise<void>
  readMarker?: () => UpdateMarkerState | null
  readAck?: () => UpdaterHandoffAck | null
  queryCreatedAt?: (pid: number) => Promise<number | null>
}

/**
 * Wait until the marker is owned by a process that proved it is our updater.
 *
 * All of the following must hold, or the handoff is refused and the Desktop
 * stays alive: the ack exists, it carries OUR nonce, it names the same pid the
 * marker names, that pid is not one we excluded, and the OS still reports that
 * pid with the creation time the ack recorded — inside the window that opened
 * when we spawned. A same-user impostor writing a plausible marker body no
 * longer passes, because it never saw the nonce.
 */
export async function waitForAcknowledgedUpdaterClaim(deps: AcknowledgedUpdaterClaimDeps): Promise<boolean> {
  const now = deps.now ?? Date.now
  const sleep = deps.sleep ?? ((ms: number) => new Promise<void>(resolve => { setTimeout(resolve, ms) }))
  const readMarker = deps.readMarker ?? (() => readLiveUpdateMarker(deps.hermesHome))
  const readAck = deps.readAck ?? (() => readUpdateHandoffAck(deps.hermesHome))
  const queryCreatedAt = deps.queryCreatedAt ?? ((pid: number) => queryProcessCreatedAt(pid))
  const pollMs = Math.max(1, deps.pollMs ?? DEFAULT_CLAIM_POLL_MS)
  const deadline = now() + (deps.timeoutMs ?? DEFAULT_CLAIM_TIMEOUT_MS)

  if (typeof deps.nonce !== 'string' || deps.nonce.length === 0) {
    return false
  }

  while (now() < deadline) {
    const owner = readMarker()

    if (owner?.kind === 'live' && !deps.excludedPids.includes(owner.pid)) {
      const ack = readAck()

      if (ack && ack.nonce === deps.nonce && ack.pid === owner.pid) {
        const createdAt = await queryCreatedAt(owner.pid)

        const live = typeof createdAt === 'number' && Number.isFinite(createdAt) && createdAt > 0

        if (live &&
          Math.abs(createdAt - ack.createdAt) <= CLAIM_IDENTITY_TOLERANCE_SECONDS &&
          Math.abs(createdAt - owner.startedAt) <= CLAIM_IDENTITY_TOLERANCE_SECONDS &&
          createdAt >= deps.startedAfter - CLAIM_IDENTITY_TOLERANCE_SECONDS) {
          return true
        }
      }
    }

    await sleep(Math.min(pollMs, Math.max(0, deadline - now())))
  }

  return false
}

export interface RecoveryUpdaterAuthenticationDeps {
  hermesHome: string
  nonce: string
  /** PID from OUR spawn() — not attacker-supplied — and its captured generation. */
  childPid: number
  childCreatedAt: number | null
  /**
   * True on the full-repair path, where the Desktop still holds the marker for
   * gate continuity. The child physically cannot claim it, so no marker read
   * can ever be evidence about the child; only the ack or the spawn handle can.
   */
  desktopHoldsMarker: boolean
  startedAfter: number
  isChildGenerationActive: () => boolean
  timeoutMs?: number
  pollMs?: number
  now?: () => number
  sleep?: (ms: number) => Promise<void>
  readMarker?: () => UpdateMarkerState | null
  readAck?: () => UpdaterHandoffAck | null
  queryCreatedAt?: (pid: number) => Promise<number | null>
}

/**
 * Authenticate the bootstrap-recovery updater (#B4).
 *
 * The old check planted the child's PID into the marker with
 * `transferUpdateMarkerIfOwnedBy` and then waited for a marker owned by that
 * PID — satisfied by construction, so it proved nothing at all. What is
 * actually unforgeable here is the spawn handle: `childPid` came from our own
 * `spawn()`, so no third party can be it.
 *
 * Accepted evidence, strongest first:
 *  1. an ack sidecar carrying OUR nonce and naming the child, still live with
 *     the creation time it recorded;
 *  2. off the repair path only, a marker the child claimed on its own (the
 *     staged updater's UpdateMarkerGuard) — evidence precisely because the
 *     Desktop did not write it;
 *  3. on the repair path, where 2 is impossible, the child generation we
 *     spawned still being alive with its original kernel creation time.
 */
export async function authenticateRecoveryUpdaterHandoff(
  deps: RecoveryUpdaterAuthenticationDeps
): Promise<boolean> {
  const now = deps.now ?? Date.now
  const sleep = deps.sleep ?? ((ms: number) => new Promise<void>(resolve => { setTimeout(resolve, ms) }))
  const readMarker = deps.readMarker ?? (() => readLiveUpdateMarker(deps.hermesHome))
  const readAck = deps.readAck ?? (() => readUpdateHandoffAck(deps.hermesHome))
  const queryCreatedAt = deps.queryCreatedAt ?? ((pid: number) => queryProcessCreatedAt(pid))
  const pollMs = Math.max(1, deps.pollMs ?? DEFAULT_CLAIM_POLL_MS)
  const deadline = now() + (deps.timeoutMs ?? DEFAULT_CLAIM_TIMEOUT_MS)

  if (!Number.isSafeInteger(deps.childPid) || deps.childPid <= 0 ||
    deps.childCreatedAt === null || !Number.isFinite(deps.childCreatedAt) || deps.childCreatedAt <= 0) {
    return false
  }

  const generationIsIntact = async (): Promise<boolean> => {
    if (!deps.isChildGenerationActive()) { return false }

    const createdAt = await queryCreatedAt(deps.childPid)

    return typeof createdAt === 'number' && createdAt === deps.childCreatedAt
  }

  while (now() < deadline) {
    const ack = readAck()

    if (ack && deps.nonce && ack.nonce === deps.nonce && ack.pid === deps.childPid &&
      Math.abs(ack.createdAt - deps.childCreatedAt) <= CLAIM_IDENTITY_TOLERANCE_SECONDS &&
      await generationIsIntact()) {
      return true
    }

    if (!deps.desktopHoldsMarker) {
      const owner = readMarker()

      if (owner?.kind === 'live' && owner.pid === deps.childPid &&
        owner.startedAt >= deps.childCreatedAt - 1 &&
        owner.startedAt >= deps.startedAfter - CLAIM_IDENTITY_TOLERANCE_SECONDS &&
        await generationIsIntact()) {
        return true
      }
    }

    await sleep(Math.min(pollMs, Math.max(0, deadline - now())))
  }

  // Repair keeps the marker under the Desktop's claim on purpose, so the only
  // honest proof left is the child we started still running as we started it.
  return deps.desktopHoldsMarker ? generationIsIntact() : false
}

/** Best-effort removal of a spent ack so a later handoff cannot reuse it. */
export function discardUpdateHandoffAck(
  hermesHome: string,
  remove: (file: string) => void = file => fs.unlinkSync(file)
): void {
  try {
    remove(updateHandoffAckPath(hermesHome))
  } catch {
    // A missing or locked ack is not a failure: the nonce is single-use, so a
    // leftover file can never authenticate the next handoff.
  }
}

export interface RecoveryHandoffRestoration<TChild> {
  /** null when the spawn itself threw. */
  child: TChild | null
  /** The generation we captured, or null if we never got one. */
  createdAt: number | null
  /** True only if the repair claim was handed to the child before the abort. */
  markerTransferred: boolean
}

export interface RecoveryUpdaterHandoffDeps<TChild, TValue> {
  hermesHome: string
  /** Single-use nonce handed to this updater through its environment. */
  nonce: string
  /** Unix seconds captured before the spawn; earlier claims are not ours. */
  startedAfter: number
  /**
   * The marker claim the Desktop holds on the full-repair path, or null on the
   * gentle path where the staged updater claims the marker itself. Its
   * presence is what makes the handoff a repair handoff.
   */
  repairClaim: UpdateMarkerClaim | null
  spawn: () => TChild
  observe: (child: TChild) => Promise<UpdaterHandoffObservation>
  /** PID from OUR spawn handle, or null when the child never started. */
  childPid: (child: TChild) => number | null
  /** Kernel creation time of the spawned generation; read once, before any check. */
  captureCreatedAt: (child: TChild, pid: number) => Promise<number | null>
  isChildGenerationActive: (child: TChild) => boolean
  commit: (child: TChild) => TValue | Promise<TValue>
  restore: (restoration: RecoveryHandoffRestoration<TChild>) => void | Promise<void>
  authenticationError: string
  authenticate?: (deps: RecoveryUpdaterAuthenticationDeps) => Promise<boolean>
  transferMarker?: (
    hermesHome: string,
    owner: UpdateMarkerClaim,
    next: { pid: number; startedAt: number }
  ) => Promise<boolean>
}

/**
 * The bootstrap-recovery handoff, as one transaction.
 *
 * Recovery runs before any window exists, so it cannot go through
 * applyWindowsUpdate (no progress channel, no preflight, and on the repair
 * path the marker is already held). What it must not do is re-derive the
 * handoff for itself: the version that lived in main.ts transferred the marker
 * to the child and then treated the resulting marker as proof the child had
 * claimed it (#B4).
 *
 * The order is the contract. Capture the spawned generation, authenticate
 * against evidence the child alone could produce, and only then hand over the
 * marker. A failed transfer fails the handoff, so the gate never ends up owned
 * by a process we did not authenticate.
 */
export async function runRecoveryUpdaterHandoff<TChild, TValue>(
  deps: RecoveryUpdaterHandoffDeps<TChild, TValue>
): Promise<UpdaterHandoffResult<TValue>> {
  const authenticate = deps.authenticate ?? authenticateRecoveryUpdaterHandoff
  const transferMarker = deps.transferMarker ?? transferUpdateMarkerIfOwnedBy
  let createdAt: number | null = null
  let markerTransferred = false

  return runUpdaterHandoffTransaction<TChild, TValue>({
    spawn: deps.spawn,
    observe: deps.observe,
    authenticate: async child => {
      const pid = deps.childPid(child)

      if (pid === null) { return false }

      createdAt = await deps.captureCreatedAt(child, pid)

      if (createdAt === null) { return false }

      const authenticated = await authenticate({
        hermesHome: deps.hermesHome,
        nonce: deps.nonce,
        childPid: pid,
        childCreatedAt: createdAt,
        desktopHoldsMarker: deps.repairClaim !== null,
        startedAfter: deps.startedAfter,
        isChildGenerationActive: () => deps.isChildGenerationActive(child)
      })

      if (!authenticated) { return false }

      if (deps.repairClaim) {
        // A hand-over of the gate, never evidence for it.
        markerTransferred = await transferMarker(
          deps.hermesHome,
          deps.repairClaim,
          { pid, startedAt: createdAt }
        )

        if (!markerTransferred) { return false }
      }

      return true
    },
    commit: deps.commit,
    restore: child => deps.restore({ child, createdAt, markerTransferred }),
    authenticationError: deps.authenticationError
  })
}

export type WindowsUpdatePhase = 'idle' | 'updating' | 'restoring'

export interface WindowsUpdateState {
  phase: WindowsUpdatePhase
}

export function windowsUpdateIsBusy(state: WindowsUpdateState): boolean {
  return state.phase !== 'idle'
}

/** The owned restoration phase may start a backend, but still excludes UI retries. */
export function windowsUpdateBlocksBackendStart(state: WindowsUpdateState): boolean {
  return state.phase === 'updating'
}

export type PreparedWindowsUpdate<TTransport> =
  | { kind: 'manual'; command: string; hermesRoot: string }
  | { kind: 'handoff'; transport: TTransport; updateRoot: string; branch: string }

export interface WindowsUpdateLaunch<TLaunch> {
  launch: TLaunch
  updater: string
}

export interface WindowsUpdateApplyDeps<TTransport, TLaunch> {
  state: WindowsUpdateState
  hermesHome: string
  prepare: () => Promise<PreparedWindowsUpdate<TTransport>>
  emitProgress: (progress: { stage: string; message: string; percent: number | null }) => void
  preflightStateDb: () => void
  runPreflight: (
    prepared: Extract<PreparedWindowsUpdate<TTransport>, { kind: 'handoff' }>,
    claim: UpdateMarkerClaim
  ) => Promise<UpdatePreflightOutcome>
  stopSafeBlockers: (updateRoot: string, result: VenvBlockerScanResult) => Promise<unknown>
  launch: (
    permit: UpdateMutationPermit,
    prepared: Extract<PreparedWindowsUpdate<TTransport>, { kind: 'handoff' }>,
    claim: UpdateMarkerClaim
  ) => WindowsUpdateLaunch<TLaunch>
  observe: (launch: TLaunch) => Promise<{ ok: boolean; message?: string }>
  authenticate: (launch: TLaunch, claim: UpdateMarkerClaim) => Promise<boolean>
  commit: (launch: WindowsUpdateLaunch<TLaunch>) => void
  waitForMarkerClearance: () => Promise<'clear' | 'finished' | 'timeout'>
  restoreBackends: () => Promise<void>
  log: (line: string) => void
  acquireMarker?: (hermesHome: string) => ReturnType<typeof acquireUpdateMarker>
  releaseMarker?: (hermesHome: string, pid: number, startedAt: number) => boolean
  authorize?: (preflight: UpdatePreflightOutcome) => UpdateMutationPermit | null
  runAuthorized?: <T>(permit: UpdateMutationPermit, operation: () => T) => T
}

export type WindowsUpdateApplyResult =
  | { ok: true; manual: true; command: string; hermesRoot: string }
  | { ok: true; handedOff: true; updater: string }
  | { ok: false; error: string; message: string; blockers?: VenvBlockerScanResult['processes'] }

export async function applyWindowsUpdate<TTransport, TLaunch>(
  opts: { stopSafeBlockers?: boolean },
  deps: WindowsUpdateApplyDeps<TTransport, TLaunch>
): Promise<WindowsUpdateApplyResult> {
  if (windowsUpdateIsBusy(deps.state)) {
    throw new Error('An update is already in progress.')
  }

  deps.state.phase = 'updating'
  let claim: UpdateMarkerClaim | undefined
  let handedOff = false
  let abortRestored = false

  const restoreAfterAbort = async (): Promise<void> => {
    if (!claim || abortRestored) {
      return
    }
    abortRestored = true

    const releaseMarker = deps.releaseMarker ?? releaseUpdateMarkerIfOwnedBy
    const released = releaseMarker(deps.hermesHome, claim.pid, claim.startedAt)
    const clearance = released ? 'clear' : await deps.waitForMarkerClearance()

    if (clearance === 'timeout') {
      return
    }

    // Marker clearance is proven before the gate permits this owned backend
    // restoration. The UI remains busy until restoration finishes.
    deps.state.phase = 'restoring'
    await deps.restoreBackends()
  }

  try {
    const prepared = await deps.prepare()

    if (prepared.kind === 'manual') {
      deps.emitProgress({ stage: 'manual', message: prepared.command, percent: null })

      return { ok: true, manual: true, command: prepared.command, hermesRoot: prepared.hermesRoot }
    }

    const acquireMarker = deps.acquireMarker ?? acquireUpdateMarker
    const acquired = acquireMarker(deps.hermesHome)

    if (acquired.acquired === false) {
      deps.emitProgress({ stage: 'error', message: acquired.message, percent: null })

      return { ok: false, error: 'update-already-running', message: acquired.message }
    }

    claim = acquired.owner
    deps.preflightStateDb()
    deps.emitProgress({
      stage: 'restart',
      message: 'Updating Hermes — this window will close and Hermes will restart when the update finishes.',
      percent: 100
    })

    let preflight = await deps.runPreflight(prepared, claim)

    if (preflight.kind === 'blocked' && preflight.result && opts.stopSafeBlockers) {
      await deps.stopSafeBlockers(prepared.updateRoot, preflight.result)
      preflight = await deps.runPreflight(prepared, claim)
    }

    if (preflight.kind !== 'clear') {
      const error = preflight.kind === 'probe-failure' ? 'venv-probe-failed' : 'venv-blocked'
      deps.log(`[updates] preflight refused: ${preflight.message}`)
      deps.emitProgress({ stage: 'error', message: preflight.message, percent: null })

      return {
        ok: false,
        error,
        message: preflight.message,
        ...(preflight.kind === 'blocked' && preflight.result ? { blockers: preflight.result.processes } : {})
      }
    }

    const authorize = deps.authorize ?? authorizeUpdateMutation
    const permit = authorize(preflight)

    if (!permit) {
      throw new Error('Clear update preflight did not authorize handoff.')
    }

    const runAuthorized = deps.runAuthorized ?? runAuthorizedUpdateMutation
    const launched = runAuthorized(permit, () => deps.launch(permit, prepared, claim!))
    // Attach process observation before authentication can yield.
    const observation = deps.observe(launched.launch)

    const [authenticated, observed] = await Promise.all([deps.authenticate(launched.launch, claim), observation])

    if (!authenticated || !observed.ok) {
      const error = authenticated ? 'updater-spawn-failed' : 'update-handoff-unacknowledged'

      const message = authenticated
        ? observed.message || 'Updater process exited before handoff completed.'
        : 'Update aborted: the updater did not acknowledge the handoff. Retry after any running update finishes.'

      await restoreAfterAbort()
      deps.emitProgress({ stage: 'error', message, percent: null })

      return { ok: false, error, message }
    }

    handedOff = true
    deps.commit(launched)

    return { ok: true, handedOff: true, updater: launched.updater }
  } finally {
    try {
      if (claim && !handedOff && !abortRestored) {
        await restoreAfterAbort()
      }
    } finally {
      deps.state.phase = 'idle'
    }
  }
}
