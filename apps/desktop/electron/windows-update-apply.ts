import { acquireUpdateMarker, releaseUpdateMarkerIfOwnedBy, type UpdateMarkerClaim } from './update-marker'
import {
  authorizeUpdateMutation,
  runAuthorizedUpdateMutation,
  type UpdateMutationPermit,
  type UpdatePreflightOutcome
} from './update-preflight'
import type { VenvBlockerScanResult } from './venv-blocker-scan'

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
