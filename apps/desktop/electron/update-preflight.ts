import type { UpdateMarkerClaim } from './update-marker'
import {
  type DesktopPluginServiceProcess,
  desktopPluginServiceUnits,
  formatBlockerMessage,
  formatProbeFailedMessage,
  isExactActionableDesktopPluginService,
  isExactActionableMcpBridge,
  type McpBridgeProcess,
  type ScanOutcome,
  type VenvBlockerScanResult
} from './venv-blocker-scan'
import type { ForceReleaseHolder, WindowsUpdateForceReleaseOutcome } from './windows-update-force-release'

export type UpdatePreflightPurpose = 'normal-update' | 'bootstrap-recovery'

export interface UpdatePreflightDeps {
  claim: UpdateMarkerClaim
  ownsUpdateMarker: (claim: UpdateMarkerClaim) => boolean
  now?: () => number
  releaseTrackedBackendTrees: () => Promise<{ unlocked: boolean }>
  /**
   * Non-elevated ≤5s force-release of exact install holders when tracked
   * backends alone did not unlock the install. Optional only for unit tests
   * that intentionally exercise the legacy unlock-failed path.
   */
  forceReleaseInstallHolders?: () => Promise<WindowsUpdateForceReleaseOutcome>
  scan: () => Promise<ScanOutcome>
  terminateDesktopPluginService: (service: DesktopPluginServiceProcess) => Promise<boolean>
  terminateMcpBridge: (bridge: McpBridgeProcess) => Promise<boolean>
  wait?: (delayMs: number) => Promise<void>
}

export interface UpdatePreflightTiming {
  cooperativeExitMs?: number
  genericHolderPollMs?: number
  genericHolderTimeoutMs?: number
  respawnIntervalMs?: number
  terminationSettleMs?: number
}

export type UpdatePreflightOutcome =
  | { kind: 'clear'; claim: UpdateMarkerClaim }
  | {
      kind: 'blocked'
      message: string
      reason: 'unlock-failed' | 'holders' | 'marker-unavailable' | 'quiesce-incomplete' | 'needs-elevation'
      result?: VenvBlockerScanResult
      elevationHolders?: ForceReleaseHolder[]
    }
  | { kind: 'probe-failure'; error: string; message: string }

export interface UpdateMutationPermit {
  readonly preflight: Extract<UpdatePreflightOutcome, { kind: 'clear' }>
}

const successfulPreflightPermits = new WeakMap<object, UpdateMutationPermit>()
const updateMutationPermits = new WeakMap<object, () => boolean>()

export function authorizeUpdateMutation(preflight: UpdatePreflightOutcome): UpdateMutationPermit | null {
  const permit = successfulPreflightPermits.get(preflight) ?? null

  if (!permit) {return null}
  successfulPreflightPermits.delete(preflight)

  return permit
}

export function runAuthorizedUpdateMutation<T>(permit: UpdateMutationPermit, operation: () => T): T {
  if (!updateMutationPermits.has(permit)) {
    throw new Error('update mutation requires a clear-preflight permit')
  }

  if (updateMutationPermits.get(permit)?.() !== true) {
    throw new Error('update mutation requires the original update marker claim')
  }

  return operation()
}

const DEFAULT_COOPERATIVE_EXIT_MS = 1_500
const DEFAULT_GENERIC_HOLDER_POLL_MS = 1_000
const DEFAULT_GENERIC_HOLDER_TIMEOUT_MS = 30_000
const DEFAULT_RESPAWN_INTERVAL_MS = 1_500
const DEFAULT_TERMINATION_SETTLE_MS = 750
const MAX_FALLBACK_DRAIN_GROUPS = 32
const MAX_FALLBACK_DRAIN_RECORDS = 64

function wait(delayMs: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, delayMs))
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function exactDrainableOnly(result: VenvBlockerScanResult): boolean {
  return (
    result.processes.length === 0 &&
    result.mcpBridges.every(isExactActionableMcpBridge) &&
    result.desktopPluginServices.every(isExactActionableDesktopPluginService) &&
    result.mcpBridges.length + result.desktopPluginServices.length > 0
  )
}

function logicalDrainGroupCount(processes: ReadonlyArray<{ pid: number; wrapperPid?: number }>): number {
  return new Set(processes.map(process => process.wrapperPid ?? process.pid)).size
}

function genericHoldersOnly(result: VenvBlockerScanResult): boolean {
  return result.processes.length > 0 && result.mcpBridges.length === 0
}

function quiesceIncompleteMessage(): string {
  return (
    'Update aborted: processes from this Hermes installation did not remain stopped. ' +
    'Close or stop the restarting service, then retry.'
  )
}

export async function runWindowsUpdatePreflight(
  _purpose: UpdatePreflightPurpose,
  deps: UpdatePreflightDeps,
  timing: UpdatePreflightTiming = {}
): Promise<UpdatePreflightOutcome> {
  const sleep = deps.wait ?? wait
  const now = deps.now ?? Date.now
  const cooperativeExitMs = timing.cooperativeExitMs ?? DEFAULT_COOPERATIVE_EXIT_MS
  const genericHolderPollMs = Math.max(1, timing.genericHolderPollMs ?? DEFAULT_GENERIC_HOLDER_POLL_MS)
  const genericHolderTimeoutMs = Math.max(0, timing.genericHolderTimeoutMs ?? DEFAULT_GENERIC_HOLDER_TIMEOUT_MS)
  const respawnIntervalMs = timing.respawnIntervalMs ?? DEFAULT_RESPAWN_INTERVAL_MS
  const terminationSettleMs = timing.terminationSettleMs ?? DEFAULT_TERMINATION_SETTLE_MS

  // The caller claims the shared updater marker before entering preflight,
  // and remains responsible for release on every unsuccessful outcome.
  const claim = Object.freeze({ ...deps.claim })

  const ownsClaim = (): boolean => {
    try {
      return Number.isSafeInteger(claim.pid) && claim.pid > 0 &&
        Number.isSafeInteger(claim.startedAt) && claim.startedAt > 0 &&
        deps.ownsUpdateMarker(claim) === true
    } catch {
      return false
    }
  }

  const requireClaim = (): void => {
    if (!ownsClaim()) { throw new Error('the original update marker claim is no longer owned') }
  }

  if (!ownsClaim()) {
    return {
      kind: 'blocked', reason: 'marker-unavailable',
      message: 'Update aborted: Hermes no longer owns the update marker. Retry after any other update finishes.'
    }
  }

  let lock: { unlocked: boolean }

  try {
    lock = await deps.releaseTrackedBackendTrees()
  } catch {
    lock = { unlocked: false }
  }

  // Selecting Update authorizes a bounded force-release of exact install
  // holders. Do not dead-end on the tracked-backend unlock wait alone — that
  // is how a stale `hermes tools | head` tree blocked the scanner forever.
  if (lock?.unlocked !== true) {
    if (typeof deps.forceReleaseInstallHolders === 'function') {
      let forceOutcome: WindowsUpdateForceReleaseOutcome

      try {
        requireClaim()
        forceOutcome = await deps.forceReleaseInstallHolders()
      } catch (error) {
        return {
          kind: 'probe-failure',
          error: `force-release threw: ${errorText(error)}`,
          message: formatProbeFailedMessage()
        }
      }

      if (forceOutcome.kind === 'clear') {
        lock = { unlocked: true }
      } else if (forceOutcome.kind === 'needs-elevation') {
        return {
          kind: 'blocked',
          reason: 'needs-elevation',
          message: forceOutcome.message,
          elevationHolders: forceOutcome.holders
        }
      } else {
        return {
          kind: 'blocked',
          reason: 'unlock-failed',
          message: forceOutcome.message
        }
      }
    } else {
      return {
        kind: 'blocked',
        reason: 'unlock-failed',
        message:
          'Update aborted: Hermes could not release its tracked local processes. ' +
          'Close other Hermes windows and terminals, then retry.'
      }
    }
  }

  // Every scan and destructive step remains under the same update claim.
  const scanFailClosed = async (): Promise<ScanOutcome> => {
    try {
      requireClaim()

      return await deps.scan()
    } catch (error) {
      return { kind: 'probe-failure', error: `scanner threw: ${errorText(error)}` }
    }
  }

  const forceReleaseAndRescan = async (
    blockedResult: VenvBlockerScanResult
  ): Promise<{ scan: ScanOutcome } | { outcome: UpdatePreflightOutcome }> => {
    if (typeof deps.forceReleaseInstallHolders !== 'function') {
      return {
        outcome: {
          kind: 'blocked',
          reason: 'holders',
          result: blockedResult,
          message: formatBlockerMessage(blockedResult)
        }
      }
    }

    let forced: WindowsUpdateForceReleaseOutcome

    try {
      requireClaim()
      forced = await deps.forceReleaseInstallHolders()
    } catch (error) {
      return {
        outcome: {
          kind: 'probe-failure',
          error: `force-release threw: ${errorText(error)}`,
          message: formatProbeFailedMessage()
        }
      }
    }

    if (forced.kind === 'needs-elevation') {
      return {
        outcome: {
          kind: 'blocked',
          reason: 'needs-elevation',
          message: forced.message,
          elevationHolders: forced.holders
        }
      }
    }

    if (forced.kind !== 'clear') {
      return {
        outcome: {
          kind: 'blocked',
          reason: 'unlock-failed',
          result: blockedResult,
          message: forced.message
        }
      }
    }

    return { scan: await scanFailClosed() }
  }

  const pollGenericHoldersUntil = async (current: ScanOutcome, deadline: number): Promise<ScanOutcome> => {
    let outcome = current

    while (outcome.kind === 'blocked' && genericHoldersOnly(outcome.result)) {
      const remainingMs = deadline - now()

      if (remainingMs <= 0) {
        break
      }

      await sleep(Math.min(genericHolderPollMs, remainingMs))
      outcome = await scanFailClosed()
    }

    return outcome
  }

  let observed = await scanFailClosed()

  if (observed.kind === 'probe-failure') {
    return { kind: 'probe-failure', error: observed.error, message: formatProbeFailedMessage() }
  }

  if (observed.kind === 'blocked' && !exactDrainableOnly(observed.result)) {
    const forced = await forceReleaseAndRescan(observed.result)

    if ('outcome' in forced) {
      return forced.outcome
    }

    observed = forced.scan

    if (observed.kind === 'probe-failure') {
      return { kind: 'probe-failure', error: observed.error, message: formatProbeFailedMessage() }
    }

    if (observed.kind === 'blocked' && !exactDrainableOnly(observed.result)) {
      return {
        kind: 'blocked',
        reason: 'holders',
        result: observed.result,
        message: formatBlockerMessage(observed.result)
      }
    }
  }

  try {
    await sleep(cooperativeExitMs)
    let firstClear = await scanFailClosed()
    let genericHolderDeadline: number | null = null

    if (firstClear.kind === 'probe-failure') {
      return { kind: 'probe-failure', error: firstClear.error, message: formatProbeFailedMessage() }
    }

    if (
      firstClear.kind === 'blocked' &&
      genericHoldersOnly(firstClear.result) &&
      !exactDrainableOnly(firstClear.result)
    ) {
      if (typeof deps.forceReleaseInstallHolders === 'function') {
        const forced = await forceReleaseAndRescan(firstClear.result)

        if ('outcome' in forced) {
          return forced.outcome
        }

        firstClear = forced.scan
      } else {
        genericHolderDeadline = now() + genericHolderTimeoutMs
        firstClear = await pollGenericHoldersUntil(firstClear, genericHolderDeadline)
      }

      if (firstClear.kind === 'probe-failure') {
        return { kind: 'probe-failure', error: firstClear.error, message: formatProbeFailedMessage() }
      }

      if (firstClear.kind === 'blocked' && genericHoldersOnly(firstClear.result)) {
        return {
          kind: 'blocked',
          reason: 'quiesce-incomplete',
          result: firstClear.result,
          message: formatBlockerMessage(firstClear.result)
        }
      }
    }

    if (firstClear.kind === 'blocked') {
      const forceableMcpAndDesktopServices =
        firstClear.result.mcpBridges.every(isExactActionableMcpBridge) &&
        firstClear.result.desktopPluginServices.every(isExactActionableDesktopPluginService)

      const naturallyExitingGenericHolders = firstClear.result.processes.length > 0

      const exactCurrentHolders =
        exactDrainableOnly(firstClear.result) ||
        (forceableMcpAndDesktopServices && naturallyExitingGenericHolders)

      const logicalDrainGroups =
        logicalDrainGroupCount(firstClear.result.mcpBridges) +
        logicalDrainGroupCount(firstClear.result.desktopPluginServices)

      const drainRecordCount =
        firstClear.result.processes.length +
        firstClear.result.mcpBridges.length +
        firstClear.result.desktopPluginServices.length

      if (
        !exactCurrentHolders ||
        logicalDrainGroups > MAX_FALLBACK_DRAIN_GROUPS ||
        drainRecordCount > MAX_FALLBACK_DRAIN_RECORDS
      ) {
        return {
          kind: 'blocked',
          reason: 'quiesce-incomplete',
          result: firstClear.result,
          message: exactDrainableOnly(firstClear.result)
            ? quiesceIncompleteMessage()
            : formatBlockerMessage(firstClear.result)
        }
      }

      // One bounded force-stop pass. The user already chose Update. Each call
      // re-scans the target installation and verifies PID/create-time before
      // stopping that single holder; there is deliberately no process-tree kill.
      const terminationOrder = [...firstClear.result.mcpBridges].sort(
        (left, right) => Number(right.role === 'mcp_bridge_worker') - Number(left.role === 'mcp_bridge_worker')
      )

      // A managed-runtime worker can derive its exact identity from its live
      // wrapper ancestry. Drain workers before wrappers and await each one so
      // the wrapper cannot disappear before the scanner revalidates a worker.
      for (const bridge of terminationOrder) {
        try {
          requireClaim()

          if (!(await deps.terminateMcpBridge(bridge))) {
            return {
              kind: 'blocked',
              reason: 'quiesce-incomplete',
              result: firstClear.result,
              message: quiesceIncompleteMessage()
            }
          }
        } catch {
          return {
            kind: 'blocked',
            reason: 'quiesce-incomplete',
            result: firstClear.result,
            message: quiesceIncompleteMessage()
          }
        }
      }

      // One stop per plugin service UNIT. The scanner stops the proven
      // supervisor first, then the venv wrapper, then the worker(s), so a
      // second call for the other member would find nothing to prove and
      // report false. (Worker-first, two-call draining left the Windows
      // Script Host loop alive to respawn the service ten seconds later.)
      for (const service of desktopPluginServiceUnits(firstClear.result.desktopPluginServices)) {
        try {
          requireClaim()

          if (!(await deps.terminateDesktopPluginService(service))) {
            return {
              kind: 'blocked',
              reason: 'quiesce-incomplete',
              result: firstClear.result,
              message: quiesceIncompleteMessage()
            }
          }
        } catch {
          return {
            kind: 'blocked',
            reason: 'quiesce-incomplete',
            result: firstClear.result,
            message: quiesceIncompleteMessage()
          }
        }
      }

      // A scheduled gateway-status or presence probe can start after the
      // first scan and briefly share this venv with an exact MCP bridge. A
      // holder without an identity timestamp cannot be force-stopped, so give
      // it one bounded chance to exit before refusing. The deadline includes
      // the first post-termination settle scan.
      genericHolderDeadline ??= now() + genericHolderTimeoutMs

      await sleep(terminationSettleMs)
      firstClear = await scanFailClosed()

      if (firstClear.kind === 'probe-failure') {
        return { kind: 'probe-failure', error: firstClear.error, message: formatProbeFailedMessage() }
      }

      if (
        firstClear.kind === 'blocked' &&
        genericHoldersOnly(firstClear.result) &&
        typeof deps.forceReleaseInstallHolders === 'function'
      ) {
        const forced = await forceReleaseAndRescan(firstClear.result)

        if ('outcome' in forced) {
          return forced.outcome
        }

        firstClear = forced.scan
      } else {
        firstClear = await pollGenericHoldersUntil(firstClear, genericHolderDeadline)
      }

      if (firstClear.kind === 'probe-failure') {
        return { kind: 'probe-failure', error: firstClear.error, message: formatProbeFailedMessage() }
      }

      if (firstClear.kind === 'blocked') {
        return {
          kind: 'blocked',
          reason: 'quiesce-incomplete',
          result: firstClear.result,
          message:
            firstClear.result.mcpBridges.length > 0
              ? quiesceIncompleteMessage()
              : formatBlockerMessage(firstClear.result)
        }
      }
    }

    // Bound repeated force-release attempts across stability scans. The
    // non-force path starts its existing grace period when a holder appears.
    if (typeof deps.forceReleaseInstallHolders === 'function') {
      genericHolderDeadline ??= now() + genericHolderTimeoutMs
    }

    while (true) {
      await sleep(respawnIntervalMs)
      let secondClear = await scanFailClosed()

      if (secondClear.kind === 'probe-failure') {
        return { kind: 'probe-failure', error: secondClear.error, message: formatProbeFailedMessage() }
      }

      if (secondClear.kind === 'blocked' && genericHoldersOnly(secondClear.result)) {
        if (typeof deps.forceReleaseInstallHolders === 'function') {
          if (genericHolderDeadline !== null && now() >= genericHolderDeadline) {
            return {
              kind: 'blocked',
              reason: 'quiesce-incomplete',
              result: secondClear.result,
              message: quiesceIncompleteMessage()
            }
          }

          const forced = await forceReleaseAndRescan(secondClear.result)

          if ('outcome' in forced) {
            return forced.outcome
          }

          secondClear = forced.scan
        } else {
          genericHolderDeadline ??= now() + genericHolderTimeoutMs
          secondClear = await pollGenericHoldersUntil(secondClear, genericHolderDeadline)
        }

        if (secondClear.kind === 'probe-failure') {
          return { kind: 'probe-failure', error: secondClear.error, message: formatProbeFailedMessage() }
        }

        if (secondClear.kind === 'clear') {
          // The holder exited, but the last clear observation has not yet
          // survived a full stability interval.
          continue
        }
      }

      if (secondClear.kind === 'blocked') {
        return {
          kind: 'blocked',
          reason: 'quiesce-incomplete',
          result: secondClear.result,
          message: exactDrainableOnly(secondClear.result)
            ? quiesceIncompleteMessage()
            : formatBlockerMessage(secondClear.result)
        }
      }

      break
    }

    requireClaim()

    const outcome: Extract<UpdatePreflightOutcome, { kind: 'clear' }> = Object.freeze({
      kind: 'clear',
      claim
    })

    const permit: UpdateMutationPermit = Object.freeze({ preflight: outcome })
    successfulPreflightPermits.set(outcome, permit)
    updateMutationPermits.set(permit, ownsClaim)

    return outcome
  } catch (error) {
    const detail = `preflight transaction failed: ${errorText(error)}`

    return { kind: 'probe-failure', error: detail, message: formatProbeFailedMessage() }
  }
}
