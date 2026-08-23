/**
 * Desktop self-update store. Tracks distance from the configured branch,
 * surfaces it as an ambient pill, and orchestrates the apply flow.
 */

import { atom } from 'nanostores'

import type {
  DesktopUpdateApplyOptions,
  DesktopUpdateApplyResult,
  DesktopUpdateBlocker,
  DesktopUpdateProgress,
  DesktopUpdateStage,
  DesktopUpdateStatus,
  DesktopVersionInfo
} from '@/global'
import { checkHermesUpdate, getActionStatus, updateHermes } from '@/hermes'
import { translateNow } from '@/i18n'
import { persistString, storedString } from '@/lib/storage'
import { $connectionsRegistry, refreshConnectionsRegistry } from '@/store/connections'
import { dismissNotification, notify } from '@/store/notifications'
import { $connection } from '@/store/session'
import type { BackendUpdateCheckResponse } from '@/types/hermes'

export interface UpdateApplyState {
  applying: boolean
  stage: DesktopUpdateStage
  message: string
  percent: number | null
  error: string | null
  /** When the stage is 'manual': the exact command the user should run
   *  (CLI install with no staged updater). */
  command: string | null
  /** Structured update blockers used by the safe close-and-update confirmation. */
  blockers?: readonly DesktopUpdateBlocker[] | null
  log: readonly { stage: DesktopUpdateStage; message: string; at: number }[]
}

const IDLE: UpdateApplyState = {
  applying: false,
  stage: 'idle',
  message: '',
  percent: null,
  error: null,
  command: null,
  log: []
}

export const $desktopVersion = atom<DesktopVersionInfo | null>(null)
export const $updateApply = atom<UpdateApplyState>(IDLE)
export const $updateChecking = atom<boolean>(false)
export const $updateOverlayOpen = atom<boolean>(false)
export const $updateStatus = atom<DesktopUpdateStatus | null>(null)

// Client and backend are independently updatable; each keeps its own state.
export const $backendUpdateStatus = atom<DesktopUpdateStatus | null>(null)
export const $backendUpdateApply = atom<UpdateApplyState>(IDLE)
export const $backendUpdateChecking = atom<boolean>(false)

export type UpdateTarget = 'client' | 'backend'
export const $updateOverlayTarget = atom<UpdateTarget>('client')

export const setUpdateOverlayOpen = (open: boolean) => $updateOverlayOpen.set(open)

export const openUpdateOverlayFor = (target: UpdateTarget) => {
  $updateOverlayTarget.set(target)
  $updateOverlayOpen.set(true)
  void (target === 'backend' ? checkBackendUpdates() : checkUpdates())
}

export const resetUpdateApplyState = () => {
  $updateApply.set(IDLE)
  $backendUpdateApply.set(IDLE)
}

const UPDATE_TOAST_ID = 'desktop-update-available'
// Time-based snooze instead of per-sha dismissal: this repo lands ~100 commits
// a day, so a "don't show this exact sha again" guard re-popped the toast on
// every new commit. We instead suppress the toast for a cooldown window that
// (re)starts whenever the user closes it.
const UPDATE_TOAST_SNOOZE_KEY = 'hermes:update-toast-snooze-until'
const UPDATE_TOAST_COOLDOWN_MS = 24 * 60 * 60 * 1000

function snoozeUpdateToast(): void {
  persistString(UPDATE_TOAST_SNOOZE_KEY, String(Date.now() + UPDATE_TOAST_COOLDOWN_MS))
}

function isUpdateToastSnoozed(): boolean {
  const until = Number(storedString(UPDATE_TOAST_SNOOZE_KEY) || 0)

  return Number.isFinite(until) && Date.now() < until
}

// Must match tui_gateway's DESKTOP_BACKEND_CONTRACT that this build was written
// against. The backend reports its own value in session runtime info; a lower
// value (or none — a pre-GUI checkout) means GUI<->backend skew.
// v2: requires the file.attach RPC (remote-gateway non-image file upload).
// v3: requires approvals.mode config RPCs and session.info reconciliation.
// v4: requires explicit Fast-off session creation and session-scoped Fast edits.
// v5: requires raised WebSocket frame size for large one-shot file.attach.
// v6: requires key-addressed plugins.manage rows (keyless rows render
//     read-only in Settings → Plugins).
const REQUIRED_BACKEND_CONTRACT = 6
const SKEW_TOAST_ID = 'backend-contract-skew'
// The contract check runs on every session.resume (applyRuntimeInfo), so
// without a snooze the warning re-popped on every thread the user opened, even
// right after they closed it. Mirror the update toast: persist a cooldown when
// the user dismisses it. It still reminds again after the window if the backend
// is still behind, and clears immediately once the backend catches up.
const SKEW_TOAST_SNOOZE_KEY = 'hermes:backend-skew-toast-snooze-until'
const SKEW_TOAST_COOLDOWN_MS = 24 * 60 * 60 * 1000

function snoozeSkewToast(): void {
  persistString(SKEW_TOAST_SNOOZE_KEY, String(Date.now() + SKEW_TOAST_COOLDOWN_MS))
}

function isSkewToastSnoozed(): boolean {
  const until = Number(storedString(SKEW_TOAST_SNOOZE_KEY) || 0)

  return Number.isFinite(until) && Date.now() < until
}

const INSTALL_METHOD_TOAST_ID = 'install-method-not-supported'
// Same time-based snooze pattern as the update/skew toasts: the warning is
// re-derived from every session.info (session.create/resume/activate all
// route through applyRuntimeInfo), so without a snooze it would re-pop on
// every session switch even right after the user dismissed it.
const INSTALL_METHOD_TOAST_SNOOZE_KEY = 'hermes:install-method-toast-snooze-until'
const INSTALL_METHOD_TOAST_COOLDOWN_MS = 24 * 60 * 60 * 1000

function snoozeInstallMethodToast(): void {
  persistString(INSTALL_METHOD_TOAST_SNOOZE_KEY, String(Date.now() + INSTALL_METHOD_TOAST_COOLDOWN_MS))
}

function isInstallMethodToastSnoozed(): boolean {
  const until = Number(storedString(INSTALL_METHOD_TOAST_SNOOZE_KEY) || 0)

  return Number.isFinite(until) && Date.now() < until
}

/**
 * Guard against a desktop GUI talking to a backend that predates its contract
 * (e.g. a bb/gui-built app pointed at a `main` checkout). Rather than failing
 * cryptically downstream, surface a warning with a one-click align that runs
 * the normal update flow (which self-heals to the right branch).
 *
 * Runs on every session open; closing the toast snoozes it for a cooldown so it
 * doesn't nag on every thread switch.
 */
export function reportBackendContract(contract: number | undefined): void {
  if ((contract ?? 0) >= REQUIRED_BACKEND_CONTRACT) {
    dismissNotification(SKEW_TOAST_ID)
    // Backend caught up — forget any prior snooze so a future regression warns
    // immediately rather than staying silent for the rest of the window.
    persistString(SKEW_TOAST_SNOOZE_KEY, null)

    return
  }

  if (isSkewToastSnoozed()) {
    return
  }

  notify({
    action: {
      label: translateNow('notifications.updateHermes'),
      onClick: () => {
        snoozeSkewToast()
        void applyBackendUpdate()
      }
    },
    durationMs: 0,
    id: SKEW_TOAST_ID,
    kind: 'warning',
    message: translateNow('notifications.backendOutOfDateMessage'),
    onDismiss: () => snoozeSkewToast(),
    title: translateNow('notifications.backendOutOfDateTitle')
  })
}

export function reportInstallMethodWarning(message: string | undefined): void {
  if (!message) {
    dismissNotification(INSTALL_METHOD_TOAST_ID)

    return
  }

  if (isInstallMethodToastSnoozed()) {
    return
  }

  notify({
    durationMs: 0,
    id: INSTALL_METHOD_TOAST_ID,
    kind: 'warning',
    message,
    onDismiss: () => snoozeInstallMethodToast(),
    title: translateNow('notifications.installMethodUnsupportedTitle')
  })
}

/**
 * Fire a toast when an update is available, at most once per cooldown window.
 * Closing the toast — dismissing it or opening the updates window from it —
 * (re)starts the cooldown, so a busy upstream branch doesn't re-spam the user
 * on every new commit. The snooze is persisted, so it survives relaunches too.
 */
export function maybeNotifyUpdateAvailable(status: DesktopUpdateStatus | null) {
  if (!status || status.supported === false || status.error || !status.targetSha) {
    return
  }

  const behind = typeof status.behind === 'number' ? status.behind : null

  // behind === null means "update available, exact count unknown" (shallow
  // clone). That still deserves the toast — just with count-free copy.
  if ((behind ?? 0) <= 0 && !status.updateAvailable) {
    return
  }

  if (isUpdateToastSnoozed()) {
    return
  }

  if ($updateApply.get().applying) {
    return
  }

  notify({
    action: {
      label: translateNow('notifications.seeWhatsNew'),
      onClick: () => {
        snoozeUpdateToast()
        openUpdatesWindow()
      }
    },
    durationMs: 0,
    icon: 'gift',
    id: UPDATE_TOAST_ID,
    kind: 'info',
    message:
      behind !== null && behind > 0
        ? translateNow('notifications.updateReadyMessage', behind)
        : translateNow('notifications.updateReadyMessageUnknown'),
    onDismiss: () => snoozeUpdateToast(),
    title: translateNow('notifications.updateReadyTitle')
  })
}

export function openUpdatesWindow(): void {
  openUpdateOverlayFor(isRemoteMode() ? 'backend' : 'client')
}

/**
 * Start applying the available update for the active target right away. Opens
 * the updates overlay first so the user sees apply progress (the overlay
 * renders ApplyingView once `applying` flips true), then kicks off the install.
 * Used by the "Update now" affordance on the About panel, which would otherwise
 * only be able to open the changelog overlay.
 *
 * Multi-target installs (remote mode / multi-connection registry) route
 * through the everything-flow so "update" means every machine, not just the
 * active target — the single-target ternary is what left remote-mode users
 * updating the backend forever while the GUI itself went stale.
 */
export function startActiveUpdate(): void {
  if (hasMultipleUpdateTargets()) {
    $updateOverlayOpen.set(true)
    void applyEverythingUpdate()

    return
  }

  const target: UpdateTarget = isRemoteMode() ? 'backend' : 'client'
  $updateOverlayTarget.set(target)
  $updateOverlayOpen.set(true)
  void (target === 'backend' ? applyBackendUpdate() : applyUpdates())
}

/**
 * Command-palette entry point. The About panel's "Update now" only renders once
 * we know an update is waiting; this row is always listed, so it also has to
 * handle "already current" — open the overlay for the active target and let its
 * check answer, and only apply when there's something to install. On
 * multi-target installs an update waiting on EITHER the client or the backend
 * triggers the everything-flow.
 */
export function requestActiveUpdate(): void {
  if (hasMultipleUpdateTargets()) {
    const clientStatus = $updateStatus.get()
    const backendStatus = $backendUpdateStatus.get()

    const anyBehind =
      (clientStatus?.behind ?? 0) > 0 ||
      clientStatus?.updateAvailable ||
      (backendStatus?.behind ?? 0) > 0 ||
      backendStatus?.updateAvailable

    if (anyBehind) {
      startActiveUpdate()

      return
    }
  }

  const target: UpdateTarget = isRemoteMode() ? 'backend' : 'client'
  const status = target === 'backend' ? $backendUpdateStatus.get() : $updateStatus.get()

  if ((status?.behind ?? 0) > 0 || status?.updateAvailable) {
    startActiveUpdate()

    return
  }

  openUpdateOverlayFor(target)
}

/** Re-read the running app's version from the Electron main process and
 *  publish it on `$desktopVersion`. Called when the About panel mounts, the
 *  update flow finishes, and the window regains focus, so the About text
 *  stays in sync with the just-installed binary instead of frozen at the
 *  value captured at first-load. */
export async function refreshDesktopVersion(): Promise<DesktopVersionInfo | null> {
  if (typeof window === 'undefined') {
    return null
  }

  // Best-effort UI sync: callers (checkUpdates, startUpdatePoller, window
  // focus handler) all kick this off with `void refreshDesktopVersion()`,
  // so any rejection from the IPC bridge (e.g. main process shutting down
  // mid-reload, or the bridge not yet ready on first paint) would surface
  // as an unhandled promise rejection in the renderer. Swallow it.
  try {
    const next = await window.hermesDesktop?.getVersion?.()

    if (next) {
      $desktopVersion.set(next)
    }

    return next ?? null
  } catch {
    return null
  }
}

function isRemoteMode(): boolean {
  return $connection.get()?.mode === 'remote'
}

function mapBackendCheck(res: BackendUpdateCheckResponse): DesktopUpdateStatus {
  const behind = res.behind ?? 0

  return {
    supported: res.can_apply,
    message: res.message ?? undefined,
    updateAvailable: res.update_available,
    behind: behind > 0 ? behind : 0,
    currentVersion: res.current_version,
    targetSha: res.update_available ? `backend:${res.current_version}` : undefined,
    commits: res.commits,
    fetchedAt: Date.now()
  }
}

export async function checkBackendUpdates(): Promise<DesktopUpdateStatus | null> {
  if (!isRemoteMode() || $backendUpdateChecking.get()) {
    return $backendUpdateStatus.get()
  }

  $backendUpdateChecking.set(true)

  try {
    const status = mapBackendCheck(await checkHermesUpdate(true))
    $backendUpdateStatus.set(status)
    maybeNotifyUpdateAvailable(status)

    return status
  } catch (error) {
    const fallback: DesktopUpdateStatus = {
      supported: $backendUpdateStatus.get()?.supported ?? true,
      error: 'check-failed',
      message: error instanceof Error ? error.message : String(error),
      fetchedAt: Date.now()
    }

    $backendUpdateStatus.set(fallback)

    return fallback
  } finally {
    $backendUpdateChecking.set(false)
  }
}

export async function checkUpdates(): Promise<DesktopUpdateStatus | null> {
  const bridge = window.hermesDesktop?.updates

  if (!bridge || $updateChecking.get()) {
    return $updateStatus.get()
  }

  $updateChecking.set(true)

  try {
    const status = await bridge.check()
    $updateStatus.set(status)
    maybeNotifyUpdateAvailable(status)
    void refreshDesktopVersion()

    return status
  } catch (error) {
    const previous = $updateStatus.get()

    const fallback: DesktopUpdateStatus = {
      supported: previous?.supported ?? true,
      branch: previous?.branch,
      error: 'check-failed',
      message: error instanceof Error ? error.message : String(error),
      fetchedAt: Date.now()
    }

    $updateStatus.set(fallback)

    return fallback
  } finally {
    $updateChecking.set(false)
  }
}

export async function applyUpdates(opts: DesktopUpdateApplyOptions = {}): Promise<DesktopUpdateApplyResult> {
  const bridge = window.hermesDesktop?.updates

  if (!bridge) {
    return { ok: false, error: 'unavailable', message: 'Desktop bridge unavailable.' }
  }

  dismissNotification(UPDATE_TOAST_ID)
  $updateApply.set({ ...IDLE, applying: true, stage: 'prepare', message: 'Starting update…' })

  try {
    const result = await bridge.apply(opts)

    // CLI install with no staged updater: not an error — the user just runs
    // `hermes update` themselves. Land on a dedicated manual state so the
    // overlay shows the command + copy button instead of a dead retry loop.
    if (result?.manual) {
      $updateApply.set({
        ...IDLE,
        applying: false,
        stage: 'manual',
        message: result.command ?? 'hermes update',
        command: result.command ?? 'hermes update'
      })

      return result
    }

    // A detached relauncher took over (macOS bundle swap / Linux re-exec): the
    // app is about to quit and reopen, so hold the "Restarting…" view until it
    // does. Every other resolved outcome MUST land on a terminal, closeable
    // state: the apply IPC resolves here, but the progress stream may have left
    // us on a non-terminal stage (e.g. 'done'/'rebuild'), which renders as a
    // spinner with no close button — the exact hang this guards against.
    // Linux GUI/backend skew (#45205): the backend was updated but the running
    // desktop app PACKAGE was not changed (AppImage/.deb/.rpm). We must NOT tell
    // the user "the new version loads next launch" — that's false; this packaged
    // shell keeps running old GUI code against the new backend. Land on the
    // dedicated, closeable guiSkew terminal state telling them to update/reinstall
    // the desktop app.
    if (result?.guiSkew) {
      $updateApply.set({
        ...IDLE,
        applying: false,
        stage: 'guiSkew',
        message: result.message ?? translateNow('updates.guiSkewBody')
      })

      return result
    }

    // Backend updated but the app couldn't auto-relaunch (e.g. the rebuilt
    // sandbox helper isn't launchable): keep a closeable manual-restart state so
    // the user keeps a working window instead of a dead app or a stuck spinner.
    if (result?.ok && result?.manualRestart) {
      $updateApply.set({
        ...IDLE,
        applying: false,
        stage: 'manual',
        message: result.message ?? translateNow('updates.manualPickedUp')
      })

      return result
    }

    if (!result?.handedOff) {
      if (result?.ok) {
        // Updated, but couldn't relaunch in place (AppImage / dev run). Dismiss
        // the overlay and let the user know the new version loads next launch
        // rather than stranding them on an un-closeable spinner.
        setUpdateOverlayOpen(false)
        resetUpdateApplyState()
        notify({
          durationMs: 8000,
          id: UPDATE_TOAST_ID,
          kind: 'success',
          message: translateNow('updates.manualPickedUp'),
          // No action button here, but it's still update-lifecycle news — keep
          // it with the other update toasts instead of the ambient bottom-right
          // stack.
          placement: 'default',
          title: translateNow('updates.allSetTitle')
        })
      } else {
        $updateApply.set({
          ...$updateApply.get(),
          applying: false,
          stage: 'error',
          error: result?.error ?? 'apply-failed',
          message: result?.message ?? translateNow('updates.errorBody'),
          blockers: result?.blockers ?? null
        })
      }
    }

    return result
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    $updateApply.set({ ...$updateApply.get(), applying: false, stage: 'error', error: 'apply-failed', message })

    return { ok: false, error: 'apply-failed', message }
  }
}

const BACKEND_ACTION_POLL_MS = 1500
const BACKEND_ACTION_MAX_MS = 6 * 60 * 1000
const BACKEND_RETURN_MAX_MS = 4 * 60 * 1000

function finishBackendApply(returned: boolean): DesktopUpdateApplyResult {
  if (returned) {
    $backendUpdateApply.set(IDLE)
    setUpdateOverlayOpen(false)
    void checkBackendUpdates()
    // The backend caught up, but the CLIENT may still be behind — the exact
    // gap that strands remote-mode users on an old GUI forever (every update
    // affordance in remote mode targets the backend, so nothing ever told
    // them the app itself was stale). Nudge with a one-click client update.
    void maybeNudgeClientAfterBackendUpdate()

    return { ok: true, message: 'Backend update applied.' }
  }

  $backendUpdateApply.set({
    ...$backendUpdateApply.get(),
    applying: false,
    stage: 'error',
    error: 'apply-failed',
    message: translateNow('updates.applyStatus.noReturn')
  })

  return { ok: false, error: 'apply-failed', message: 'Backend did not come back online.' }
}

function ingestBackendActionStatus(status: Awaited<ReturnType<typeof getActionStatus>>): void {
  const current = $backendUpdateApply.get()

  const log = status.lines
    .filter(line => line.trim().length > 0)
    .map(line => ({ at: Date.now(), message: line, stage: current.stage }))
    .slice(-50)

  const latest = log.at(-1)?.message

  if (log.length === 0 && !latest) {
    return
  }

  $backendUpdateApply.set({
    ...current,
    log,
    message: latest ?? current.message
  })
}

function completedAfterRestart(
  status: Awaited<ReturnType<typeof getActionStatus>>,
  actionId: string | undefined
): boolean {
  return !!actionId && status.lines.some(line => line === `=== hermes-update completed ${actionId} ===`)
}

/** Whether the durable update receipt attached to the status proves the
 *  outcome of THIS apply (#91277 bullet 3). Only a finished receipt whose
 *  run started at-or-after we kicked the update off counts — an older
 *  receipt describes a previous update, and a still-running one proves
 *  nothing yet. The 60s slack absorbs client/backend clock skew. */
function receiptProvesOutcome(status: Awaited<ReturnType<typeof getActionStatus>>, applyStartedAtMs: number): boolean {
  const receipt = status.receipt

  if (!receipt || !receipt.finished_at || !receipt.started_at) {
    return false
  }

  if (receipt.outcome !== 'success' && receipt.outcome !== 'partial' && receipt.outcome !== 'failed') {
    return false
  }

  const startedMs = Date.parse(receipt.started_at)

  return Number.isFinite(startedMs) && startedMs >= applyStartedAtMs - 60_000
}

function legacyBackendReachedTarget(
  status: BackendUpdateCheckResponse,
  targetSha: string | undefined,
  previousVersion: string | undefined
): boolean {
  if (status.behind === 0) {
    return true
  }

  if (previousVersion && status.current_version !== previousVersion) {
    return true
  }

  return !!targetSha && !!status.commits?.length && !status.commits.some(commit => commit.sha === targetSha)
}

let backendUpdateInFlight: Promise<DesktopUpdateApplyResult> | null = null

async function runBackendUpdate(): Promise<DesktopUpdateApplyResult> {
  dismissNotification(UPDATE_TOAST_ID)
  $backendUpdateApply.set({
    ...IDLE,
    applying: true,
    stage: 'prepare',
    message: translateNow('updates.applyStatus.preparing')
  })

  try {
    const previousStatus = $backendUpdateStatus.get()
    const requestedTargetSha = previousStatus?.commits?.at(0)?.sha

    const previousVersion = previousStatus?.targetSha?.startsWith('backend:')
      ? previousStatus.targetSha.slice('backend:'.length)
      : undefined

    const started = await updateHermes()
    const applyStartedAtMs = Date.now()

    if (!started.ok) {
      const message = (started as { message?: string }).message || translateNow('updates.applyStatus.notAvailable')
      const command = (started as { update_command?: string }).update_command || 'hermes update'
      $backendUpdateApply.set({ ...IDLE, applying: false, stage: 'manual', message, command })

      return { ok: false, error: 'manual', manual: true, message, command }
    }

    $backendUpdateApply.set({
      ...IDLE,
      applying: true,
      stage: 'pull',
      message: translateNow('updates.applyStatus.pulling')
    })

    let last: Awaited<ReturnType<typeof getActionStatus>> | null = null
    // Backups, dependency repair, and builds can legitimately take several
    // minutes. Keep the generous cap only as a guard against a stuck action.
    const actionDeadline = Date.now() + BACKEND_ACTION_MAX_MS
    let deadline = actionDeadline
    let reconnecting = false

    while (Date.now() < deadline) {
      await new Promise(resolve => globalThis.setTimeout(resolve, BACKEND_ACTION_POLL_MS))

      try {
        last = await getActionStatus(started.name, 2000)
        ingestBackendActionStatus(last)
      } catch {
        if (!reconnecting) {
          reconnecting = true
          deadline = Date.now() + BACKEND_RETURN_MAX_MS
          $backendUpdateApply.set({
            ...$backendUpdateApply.get(),
            applying: true,
            stage: 'restart',
            message: translateNow('updates.applyStatus.restarting')
          })
        }

        continue
      }

      if (last.running) {
        if (reconnecting) {
          reconnecting = false
          deadline = actionDeadline
          $backendUpdateApply.set({
            ...$backendUpdateApply.get(),
            applying: true,
            stage: 'pull',
            message: translateNow('updates.applyStatus.pulling')
          })
        }

        continue
      }

      if (last.exit_code === 0 || (last.exit_code === null && completedAfterRestart(last, started.action_id))) {
        return finishBackendApply(true)
      }

      // #91277 bullet 3: the backend now attaches the durable update
      // receipt to the status. A receipt whose run STARTED after we kicked
      // this update off is authoritative — read its outcome instead of
      // inferring from log markers or timing out across the restart gap.
      if (last.exit_code === null && receiptProvesOutcome(last, applyStartedAtMs)) {
        return finishBackendApply(last.receipt!.outcome === 'success')
      }

      if (!started.action_id && last.exit_code === null) {
        try {
          const status = await checkHermesUpdate(true)

          if (legacyBackendReachedTarget(status, requestedTargetSha, previousVersion)) {
            return finishBackendApply(true)
          }
        } catch {
          continue
        }
      }

      if (last.exit_code !== null) {
        break
      }
    }

    $backendUpdateApply.set({
      ...$backendUpdateApply.get(),
      applying: false,
      stage: 'error',
      error: 'apply-failed',
      message: translateNow('updates.applyStatus.failed')
    })

    return { ok: false, error: 'apply-failed', message: 'Backend update failed.' }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    $backendUpdateApply.set({
      ...$backendUpdateApply.get(),
      applying: false,
      stage: 'error',
      error: 'apply-failed',
      message
    })

    return { ok: false, error: 'apply-failed', message }
  }
}

export function applyBackendUpdate(): Promise<DesktopUpdateApplyResult> {
  if (backendUpdateInFlight) {
    return backendUpdateInFlight
  }

  backendUpdateInFlight = runBackendUpdate().finally(() => {
    backendUpdateInFlight = null
  })

  return backendUpdateInFlight
}

// ── Update everything: the client + every registered backend in one action ──
//
// Remote-mode installs update on two (or more) clocks: the GUI app on this
// machine, the connected backend, and any other registered sources. Each has
// its own updater, and before this flow existed every remote-mode affordance
// targeted only the backend — so users "updated" and stayed on a stale GUI.
// This orchestration drives all of them:
//   1. The ACTIVE backend (remote mode) through the detailed-progress path.
//   2. Every OTHER eligible registered connection via the Electron fan-out
//      (cloud rows are platform-managed and report as skipped).
//   3. The local client LAST — its apply relaunches or hands off the app, so
//      it must not preempt the dispatches above.

const CLIENT_BEHIND_TOAST_ID = 'client-update-after-backend'

/** After a successful backend update, tell the user when the desktop app
 *  itself is still behind, with a one-click client update. Silent when the
 *  client is current, so aligned installs never see it. */
async function maybeNudgeClientAfterBackendUpdate(): Promise<void> {
  if (typeof window === 'undefined') {
    return
  }

  const status = (await checkUpdates().catch(() => null)) ?? $updateStatus.get()

  if (!status || status.error || (!status.updateAvailable && (status.behind ?? 0) <= 0)) {
    return
  }

  notify({
    action: {
      label: translateNow('updates.clientAlsoBehindAction'),
      onClick: () => {
        dismissNotification(CLIENT_BEHIND_TOAST_ID)
        $updateOverlayTarget.set('client')
        $updateOverlayOpen.set(true)
        void applyUpdates()
      }
    },
    durationMs: 0,
    id: CLIENT_BEHIND_TOAST_ID,
    kind: 'warning',
    message: translateNow('updates.clientAlsoBehindMessage'),
    title: translateNow('updates.clientAlsoBehindTitle')
  })
}

export interface UpdateEverythingState {
  running: boolean
}

export const $updateEverything = atom<UpdateEverythingState>({ running: false })

/** True when this install has more than one update target — a remote-mode
 *  window (backend + client) or a multi-connection registry. Gates the
 *  "Update everything" affordance so single-machine installs keep the
 *  one-button experience. */
export function hasMultipleUpdateTargets(): boolean {
  return isRemoteMode() || ($connectionsRegistry.get()?.connections.length ?? 0) > 1
}

let updateEverythingInFlight: Promise<void> | null = null

export function applyEverythingUpdate(): Promise<void> {
  if (updateEverythingInFlight) {
    return updateEverythingInFlight
  }

  updateEverythingInFlight = runEverythingUpdate().finally(() => {
    updateEverythingInFlight = null
  })

  return updateEverythingInFlight
}

async function runEverythingUpdate(): Promise<void> {
  $updateEverything.set({ running: true })

  try {
    // 1. Active backend first (remote mode), with the detailed overlay flow.
    //    Its own finish path re-checks and nudges, but the everything-flow
    //    continues regardless of the outcome: one unreachable backend must
    //    not strand the other machines or the client.
    if (isRemoteMode()) {
      $updateOverlayTarget.set('backend')

      await applyBackendUpdate().catch(() => null)
    }

    // 2. Fan out to every OTHER eligible registered connection. The active
    //    backend was just updated (excluded), and the local runtime updates
    //    with the client in step 3 (excluded). No registry/bridge → skip.
    const bridge = window.hermesDesktop?.connections
    const registry = $connectionsRegistry.get() ?? (await refreshConnectionsRegistry().catch(() => null))
    const excludeIds = ['local']
    const activeConnectionId = $connection.get()?.connectionId

    if (isRemoteMode() && activeConnectionId && !excludeIds.includes(activeConnectionId)) {
      excludeIds.push(activeConnectionId)
    }

    const remaining = (registry?.connections ?? []).filter(connection => !excludeIds.includes(connection.id))

    if (bridge?.updateAll && remaining.length > 0) {
      try {
        const { results } = await bridge.updateAll({ excludeIds })

        for (const row of results) {
          if (row.ok) {
            notify({ title: row.label, message: row.detail || translateNow('updates.everythingDispatched') })
          } else if (row.skipped) {
            notify({
              title: row.label,
              message: row.detail || row.reason || translateNow('updates.everythingSkipped')
            })
          } else {
            notify({
              kind: 'warning',
              title: row.label,
              message: row.error || row.detail || translateNow('updates.everythingRowFailed')
            })
          }
        }
      } catch (error) {
        notify({
          kind: 'warning',
          title: translateNow('updates.everythingFanoutFailedTitle'),
          message: error instanceof Error ? error.message : String(error)
        })
      }
    }

    // 3. The client last — its apply relaunches or hands off the app, so it
    //    must come after every dispatch above. Skipped when already current.
    const clientStatus = $updateStatus.get() ?? (await checkUpdates())

    if ((clientStatus?.behind ?? 0) > 0 || clientStatus?.updateAvailable) {
      $updateOverlayTarget.set('client')
      $updateOverlayOpen.set(true)
      await applyUpdates()
    }
  } finally {
    $updateEverything.set({ running: false })
  }
}

function ingestProgress(payload: DesktopUpdateProgress): void {
  const current = $updateApply.get()
  const log = [...current.log, { stage: payload.stage, message: payload.message, at: payload.at }].slice(-50)

  const terminal =
    payload.stage === 'error' ||
    payload.stage === 'restart' ||
    payload.stage === 'manual' ||
    payload.stage === 'guiSkew'

  $updateApply.set({
    applying: !terminal,
    stage: payload.stage,
    message: payload.message,
    // Streamed log lines carry percent: null; keep the last milestone percent
    // (10/60/…) instead of resetting the bar to indeterminate on every line.
    percent: payload.percent ?? current.percent,
    error: payload.error,
    // 'manual' carries the command to run in its message field.
    command: payload.stage === 'manual' ? payload.message : current.command,
    log
  })
}

let pollerStarted = false
let backgroundTimer: ReturnType<typeof setInterval> | null = null
let lastFocusAt = 0
let connectionUnsub: (() => void) | null = null
let lastConnectionMode: string | undefined

/** Wire up background polling + progress streaming. Idempotent. */
export function startUpdatePoller(): void {
  if (pollerStarted || typeof window === 'undefined') {
    return
  }

  const bridge = window.hermesDesktop?.updates

  if (!bridge) {
    return
  }

  pollerStarted = true
  void checkUpdates()
  void checkBackendUpdates()
  void refreshDesktopVersion()
  bridge.onProgress(ingestProgress)

  // The poller starts at mount, before the gateway connects — so the first
  // backend check above sees mode≠remote and no-ops. Re-check once the
  // connection resolves to remote.
  connectionUnsub = $connection.subscribe(conn => {
    if (conn?.mode === lastConnectionMode) {
      return
    }

    lastConnectionMode = conn?.mode

    if (conn?.mode === 'remote') {
      void checkBackendUpdates()
    }
  })

  window.addEventListener('focus', onFocus)
  backgroundTimer = setInterval(
    () => {
      void checkUpdates()
      void checkBackendUpdates()
    },
    30 * 60 * 1000
  )
}

export function stopUpdatePoller(): void {
  if (backgroundTimer !== null) {
    clearInterval(backgroundTimer)
    backgroundTimer = null
  }

  connectionUnsub?.()
  connectionUnsub = null
  lastConnectionMode = undefined
  window.removeEventListener('focus', onFocus)
  pollerStarted = false
}

function onFocus() {
  const now = Date.now()

  if (now - lastFocusAt < 5 * 60 * 1000) {
    return
  }

  lastFocusAt = now
  void checkUpdates()
  void checkBackendUpdates()
  void refreshDesktopVersion()
}
