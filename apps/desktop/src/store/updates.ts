/**
 * Desktop self-update store. Tracks distance from the configured branch,
 * surfaces it as an ambient pill, and orchestrates the apply flow.
 */

import { atom } from 'nanostores'

import type {
  DesktopUpdateApplyOptions,
  DesktopUpdateApplyResult,
  DesktopUpdateProgress,
  DesktopUpdateStage,
  DesktopUpdateStatus,
  DesktopVersionInfo
} from '@/global'
import { checkHermesUpdate, getActionStatus, updateHermes } from '@/hermes'
import { translateNow } from '@/i18n'
import { persistString, storedString } from '@/lib/storage'
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
const REQUIRED_BACKEND_CONTRACT = 2
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

  if ((status.behind ?? 0) <= 0) {
    return
  }

  if (isUpdateToastSnoozed()) {
    return
  }

  if ($updateApply.get().applying) {
    return
  }

  const behind = status.behind ?? 0

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
    message: translateNow('notifications.updateReadyMessage', behind),
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
 */
export function startActiveUpdate(): void {
  const target: UpdateTarget = isRemoteMode() ? 'backend' : 'client'
  $updateOverlayTarget.set(target)
  $updateOverlayOpen.set(true)
  void (target === 'backend' ? applyBackendUpdate() : applyUpdates())
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
          title: translateNow('updates.allSetTitle')
        })
      } else {
        $updateApply.set({
          ...$updateApply.get(),
          applying: false,
          stage: 'error',
          error: result?.error ?? 'apply-failed',
          message: result?.message ?? translateNow('updates.errorBody')
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

const BACKEND_RETURN_POLL_MS = 1500
const BACKEND_RETURN_MAX_ATTEMPTS = 40

async function waitForBackendReturn(): Promise<boolean> {
  for (let attempt = 0; attempt < BACKEND_RETURN_MAX_ATTEMPTS; attempt += 1) {
    await new Promise(resolve => globalThis.setTimeout(resolve, BACKEND_RETURN_POLL_MS))

    try {
      await checkHermesUpdate()

      return true
    } catch {
      continue
    }
  }

  return false
}

function finishBackendApply(returned: boolean): DesktopUpdateApplyResult {
  if (returned) {
    $backendUpdateApply.set(IDLE)
    setUpdateOverlayOpen(false)
    void checkBackendUpdates()

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

export async function applyBackendUpdate(): Promise<DesktopUpdateApplyResult> {
  dismissNotification(UPDATE_TOAST_ID)
  $backendUpdateApply.set({
    ...IDLE,
    applying: true,
    stage: 'prepare',
    message: translateNow('updates.applyStatus.preparing')
  })

  try {
    const started = await updateHermes()

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

    for (let attempt = 0; attempt < 30; attempt += 1) {
      await new Promise(resolve => globalThis.setTimeout(resolve, 1500))

      try {
        last = await getActionStatus(started.name, 200)
        ingestBackendActionStatus(last)
      } catch {
        // The dashboard restarts mid-update, dropping this connection — expected, not a failure.
        $backendUpdateApply.set({
          ...$backendUpdateApply.get(),
          applying: true,
          stage: 'restart',
          message: translateNow('updates.applyStatus.restarting')
        })

        return finishBackendApply(await waitForBackendReturn())
      }

      if (last && !last.running) {
        break
      }
    }

    const ok = !!last && (last.exit_code ?? 1) === 0

    if (ok) {
      $backendUpdateApply.set({
        ...$backendUpdateApply.get(),
        applying: true,
        stage: 'restart',
        message: translateNow('updates.applyStatus.restarting')
      })

      return finishBackendApply(await waitForBackendReturn())
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
