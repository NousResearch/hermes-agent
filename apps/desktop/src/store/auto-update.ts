/**
 * Opt-in automatic updates on the first launch after login (#123674).
 *
 * Electron owns the policy (one attempt per OS login session, deferral while a
 * gateway has turns in flight — electron/updater/auto-update.ts). This store
 * mirrors the setting for Settings → About, and on launch asks main for a
 * claim; when granted it drives the SAME check/apply flow as the "Update now"
 * button, so every install shape keeps its own updater and relaunch.
 */

import { atom } from 'nanostores'

import type {
  DesktopAutoUpdateAttempt,
  DesktopAutoUpdateOutcome,
  DesktopAutoUpdateView,
  DesktopUpdateStatus
} from '@/global'
import { translateNow } from '@/i18n'
import { notify } from '@/store/notifications'
import {
  $updateApply,
  $updateChecking,
  $updateOverlayOpen,
  $updateOverlayTarget,
  applyUpdates,
  checkUpdates
} from '@/store/updates'

export interface AutoUpdateView {
  loaded: boolean
  enabled: boolean
  supported: boolean
  saving: boolean
  lastAttempt: DesktopAutoUpdateAttempt | null
}

const INITIAL: AutoUpdateView = { loaded: false, enabled: false, supported: false, saving: false, lastAttempt: null }

export const $autoUpdate = atom<AutoUpdateView>(INITIAL)

const AUTO_UPDATE_TOAST_ID = 'desktop-auto-update'
/** How long to wait for a passive check already in flight before our forced one. */
const CHECK_IDLE_WAIT_MS = 60_000
/** Upper bound on deferral retries inside one app run (main also caps by time). */
const MAX_DEFER_RETRIES = 45

function bridge() {
  return typeof window === 'undefined' ? undefined : window.hermesDesktop?.updates?.auto
}

function applyView(view: DesktopAutoUpdateView | null | undefined): void {
  if (!view) {
    return
  }

  $autoUpdate.set({
    loaded: true,
    enabled: Boolean(view.enabled),
    supported: Boolean(view.supported),
    saving: false,
    lastAttempt: view.lastAttempt ?? null
  })
}

export async function loadAutoUpdate(): Promise<AutoUpdateView> {
  const auto = bridge()

  if (!auto) {
    $autoUpdate.set({ ...INITIAL, loaded: true })

    return $autoUpdate.get()
  }

  try {
    applyView(await auto.get())
  } catch {
    $autoUpdate.set({ ...INITIAL, loaded: true })
  }

  return $autoUpdate.get()
}

/** Optimistic toggle; the main-process answer gets the last word. */
export async function setAutoUpdateEnabled(enabled: boolean): Promise<void> {
  const auto = bridge()
  const previous = $autoUpdate.get()

  if (!auto) {
    return
  }

  $autoUpdate.set({ ...previous, enabled, saving: true })

  try {
    applyView(await auto.set(enabled))
  } catch (error) {
    $autoUpdate.set({ ...previous, saving: false })
    notify({
      kind: 'error',
      title: translateNow('updates.autoUpdate.saveFailed'),
      message: error instanceof Error ? error.message : String(error)
    })
  }
}

async function report(
  sessionKey: string,
  outcome: DesktopAutoUpdateOutcome,
  extra: { target?: string; message?: string } = {}
) {
  try {
    applyView(await bridge()?.report({ sessionKey, outcome, ...extra }))
  } catch {
    // Reporting is best-effort; the session is already claimed in main.
  }
}

function waitForIdleCheck(): Promise<void> {
  if (!$updateChecking.get()) {
    return Promise.resolve()
  }

  return new Promise(resolve => {
    const timer = globalThis.setTimeout(() => {
      unsubscribe()
      resolve()
    }, CHECK_IDLE_WAIT_MS)

    const unsubscribe = $updateChecking.listen(checking => {
      if (!checking) {
        globalThis.clearTimeout(timer)
        unsubscribe()
        resolve()
      }
    })
  })
}

function targetLabel(status: DesktopUpdateStatus): string | undefined {
  return status.latestTag || status.channel || status.branch || undefined
}

/** Classify a fresh check into "go", or the outcome that ends this session's attempt. */
export function autoUpdateCheckVerdict(status: DesktopUpdateStatus | null): 'apply' | DesktopAutoUpdateOutcome {
  if (!status || status.error) {
    return 'check-failed'
  }

  if (status.supported === false || status.retirement) {
    return 'skipped-unsupported'
  }

  if ((status.behind ?? 0) <= 0 && !status.updateAvailable) {
    return 'up-to-date'
  }

  // Never auto-update a checkout with local edits: `hermes update` would stash
  // them unattended. The user can still update by hand from this page.
  if (status.dirty) {
    return 'skipped-dirty'
  }

  return 'apply'
}

let started = false
let deferTimer: ReturnType<typeof setTimeout> | null = null

/**
 * Launch hook. Idempotent per renderer; the per-login-session guarantee lives
 * in main, so a reload or second window cannot re-trigger it either.
 */
export function runAutoUpdateOnLaunch(): void {
  if (started || !bridge()) {
    return
  }

  started = true
  void attempt(0)
}

async function attempt(retries: number): Promise<void> {
  deferTimer = null
  const auto = bridge()

  if (!auto) {
    return
  }

  let claim

  try {
    claim = await auto.claim()
  } catch {
    return
  }

  if (claim.action === 'defer') {
    if (retries === 0) {
      notify({
        durationMs: 8000,
        id: AUTO_UPDATE_TOAST_ID,
        kind: 'info',
        title: translateNow('updates.autoUpdate.deferredTitle'),
        message: translateNow('updates.autoUpdate.deferredMessage')
      })
    }

    if (retries < MAX_DEFER_RETRIES) {
      deferTimer = globalThis.setTimeout(() => void attempt(retries + 1), claim.retryInMs ?? 60_000)
    }

    return
  }

  if (claim.action !== 'run') {
    if (claim.reason === 'deferred-too-long') {
      void loadAutoUpdate()
    }

    return
  }

  await waitForIdleCheck()
  const status = await checkUpdates({ force: true })
  const verdict = autoUpdateCheckVerdict(status)

  if (verdict !== 'apply') {
    await report(claim.sessionKey, verdict, {
      target: status ? targetLabel(status) : undefined,
      message: status?.message
    })

    if (verdict === 'check-failed' || verdict === 'skipped-dirty') {
      notify({
        durationMs: 10_000,
        id: AUTO_UPDATE_TOAST_ID,
        kind: 'warning',
        title: translateNow('updates.autoUpdate.skippedTitle'),
        message:
          verdict === 'skipped-dirty'
            ? translateNow('updates.autoUpdate.skippedDirty')
            : translateNow('updates.autoUpdate.checkFailed')
      })
    }

    return
  }

  // A manual apply may already be running (user clicked first) — never double up.
  if ($updateApply.get().applying) {
    return
  }

  // The user opted in, and the app is about to close for the update: show
  // the progress instead of vanishing without a word.
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)

  const result = await applyUpdates()
  const target = status ? targetLabel(status) : undefined

  if (result.handedOff) {
    await report(claim.sessionKey, 'handed-off', { target })
  } else if (result.ok && !result.manual) {
    await report(claim.sessionKey, result.updateAvailable === false ? 'up-to-date' : 'updated', { target })
  } else {
    await report(claim.sessionKey, 'failed', {
      target,
      message: result.message || result.error || (result.manual ? result.command : undefined)
    })
  }
}

/** Test seam. */
export function _resetAutoUpdateForTests(): void {
  started = false

  if (deferTimer) {
    globalThis.clearTimeout(deferTimer)
    deferTimer = null
  }

  $autoUpdate.set(INITIAL)
}
