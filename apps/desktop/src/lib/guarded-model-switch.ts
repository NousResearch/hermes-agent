import { translateNow } from '@/i18n'
import { confirm } from '@/store/confirm'
import { notify, notifyError } from '@/store/notifications'

/** The gateway's model-switch handshake shape — shared by `config.set model`
 *  and `profiles.configure` (Bots editor). `confirm_required: true` means the
 *  switch was intentionally NOT applied and the gateway is waiting for a
 *  resend that carries `confirm_expensive_model: true`. */
export interface GuardedModelSwitchResult {
  confirm_message?: string
  confirm_required?: boolean
  deferred?: boolean
}

export interface SurfaceModelSwitchConfirmOptions<T extends GuardedModelSwitchResult> {
  /** The gateway's `confirm_message`, when present. */
  confirmMessage?: string
  /** Error-toast copy when the confirmed resend still fails. */
  failureMessage: string
  /** Runs after the confirmed resend succeeds (cache invalidation etc.). */
  finish?: (result: T | undefined) => void
  /** Staleness guard — the session/model can move on while the dialog is
   *  open. Return true to make the answered confirm a no-op (with a notice
   *  saying so) instead of clobbering the newer choice. */
  isStale?: () => boolean
  /** The model the switch targets — named in the dialog title when known. */
  model?: string
  /** Optimistically repaint the pending selection before the resend. */
  repaint?: () => void
  /** Resend the switch WITH `confirm_expensive_model: true`. */
  requestConfirmed: () => Promise<T | undefined>
  /** Undo the optimistic repaint when the confirmed resend fails. */
  rollback?: () => void
}

/**
 * THE confirm flow for guarded model switches — every surface that can
 * receive `confirm_required` from a model switch (core picker via
 * `config.set`, Bots editor via `profiles.configure`, future surfaces) routes
 * it through here so there is exactly one applier and no forked confirm
 * logic per surface (#95293).
 *
 * A refused switch is a decision, not a notification: it asks through
 * `confirm()` — the app's ConfirmDialog — so there is a real decline button,
 * Esc/backdrop/✕ all mean "keep current model", and the dialog owns focus
 * while it waits (#112458). Declining is free: nothing was applied. Only a
 * confirmed answer resends with `confirm_expensive_model: true`; a resend
 * that STILL answers `confirm_required` is treated as a failure — the gateway
 * asked twice, something is wrong; never loop.
 *
 * Resolves `true` when the switch was applied, `false` when it was declined,
 * went stale, or failed (a failure surfaces its own toast).
 */
export async function surfaceModelSwitchConfirm<T extends GuardedModelSwitchResult>(
  options: SurfaceModelSwitchConfirmOptions<T>
): Promise<boolean> {
  const accepted = await confirm({
    cancelLabel: translateNow('desktop.modelSwitchKeepLabel'),
    confirmLabel: translateNow('desktop.modelSwitchConfirmLabel'),
    description: options.confirmMessage?.trim() || translateNow('desktop.modelSwitchConfirmBody'),
    destructive: true,
    title: options.model
      ? translateNow('desktop.modelSwitchConfirmTitle', options.model)
      : translateNow('desktop.modelSwitchConfirmTitleFallback')
  })

  if (!accepted) {
    return false
  }

  if (options.isStale?.()) {
    // The session or model moved on under the dialog, so the switch it asked
    // for no longer exists. Say so — a silent no-op reads as a broken button.
    notify({ kind: 'info', message: translateNow('desktop.modelSwitchStaleNotice') })

    return false
  }

  options.repaint?.()

  try {
    const result = await options.requestConfirmed()

    if (options.isStale?.()) {
      return false
    }

    if (result?.confirm_required) {
      throw new Error(result.confirm_message?.trim() || options.failureMessage)
    }

    options.finish?.(result)

    return true
  } catch (err) {
    if (options.isStale?.()) {
      return false
    }

    options.rollback?.()
    notifyError(err, options.failureMessage)

    return false
  }
}
