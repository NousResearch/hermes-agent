import { dismissNotification, notify, notifyError } from '@/store/notifications'

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
  /** Label for the confirm action button (and default title). */
  confirmLabel: string
  /** The gateway's `confirm_message`, when present. */
  confirmMessage?: string
  /** Error-toast copy when the confirmed resend still fails. */
  failureMessage: string
  /** Message when the gateway sent no `confirm_message`. */
  fallbackMessage?: string
  /** Runs after the confirmed resend succeeds (cache invalidation etc.). */
  finish?: (result: T | undefined) => void
  /** Staleness guard — the warning can linger while the user picks a
   *  different model or switches sessions. Return true to make Confirm a
   *  no-op (the notification is dismissed) instead of clobbering the newer
   *  choice. */
  isStale?: () => boolean
  /** Optimistically repaint the pending selection before the resend. */
  repaint?: () => void
  /** Resend the switch WITH `confirm_expensive_model: true`. */
  requestConfirmed: () => Promise<T | undefined>
  /** Undo the optimistic repaint when the confirmed resend fails. */
  rollback?: () => void
  title?: string
}

/**
 * THE confirm flow for guarded model switches — every surface that can
 * receive `confirm_required` from a model switch (core picker via
 * `config.set`, Bots editor via `profiles.configure`, future surfaces) routes
 * it through here so there is exactly one applier and no forked confirm
 * logic per surface (#95293).
 *
 * Shows a warning notification whose Confirm action resends the switch with
 * `confirm_expensive_model: true`, guarded against stale confirmations. A
 * resend that STILL answers `confirm_required` is treated as a failure — the
 * gateway asked twice, something is wrong; never loop.
 *
 * Returns the notification id.
 */
export function surfaceModelSwitchConfirm<T extends GuardedModelSwitchResult>(
  options: SurfaceModelSwitchConfirmOptions<T>
): string {
  const applyConfirmedSwitch = async () => {
    if (options.isStale?.()) {
      dismissNotification(notificationId)

      return
    }

    dismissNotification(notificationId)
    options.repaint?.()

    try {
      const result = await options.requestConfirmed()

      if (result?.confirm_required) {
        throw new Error(result.confirm_message?.trim() || options.failureMessage)
      }

      options.finish?.(result)
    } catch (err) {
      options.rollback?.()
      notifyError(err, options.failureMessage)
    }
  }

  const notificationId = notify({
    action: {
      label: options.confirmLabel,
      onClick: applyConfirmedSwitch
    },
    kind: 'warning',
    message: options.confirmMessage?.trim() || options.fallbackMessage || 'Confirm this model switch?',
    title: options.title ?? options.confirmLabel
  })

  return notificationId
}
