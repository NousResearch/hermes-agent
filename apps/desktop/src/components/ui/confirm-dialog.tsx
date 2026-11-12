import type { ReactNode } from 'react'
import { useEffect, useRef, useState } from 'react'

import { ActionStatus } from '@/components/ui/action-status'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'
import { usePortalShell } from '@/store/shell-variant'

interface ConfirmDialogProps {
  open: boolean
  onClose: () => void
  // Does the work. Throw to surface an inline error and keep the dialog open.
  onConfirm: () => Promise<void> | void
  title: ReactNode
  description?: ReactNode
  children?: ReactNode
  confirmLabel?: string
  busyLabel?: string
  doneLabel?: string
  cancelLabel?: string
  destructive?: boolean
  /** Close as soon as onConfirm resolves — for optimistic actions that finish in the background. */
  dismissOnConfirm?: boolean
  /** A third, non-destructive way out, shown between Cancel and Confirm (e.g.
   *  "Remove from sidebar" beside "Delete worktree"). Closes on click. */
  secondaryAction?: ConfirmSecondaryAction
}

interface ConfirmSecondaryAction {
  label: string
  onClick: () => void
}

// Shared confirmation dialog: opens focused on Confirm, Enter confirms (from
// anywhere in the dialog), Esc/Cancel/backdrop dismiss. Owns the pending → done
// → close beat and inline error, so callers pass only an async onConfirm that
// does the work. The redesign has one confirm anatomy: no corner X, a bordered
// Cancel beside the solid action, and a destructive confirm opens on Cancel,
// where Enter answers the focused button, so a stray Enter never destroys.
export function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  description,
  children,
  confirmLabel,
  busyLabel,
  doneLabel,
  cancelLabel,
  destructive = false,
  dismissOnConfirm = false,
  secondaryAction
}: ConfirmDialogProps) {
  const { t } = useI18n()
  const portal = usePortalShell()
  const safeDefault = portal && destructive
  const confirmRef = useRef<HTMLButtonElement>(null)
  const cancelRef = useRef<HTMLButtonElement>(null)
  const closeTimerRef = useRef<null | number>(null)
  const unmountedRef = useRef(false)
  const [status, setStatus] = useState<'done' | 'idle' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)
  const busy = status === 'saving' || status === 'done'
  const resolvedConfirmLabel = confirmLabel ?? t.common.confirm
  const resolvedBusyLabel = busyLabel ?? t.common.loading
  const resolvedDoneLabel = doneLabel ?? t.common.done
  const resolvedCancelLabel = cancelLabel ?? t.common.cancel

  useEffect(() => {
    if (open) {
      setStatus('idle')
      setError(null)
    }
  }, [open])

  // Cancel the pending close timer on unmount. The timer below holds the
  // "done" beat visible for 600ms, and an unmount inside that window used to
  // leave it armed. It then called onClose on a tree that is gone, which
  // reaches setState in the parent. Under vitest the environment can be torn
  // down first, and React then reads `window` during the update and throws
  // ReferenceError.
  // The write below is a timer handle, and not a mirror of a reactive value.
  // It happens on unmount only, and it clears the handle this component owns.
  // An onConfirm that unmounts the caller (disband leaves the room) resolves
  // after cleanup ran, so the timer must not be armed once unmounted.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    unmountedRef.current = false

    return () => {
      unmountedRef.current = true

      if (closeTimerRef.current !== null) {
        window.clearTimeout(closeTimerRef.current)
        closeTimerRef.current = null
      }
    }
  }, [])

  async function run() {
    if (busy) {
      return
    }

    setError(null)

    if (dismissOnConfirm) {
      try {
        await onConfirm()
        onClose()
      } catch (err) {
        setError(err instanceof Error ? err.message : t.errors.genericFailure)
      }

      return
    }

    setStatus('saving')

    try {
      await onConfirm()

      if (unmountedRef.current) {
        return
      }

      setStatus('done')
      closeTimerRef.current = window.setTimeout(() => {
        closeTimerRef.current = null
        onClose()
      }, 600)
    } catch (err) {
      setStatus('idle')
      setError(err instanceof Error ? err.message : t.errors.genericFailure)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !busy && onClose()} open={open}>
      <DialogContent
        className="max-w-md"
        onKeyDown={event => {
          // Enter/Space confirm regardless of which button holds focus
          // (preventDefault stops a focused Cancel from swallowing it).
          if ((event.key === 'Enter' || event.key === ' ') && !busy && !safeDefault) {
            event.preventDefault()
            void run()
          }
        }}
        onOpenAutoFocus={event => {
          // Focus must land inside the dialog or the handler above never sees
          // the key: it stays on whatever opened the dialog (a menu item, a
          // sidebar row) and Enter re-triggers that instead. Radix's default
          // would take the X — confirm is the button Enter maps to.
          event.preventDefault()
          const target = safeDefault ? cancelRef : confirmRef
          target.current?.focus()
        }}
        showCloseButton={!portal}
      >
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {/* pre-line: a backend-composed description keeps its paragraph
              breaks instead of collapsing into one run-on line (#112458). */}
          {description ? <DialogDescription className="whitespace-pre-line">{description}</DialogDescription> : null}
        </DialogHeader>

        {children}
        {error && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <DialogFooter>
          <Button
            disabled={busy}
            onClick={onClose}
            ref={cancelRef}
            type="button"
            variant={portal ? 'secondary' : 'ghost'}
          >
            {resolvedCancelLabel}
          </Button>
          {secondaryAction && (
            <Button
              disabled={busy}
              onClick={() => {
                secondaryAction.onClick()
                onClose()
              }}
              type="button"
              variant="secondary"
            >
              {secondaryAction.label}
            </Button>
          )}
          <Button
            disabled={busy}
            onClick={() => void run()}
            ref={confirmRef}
            variant={destructive ? 'destructive' : 'default'}
          >
            <ActionStatus
              busy={resolvedBusyLabel}
              done={resolvedDoneLabel}
              idle={resolvedConfirmLabel}
              state={status}
            />
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
