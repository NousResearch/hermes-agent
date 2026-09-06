import type { ReactNode } from 'react'
import { useEffect, useId, useRef, useState } from 'react'

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
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'

interface ConfirmDialogProps {
  open: boolean
  onClose: () => void
  // Does the work. Throw to surface an inline error and keep the dialog open.
  onConfirm: () => Promise<void> | void
  title: ReactNode
  description?: ReactNode
  confirmLabel?: string
  busyLabel?: string
  doneLabel?: string
  cancelLabel?: string
  typedConfirmation?: string
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
// does the work.
export function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  description,
  confirmLabel,
  busyLabel,
  doneLabel,
  cancelLabel,
  destructive = false,
  dismissOnConfirm = false,
  secondaryAction,
  typedConfirmation
}: ConfirmDialogProps) {
  const { t } = useI18n()
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const submittingRef = useRef(false)
  const [confirmation, setConfirmation] = useState('')
  const ready = typedConfirmation === undefined || confirmation === typedConfirmation
  const confirmRef = useRef<HTMLButtonElement>(null)
  const closeTimerRef = useRef<null | number>(null)
  const [status, setStatus] = useState<'done' | 'idle' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)
  const busy = status === 'saving' || status === 'done'
  const resolvedConfirmLabel = confirmLabel ?? t.common.confirm
  const resolvedBusyLabel = busyLabel ?? t.common.loading
  const resolvedDoneLabel = doneLabel ?? t.common.done
  const resolvedCancelLabel = cancelLabel ?? t.common.cancel

  // Reset the submission latch for a new prompt; no reactive value is mirrored.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    if (open) {
      setConfirmation('')
      submittingRef.current = false
      setStatus('idle')
      setError(null)
    }
  }, [open, typedConfirmation])

  // Cancel the pending close timer on unmount. The timer below holds the
  // "done" beat visible for 600ms, and an unmount inside that window used to
  // leave it armed. It then called onClose on a tree that is gone, which
  // reaches setState in the parent. Under vitest the environment can be torn
  // down first, and React then reads `window` during the update and throws
  // ReferenceError.
  // The write below is a timer handle, and not a mirror of a reactive value.
  // It happens on unmount only, and it clears the handle this component owns.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    return () => {
      if (closeTimerRef.current !== null) {
        window.clearTimeout(closeTimerRef.current)
        closeTimerRef.current = null
      }
    }
  }, [])

  async function run() {
    if (busy || submittingRef.current || !ready) {
      return
    }

    submittingRef.current = true
    setStatus('saving')
    setError(null)

    if (dismissOnConfirm) {
      try {
        await onConfirm()
        onClose()
      } catch (err) {
        submittingRef.current = false
        setStatus('idle')
        setError(err instanceof Error ? err.message : t.errors.genericFailure)
      }

      return
    }

    setStatus('saving')

    try {
      await onConfirm()
      setStatus('done')
      closeTimerRef.current = window.setTimeout(() => {
        closeTimerRef.current = null
        onClose()
      }, 600)
    } catch (err) {
      submittingRef.current = false
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
          const typedInput = event.target === inputRef.current

          if (typedConfirmation !== undefined && event.target !== confirmRef.current && !typedInput) {
            return
          }

          if ((event.key === 'Enter' || (event.key === ' ' && !typedInput)) && !event.nativeEvent.isComposing) {
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
          ;(typedConfirmation === undefined ? confirmRef.current : inputRef.current)?.focus()
        }}
      >
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {description ? <DialogDescription>{description}</DialogDescription> : null}
        </DialogHeader>

        {typedConfirmation !== undefined && (
          <div className="flex flex-col gap-2">
            <label className="text-xs text-muted-foreground" htmlFor={inputId}>
              {t.common.typedConfirmation(typedConfirmation)}
            </label>
            <Input
              disabled={busy}
              id={inputId}
              onChange={event => setConfirmation(event.target.value)}
              ref={inputRef}
              value={confirmation}
            />
          </div>
        )}

        {error && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <DialogFooter>
          <Button disabled={busy} onClick={onClose} type="button" variant="ghost">
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
            disabled={busy || !ready}
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
