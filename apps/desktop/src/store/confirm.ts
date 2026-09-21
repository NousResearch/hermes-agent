import { atom } from 'nanostores'

export interface ConfirmRequest {
  title: string
  description?: string
  confirmLabel?: string
  cancelLabel?: string
  destructive?: boolean
}

export interface PendingConfirm extends ConfirmRequest {
  resolve: (confirmed: boolean) => void
}

export const $confirmRequest = atom<null | PendingConfirm>(null)

// Imperative front door to ConfirmDialog, for handlers that want the answer
// inline the way window.confirm gave it — `if (!ok) return`. A surface that
// wants the busy → done beat or an inline error should mount <ConfirmDialog>
// itself and hand it the async onConfirm.
export function confirm(request: ConfirmRequest): Promise<boolean> {
  // One modal at a time: a second ask supersedes the first, which answers no.
  settleConfirm(false)

  return new Promise<boolean>(resolve => {
    $confirmRequest.set({ ...request, resolve })
  })
}

/** Answer the open request, if there still is one. Idempotent. */
export function settleConfirm(confirmed: boolean): void {
  const pending = $confirmRequest.get()

  if (!pending) {
    return
  }

  $confirmRequest.set(null)
  pending.resolve(confirmed)
}
