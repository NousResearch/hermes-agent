import { atom } from 'nanostores'

export interface ConfirmCheckbox {
  label: string
  defaultChecked?: boolean
}

export interface ConfirmMeta {
  checkboxChecked?: boolean
}

export interface ConfirmResult {
  confirmed: boolean
  checkboxChecked?: boolean
}

export interface ConfirmRequest {
  title: string
  description?: string
  confirmLabel?: string
  busyLabel?: string
  doneLabel?: string
  details?: { label: string; value: string }[]
  checkbox?: ConfirmCheckbox
  onConfirm?: (meta?: ConfirmMeta) => Promise<void> | void
  cancelLabel?: string
  destructive?: boolean
}

export interface PendingConfirm extends ConfirmRequest {
  id: number
  resolve: (result: boolean | ConfirmResult) => void
  phase?: 'running' | 'done'
  checkboxChecked?: boolean
}

export const $confirmRequest = atom<null | PendingConfirm>(null)
let nextRequestId = 0

// Imperative front door to ConfirmDialog, for handlers that want the answer
// inline the way window.confirm gave it — `if (!ok) return`. Pass onConfirm
// to use the shared dialog's progress, completion and inline error states.
export function confirm(request: ConfirmRequest): Promise<boolean> {
  // An action already in progress must not lose its owner or completion UI.
  if ($confirmRequest.get()?.phase) {
    return Promise.resolve(false)
  }

  // One modal at a time: a second ask supersedes an unanswered question.
  settleConfirm(false)

  return new Promise<boolean>(resolve => {
    $confirmRequest.set({
      ...request,
      id: ++nextRequestId,
      resolve: (res: boolean | ConfirmResult) => {
        resolve(typeof res === 'boolean' ? res : res.confirmed)
      }
    })
  })
}

/** Confirmation helper that also surfaces the state of the optional checkbox. */
export function confirmWithMeta(request: ConfirmRequest): Promise<ConfirmResult> {
  if ($confirmRequest.get()?.phase) {
    return Promise.resolve({ confirmed: false })
  }

  settleConfirm(false)

  return new Promise<ConfirmResult>(resolve => {
    $confirmRequest.set({
      ...request,
      id: ++nextRequestId,
      resolve: (res: boolean | ConfirmResult) => {
        if (typeof res === 'boolean') {
          resolve({ confirmed: res })
        } else {
          resolve(res)
        }
      }
    })
  })
}

/** Run the captured request, not a replacement that arrived during I/O. */
export async function runConfirm(pending: PendingConfirm, meta?: ConfirmMeta): Promise<void> {
  if ($confirmRequest.get() !== pending || pending.phase) {
    return
  }

  if (meta?.checkboxChecked !== undefined) {
    pending.checkboxChecked = meta.checkboxChecked
  }

  if (!pending.onConfirm) {
    settleConfirm(true, pending)

    return
  }

  pending.phase = 'running'

  try {
    await pending.onConfirm(meta)
    pending.phase = 'done'
  } catch (error) {
    delete pending.phase
    throw error
  }
}

/** Answer the open request, if there still is one. Idempotent. */
export function settleConfirm(confirmed: boolean, expected?: PendingConfirm): void {
  const pending = $confirmRequest.get()

  if (!pending || pending.phase === 'running' || (expected && pending !== expected)) {
    return
  }

  $confirmRequest.set(null)

  if (pending.checkbox) {
    pending.resolve({
      checkboxChecked: confirmed ? (pending.checkboxChecked ?? pending.checkbox.defaultChecked ?? false) : undefined,
      confirmed
    })
  } else {
    pending.resolve(confirmed)
  }
}
