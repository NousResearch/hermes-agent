export type WisdomSyncState = 'pending' | 'syncing' | 'retryable' | 'conflict' | 'uncertain' | 'waiting_for_receipt'
export type WisdomSyncCounts = Record<WisdomSyncState, number>
export interface WisdomSyncSnapshot {
  delivery: WisdomSyncCounts
  operation: WisdomSyncCounts
  can_retry: boolean
}
export interface WisdomSyncView {
  snapshot: WisdomSyncSnapshot | null
  busy: boolean
  error: boolean
}
export const initialWisdomSyncView: WisdomSyncView = { snapshot: null, busy: false, error: false }
export const wisdomSyncCopy = {
  title: 'Receipt and report sync',
  delivery: 'Notification receipts',
  operation: 'Operation reports',
  current: 'Up to date',
  retry: 'Retry sync',
  unavailable: 'Sync status could not be confirmed.',
  scope: 'Sync only. No messages are resent and no skill operations are repeated.',
  states: {
    pending: 'Waiting to sync',
    syncing: 'Syncing',
    retryable: 'Retry available',
    conflict: 'Conflicting server record; needs investigation',
    uncertain: 'Delivery unconfirmed; message will not be resent',
    waiting_for_receipt: 'Waiting for notification receipt'
  }
}
export type WisdomSyncCopy = typeof wisdomSyncCopy

export function createWisdomSyncController(deps: {
  read: () => Promise<WisdomSyncSnapshot>
  retry: () => Promise<WisdomSyncSnapshot>
  changed: (view: WisdomSyncView) => void
}) {
  let disposed = false
  let view = initialWisdomSyncView

  const run = async (retry: boolean) => {
    if (disposed || view.busy || (retry && (!view.snapshot?.can_retry || view.error))) {return}
    view = { ...view, busy: true, error: false }
    deps.changed(view)

    try {
      const snapshot = await (retry ? deps.retry() : deps.read())

      if (!disposed) {view = { snapshot, busy: false, error: false }}
    } catch {
      if (!disposed) {view = { ...view, busy: false, error: true }}
    }

    if (!disposed) {deps.changed(view)}
  }

  return {
    refresh: () => run(false),
    retry: () => run(true),
    dispose: () => {
      disposed = true
    }
  }
}
