import { atom } from 'nanostores'

import type { PreviewUblockState } from '../../electron/preview-ublock'

const unavailableState = (enabled: boolean): PreviewUblockState => ({
  enabled,
  available: false,
  dashboardUrl: null,
  extensionId: null,
  operation: {
    failure: null,
    operationId: null,
    phase: 'idle',
    receivedBytes: 0,
    totalBytes: null
  },
  popupUrl: null,
  rulesetsReady: false,
  version: null
})

function normalizePreviewUblockState(value: Partial<PreviewUblockState> | null | undefined): PreviewUblockState {
  const enabled = value?.enabled === true
  const operation = value?.operation

  return {
    enabled,
    available: value?.available === true,
    dashboardUrl: typeof value?.dashboardUrl === 'string' ? value.dashboardUrl : null,
    extensionId: typeof value?.extensionId === 'string' ? value.extensionId : null,
    operation: {
      failure:
        operation?.failure &&
        typeof operation.failure.code === 'string' &&
        typeof operation.failure.message === 'string'
          ? {
              code: operation.failure.code as NonNullable<PreviewUblockState['operation']['failure']>['code'],
              message: operation.failure.message
            }
          : null,
      operationId: typeof operation?.operationId === 'string' ? operation.operationId : null,
      phase:
        typeof operation?.phase === 'string' ? (operation.phase as PreviewUblockState['operation']['phase']) : 'idle',
      receivedBytes:
        typeof operation?.receivedBytes === 'number' && Number.isFinite(operation.receivedBytes)
          ? Math.max(0, operation.receivedBytes)
          : 0,
      totalBytes:
        typeof operation?.totalBytes === 'number' && Number.isFinite(operation.totalBytes) && operation.totalBytes >= 0
          ? operation.totalBytes
          : null
    },
    popupUrl: typeof value?.popupUrl === 'string' ? value.popupUrl : null,
    rulesetsReady: value?.rulesetsReady === true,
    version: typeof value?.version === 'string' ? value.version : null
  }
}

export const $previewUblock = atom<PreviewUblockState>(unavailableState(false))
export const $previewUblockBlockedRequestCounts = atom<Record<number, number>>({})
let stateSubscriptionAttached = false
let blockedRequestSubscriptionAttached = false

function setPreviewUblockState(next: PreviewUblockState): void {
  $previewUblock.set(next)

  if (!next.enabled || !next.available || !next.rulesetsReady) {
    $previewUblockBlockedRequestCounts.set({})
  }
}

function setPreviewUblockBlockedRequestCount(update: unknown): void {
  const value = update as { blockedRequestCount?: unknown; webContentsId?: unknown } | null
  const webContentsId = value?.webContentsId
  const blockedRequestCount = value?.blockedRequestCount

  if (
    typeof webContentsId !== 'number' ||
    !Number.isInteger(webContentsId) ||
    typeof blockedRequestCount !== 'number' ||
    !Number.isFinite(blockedRequestCount) ||
    blockedRequestCount < 0
  ) {
    return
  }

  const counts = { ...$previewUblockBlockedRequestCounts.get() }

  if (blockedRequestCount === 0) {
    delete counts[webContentsId]
  } else {
    counts[webContentsId] = Math.floor(blockedRequestCount)
  }

  $previewUblockBlockedRequestCounts.set(counts)
}

export function getPreviewUblockBlockedRequestCount(webContentsId: number | null | undefined): number {
  return typeof webContentsId === 'number' ? ($previewUblockBlockedRequestCounts.get()[webContentsId] ?? 0) : 0
}

export async function loadPreviewUblock(): Promise<void> {
  if (typeof window === 'undefined' || typeof window.hermesDesktop?.previewUblock?.getState !== 'function') {
    return
  }

  if (!stateSubscriptionAttached && typeof window.hermesDesktop.previewUblock.onState === 'function') {
    stateSubscriptionAttached = true
    window.hermesDesktop.previewUblock.onState(next => setPreviewUblockState(normalizePreviewUblockState(next)))
  }

  if (
    !blockedRequestSubscriptionAttached &&
    typeof window.hermesDesktop.previewUblock.onBlockedRequestCount === 'function'
  ) {
    blockedRequestSubscriptionAttached = true
    window.hermesDesktop.previewUblock.onBlockedRequestCount(setPreviewUblockBlockedRequestCount)
  }

  try {
    setPreviewUblockState(normalizePreviewUblockState(await window.hermesDesktop.previewUblock.getState()))
  } catch {
    // Keep the last authoritative state if the main process is not ready.
  }
}

export async function setPreviewUblockEnabled(enabled: boolean): Promise<PreviewUblockState> {
  if (typeof window === 'undefined' || typeof window.hermesDesktop?.previewUblock?.setEnabled !== 'function') {
    throw new Error('Preview uBlock control is unavailable')
  }

  const next = normalizePreviewUblockState(await window.hermesDesktop.previewUblock.setEnabled(enabled))
  setPreviewUblockState(next)

  return next
}
