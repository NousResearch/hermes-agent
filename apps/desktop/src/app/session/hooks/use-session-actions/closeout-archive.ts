import type { ClientSessionState } from '@/app/types'
import { $pinnedSessionIds } from '@/store/layout'
import { notify, notifyError, RECOVERY_ACTIONS } from '@/store/notifications'
import { $activeSessionId, sessionPinId } from '@/store/session'
import {
  captureCloseoutTarget,
  requestSessionCloseout,
  type SessionCloseoutOptions,
  validateCloseoutReceipt
} from '@/store/session-closeout'
import { beginSessionMutation, endSessionMutation, tombstoneSessions } from '@/store/session-removal'
import { closeSessionTile, dropSessionState } from '@/store/session-states'
import { forgetSessionUnread } from '@/store/session-unread'
import { captureSessionViewer, deferCloseoutViewer } from '@/store/session-viewer-lifecycle'

import type { GatewayRequest } from '../use-prompt-actions/utils'

import { dropListedSession, findListedSession } from './utils'

interface ArchiveContext {
  requestGateway: GatewayRequest
  runtimeIdByStoredSessionIdRef: { current: Map<string, string> }
  sessionStateByRuntimeIdRef: { current: Map<string, ClientSessionState> }
  selectedStoredSessionIdRef: { current: string | null }
  startFreshSessionDraft: (replaceRoute: boolean) => void
  copy: { archiveFailed: string; archived: string }
}

const receiptIdleVersions = new WeakMap<object, number>()

function closingSelection(storedSessionId: string, runtimeId: string | undefined, context: ArchiveContext): boolean {
  return (
    context.selectedStoredSessionIdRef.current === storedSessionId ||
    Boolean(runtimeId && $activeSessionId.get() === runtimeId)
  )
}

export async function archiveWithCloseout(
  storedSessionId: string,
  options: SessionCloseoutOptions,
  context: ArchiveContext
) {
  const listed = findListedSession(storedSessionId)
  const row = listed?.session
  const ids = [storedSessionId, row?.id, row?._lineage_root_id]
  let runtimeId = context.runtimeIdByStoredSessionIdRef.current.get(storedSessionId)

  try {
    const target = captureCloseoutTarget(storedSessionId, context.requestGateway, options)
    runtimeId ??= target.runtimeId
    const viewer = runtimeId ? captureSessionViewer(runtimeId, target.destination.owner) : undefined
    const idleVersion = viewer?.idleVersion ?? 0
    const receipt = options.receipt ?? (await requestSessionCloseout(target, options))
    validateCloseoutReceipt(receipt, target)
    runtimeId ??= receipt.old.runtime_id ?? undefined

    if (runtimeId) {
      captureSessionViewer(runtimeId, target.destination.owner)
    }

    if (!receiptIdleVersions.has(receipt)) {
      receiptIdleVersions.set(receipt, idleVersion)
    }

    if (options.deferUi) {
      return receipt
    }

    if (runtimeId && !receipt.settled) {
      deferCloseoutViewer(runtimeId, receiptIdleVersions.get(receipt)!)
    }

    beginSessionMutation(ids)

    try {
      dropListedSession(storedSessionId)
      tombstoneSessions(ids)
      const pin = row ? sessionPinId(row) : storedSessionId
      $pinnedSessionIds.set($pinnedSessionIds.get().filter(id => id !== storedSessionId && id !== pin))
      forgetSessionUnread(ids, receipt.old.profile)

      if (!options.preserveView && closingSelection(storedSessionId, runtimeId, context)) {
        context.startFreshSessionDraft(true)
      }

      closeSessionTile(storedSessionId)

      if (runtimeId && receipt.settled) {
        context.runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
        context.sessionStateByRuntimeIdRef.current.delete(runtimeId)
        dropSessionState(runtimeId)
      }
    } finally {
      endSessionMutation(ids)
    }

    options.onApplied?.()
    notify({
      durationMs: 2000,
      kind: 'success',
      message: context.copy.archived,
      detail: `${receipt.receipt_id} · memory=${receipt.destinations.memory} · tasks=${receipt.destinations.tasks}`
    })

    return receipt
  } catch (error) {
    notifyError(error, context.copy.archiveFailed, { action: RECOVERY_ACTIONS.openGateways() })

    return undefined
  }
}
