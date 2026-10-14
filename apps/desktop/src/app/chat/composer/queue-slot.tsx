import type { ReactNode } from 'react'

import type { HermesGateway } from '@/hermes'
import type { Translations } from '@/i18n'
import { type QueuedPromptEntry, removeQueuedPrompt, unparkQueuedPrompts } from '@/store/composer-queue'
import { notifyError } from '@/store/notifications'

import type { QueueEditState } from './composer-utils'
import { cancelQueuedPrompt, discardLostPrompt } from './discard-lost-prompt'
import { QueuePanel } from './queue-panel'

export interface ComposerQueueSlotInputs {
  activeQueueSessionKey: null | string
  beginQueuedEdit: (entry: QueuedPromptEntry) => void
  busy: boolean
  drainNextQueued: () => Promise<unknown>
  exitQueuedEdit: (action: 'cancel' | 'save') => unknown
  gateway: HermesGateway | null | undefined
  queueEdit: QueueEditState | null
  queueParked: boolean
  queuedPrompts: QueuedPromptEntry[]
  sendQueuedNow: (id: string) => unknown
  sessionId: null | string | undefined
  steerQueuedNow: (id: string) => unknown
  t: Translations
}

/**
 * The queue node handed to the status stack: a QueuePanel wired to the
 * composer's queue engine, or null when there is nothing queued (the stack
 * drops the section entirely on a falsy node).
 */
export function renderComposerQueueSlot({
  activeQueueSessionKey,
  beginQueuedEdit,
  busy,
  drainNextQueued,
  exitQueuedEdit,
  gateway,
  queueEdit,
  queueParked,
  queuedPrompts,
  sendQueuedNow,
  sessionId,
  steerQueuedNow,
  t
}: ComposerQueueSlotInputs): ReactNode {
  return activeQueueSessionKey && queuedPrompts.length > 0 ? (
    <QueuePanel
      busy={busy}
      editingId={queueEdit?.entryId ?? null}
      entries={queuedPrompts}
      onDelete={id => {
        const serverStatus = queuedPrompts.find(entry => entry.id === id)?.serverStatus

        // A server-owned row is an admission in the authority's FIFO, not local state: Delete
        // withdraws it there (a lost `unknown` turn is acknowledged, exactly like Discard). The
        // pending fanout then retires the card; a local removal here would only hide live work.
        if (serverStatus && gateway) {
          const withdraw = serverStatus === 'unknown' ? discardLostPrompt : cancelQueuedPrompt

          withdraw(sessionId, activeQueueSessionKey, id, gateway.request.bind(gateway)).catch((error: unknown) =>
            notifyError(error, t.composer.queueCancelFailed)
          )

          return
        }

        if (removeQueuedPrompt(activeQueueSessionKey, id) && queueEdit?.entryId === id) {
          exitQueuedEdit('cancel')
        }
      }}
      onDiscardLost={
        gateway
          ? id => {
              // Server-owned row: the authority's pending fanout retires it
              // from the queue once the acknowledgement commits. Canonical
              // local sessions key the queue by their own session id, so the
              // stored key stands in when the runtime id is not bound yet.
              discardLostPrompt(sessionId, activeQueueSessionKey, id, gateway.request.bind(gateway)).catch(
                (error: unknown) => notifyError(error, t.composer.queueLostDiscard)
              )
            }
          : undefined
      }
      onEdit={beginQueuedEdit}
      onResume={() => {
        unparkQueuedPrompts(activeQueueSessionKey)

        // Idle → kick the head immediately; busy → the settle drain
        // takes over now that the park is lifted.
        if (!busy) {
          void drainNextQueued()
        }
      }}
      onSendNow={id => void sendQueuedNow(id)}
      onSteerNow={id => void steerQueuedNow(id)}
      parked={queueParked}
    />
  ) : null
}
