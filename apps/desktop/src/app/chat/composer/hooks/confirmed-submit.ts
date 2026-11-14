import type { RefObject } from 'react'

import { enqueueQueuedPrompt, getQueuedPrompts } from '@/store/composer-queue'

import type { onComposerSubmitRequest } from '../focus'
import type { ChatBarProps } from '../types'

type NativeRequest = NonNullable<Parameters<Parameters<typeof onComposerSubmitRequest>[0]>[0]['native']>

/** Synchronous claim precedes any work so overlapping surfaces cannot send
 * twice. Only native acknowledgements resolve accepted/server-queued; a local
 * FIFO receipt says queued, never completed. The caller's draft is untouched. */
export function submitConfirmedText({
  busy,
  key,
  native,
  onSubmit,
  pending,
  sessionId,
  text
}: {
  busy: boolean
  key: null | string
  native: NativeRequest
  onSubmit: ChatBarProps['onSubmit']
  pending: RefObject<boolean>
  sessionId: null | string | undefined
  text: string
}): void {
  const finish = native.claim()

  if (!finish) {
    return
  }

  if (!key) {
    finish({ status: 'rejected' })

    return
  }

  if (pending.current || busy || getQueuedPrompts(key).length > 0) {
    const entry = enqueueQueuedPrompt(key, { text, attachments: [], confirmedExternal: true })
    finish(entry ? { status: 'queued', queueId: entry.id } : { status: 'rejected' })

    return
  }

  pending.current = true

  void (async () => {
    try {
      const accepted = await onSubmit(text, {
        attachments: [],
        composerScope: key,
        confirmedExternal: true,
        sessionId,
        storedSessionId: key,
        onExternalAccepted: queued => finish({ status: queued ? 'queued' : 'accepted' })
      })

      // A boolean alone is not proof of acceptance. The first settlement wins,
      // including an acknowledgement followed by a local cleanup failure.
      finish({ status: accepted === false ? 'rejected' : 'unknown' })
    } catch {
      finish({ status: 'unknown' })
    } finally {
      pending.current = false
    }
  })()
}
