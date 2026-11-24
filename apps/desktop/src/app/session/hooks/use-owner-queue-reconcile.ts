import { useEffect } from 'react'

import { watchOwnerQueues } from '@/store/owner-queue-reconcile'

/** Re-read the owner's queue of chats holding owner work on every edge into an open gateway. */
export function useOwnerQueueReconcile(requestGateway: Parameters<typeof watchOwnerQueues>[0]): void {
  useEffect(() => watchOwnerQueues(requestGateway), [requestGateway])
}
